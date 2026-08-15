# P1 — fake-quant harness

**状态：已完成（Exit Criteria 全部满足）** · 最后更新 2026-08-11
对应 `MID_GROUP_ROADMAP.md` §5 · 代码 `vllm/fakequant_lab/` · 报告
`reports/midgroup/P1_harness.md` · 环境 `ASCEND_RT_VISIBLE_DEVICES=1`

---

## 1. 当前状态

harness 写完并全部自检通过，能在**一次权重扫描内**跑完 QwQ-32B 的五组配置，
BF16 组与原模型逐位一致。Tier1/2 全量导出（1792 行），WikiText-2 PPL 与 GK 曲线已出。

**给 P2 的一句话结论：RTN 对称出局，但加上权重零点后缺口只剩 1.56 个百分点，
方案仍然活着。** 完整 WikiText-2 test split（150 窗口 / 307k token）上，
`W4A8-mg1024` 相对 W8A8 是 **+9.03%**（对称 RTN）→ **+3.56%**（非对称 RTN），
门槛 ≤2%。

**GK 不是杠杆，AWQ 也不是**：GK 1024→128（组数 8 倍，P0 判定买不起）只赚回约 1/3 的
缺口；逐通道缩放两个方向都只值 0.3–0.5 dB（D6）。机理是 **int4 权重的难度是行内/组内
的无结构散布**，mid-group 和逐通道缩放都作用在不是瓶颈的那一侧。**而无结构散布正是
GPTQ 的适用条件。**

**P2 的路线：非对称权重设为默认 + GPTQ 补最后 1.56 个百分点（≈0.35 dB Tier-2）。**

Exit Criteria 核对：

- [x] BF16 组与原模型逐位一致（`max|diff| = 0.000e+00`，插桩无副作用）
- [x] 同一份权重产出五组配置的输出
- [x] 单 GEMM 级 SNR 可导出（Tier 1/2，`results/tier12_gk1024.csv`）

---

## 2. 已完成

### 2.1 代码

`vllm/fakequant_lab/`，7 个模块 + 2 个自检脚本。分工见该目录 `README.md`。

| 层 | 文件 | 关键点 |
|---|---|---|
| 量化原语 | `quant.py` | 分组对称/非对称，**scale 强制 bf16 且先 round 再量化**，`w_ksum`、MSD 拆分、group-major 导出 |
| GEMM | `fake_linear.py` | 与 kernel 数学等价的分组精确累加；五组配置表；`PatchedLayer` 上下文管理器 |
| 引擎 | `model_stream.py` + `runner.py` | 逐层流式，多配置锁步推进，PPL + 逐层 SNR + KL |
| 导出 | `run_tiers.py` / `run_ppl.py` | Tier1/2 CSV、PPL JSON |

### 2.2 自检（`run_selftest.py` T1–T6，`run_selftest_model.py` M1–M4）

全过。最要紧的两条：

- **T4**：分组 fp32 GEMM 与 **CPU int64 精确参考**逐 shape 比对，5 组 shape
  （含非对称）最大相对误差 **8e-8**——「fake quant 与 kernel 数学等价」不是声明，
  是验过的。
- **M1**：streaming 引擎的 BF16 logits 与 `Qwen2ForCausalLM.forward`
  **逐位相同**（`max|diff| = 0.000e+00`）。

### 2.3 数据

| 产物 | 内容 |
|---|---|
| `fakequant_lab/results/tier12_gk1024.csv` | Tier1/2，64 层 × 7 Linear × 4 配置 = 1792 行 |
| `fakequant_lab/results/gksweep_w8.json` | GK ∈ {128,256,512,1024} 的 PPL 曲线 |
| `fakequant_lab/results/smoke_ppl_gk1024_w8.json` | 五组配置 + 非对称，8 窗口 |
| `fakequant_lab/results/ppl_gk1024_full.json` | 完整 test split（150 窗口 / 307k token） |

结论表在 `reports/midgroup/P1_harness.md`，此处不复制。

---

## 3. 待办

- [ ] **P2 的第一件事：把非对称权重定为默认**（D3），并让 P3 知道 per-token 量化
      kernel 要多吐一个 `AS[m,g]`。这一步值 5.5 个 PPL 百分点，是 P1 最大的发现。
- [ ] **然后 GPTQ**（`group_size = GK`），不要再花机时在 RTN 对称上。
      目标：**Tier-2 再涨约 0.35 dB**（把 +3.56% 压到 ≤2%）。
- [x] ~~AWQ 前置探针~~ — 已跑，见 D6。结论：不做主力，但缺口只剩 1.56 个百分点后
      它的 0.3–0.5 dB 已不可忽略，留作 GPTQ 之后的补齐手段。
- [ ] 完整 split 上的 GK 曲线重跑（`results/gksweep_full.json`）——8 窗口那版
      量级偏悲观近一倍，见 D7。
- [ ] **把非对称权重当成正式候选评估**（见 D3）。P3 若要支持，需要 per-token 量化
      kernel 额外吐一个每组的激活行和 `AS[m,g]`。
- [ ] 任务评测（GSM8K + 一项中文）尚未接入——本阶段只做了 PPL。lm_eval 未安装。
      这是 P2 的门禁项（路线图 §6 说三项冲突时以任务评测为准）。
- [ ] `--mode fast` 只在合成数据上量过（比 exact 低 27 dB），真实模型上没有对拍过。
      目前所有数字都用 `exact`，要用 fast 跑大扫描前先补一次对拍。

---

## 4. 决策与坑

### D1 · fake quant 走 fp32 matmul，而不是「反量化成 bf16 再 matmul」

常见做法是把 A、W 反量化回 bf16 做一次普通 matmul。**在 910B4 上这条路会污染结果**：
实测 bf16 matmul 对整数输入的输出会被 round 到 8 位尾数（探针：K=5120 的整数 GEMM
最大误差 256，参考值量级 1.2e5）。改成喂**整数**给 fp32 matmul、scale 在外面乘：
算子是小整数（在 fp32 与 cube 的 hf32 输入格式里都精确），`|partial| < 2^24`，
所以累加精确——**等价于 cube 的 int32 累加**。

代价只有 3.5×（2048×5120×5120：1.63 vs 0.46 ms），完全负担得起。这条是整个 harness
可信度的地基，由 T4 守住。

### D2 · scale 必须先 round 到 bf16 再拿去量化

不是「量化完再把 scale 存成 bf16」。kernel 里 scale 就是 bf16，顺序反了会系统性
高估精度。T1 用「拿返回的 bf16 scale 重新量化，码字必须逐位相同」来守这条。

### D3 · 加了一个非冻结的非对称权重选项（opt-in）——**这是 P1 最大的发现**

路线图 §5 冻结的是对称 int4。但本轮数据显示瓶颈在权重侧、而缩 GK 对权重只值 1.5 dB，
于是量了非对称。完整 split 上它把 **+9.03% 一步压到 +3.56%**（5.5 个百分点），
比对称缩到 GK=128（组数 8 倍）买到的还多，而按 P0 的代价表 GK 从 1024 缩到 256
就要 2.4–4.1× kernel 时间。

kernel 代价：`Σ A_q(W_q − wz) = Σ A_q·W_q − wz[n,g]·Σ_{k∈g} A_q[m,k]`，只多一个
每组激活行和（per-token 量化 kernel 顺手产出）+ AIV 一个秩一修正。它确实落在
P0 §2.5 定位的 AIV 瓶颈上，不免费，但远便宜于把组数翻 4 倍。

**默认关闭**（`--asym` / `build_configs(include_asym=True)`），默认跑的仍是冻结的五组。

### D4 · 逐层流式，不用 device_map / accelerate

QwQ-32B 是 62 GB，要跑五组配置。一层常驻只要 ~1 GB，于是五组配置的 hidden states
共享一张卡、共享**一次**权重扫描（否则要读五遍 62 GB）。副产品：逐层 SNR（Tier 3）
免费拿到，也不用赌 accelerate 在 NPU 上的设备放置。

### D5 · 判据不能从 INT8 研究直接继承

`snr_midgroup_sweep.py` 的 32/35 dB 是在**预训练模型 + INT8**上定的。现成 QwQ-32B
上**连 `w8a8` 自己都过不了**（T1-X 最差 14.53 dB、T2 最差 9.40 dB），可它 PPL 无损。
**P2 的门禁改用 PPL / 任务评测、以 W8A8 为参照系**；Tier1/2 降级为定位问题层的
诊断工具（这个用途上它很好用，直接指出了「瓶颈在权重侧」）。

`logit SNR` 同样不可用（`w8a8` 13.3 dB 却无损，误差是 softmax 会抵消的共模分量）。
runner 因此改为额外输出 **KL(bf16 ‖ config)**。

### D7 · 小样本会**系统性高估**退化，不是噪声——门禁必须跑完整 split

先用 8 窗口（16k token）跑了一轮，与完整 150 窗口差了近一倍：
`w4a8-mg1024` +18.67% → **+9.03%**，非对称 +11.02% → **+3.56%**。

不是方差：前 8 个窗口的 BF16 PPL 是 **4.89**，全集是 **6.66**——开头那段文本明显更容易，
模型更自信，量化噪声的相对伤害就更大。**换种子救不回来。** 完整 split 一轮约 53 分钟
（6 配置 × 150 窗口 × 64 层），别为了省这个时间用子集下结论。

### D6 · AWQ 探针跑过了：不做主力，但别扔掉

逐输入通道缩放（`W·diag(s)` / `X·diag(1/s)`，推理时免费）两个方向都试了：
`s = mean|X_j|^α`（AWQ 方向，W 更难 A 更容易）与 `s = rms(W[:,j])^-β`（镜像方向）。
全 64 层，**连 AWQ 自己的逐 Linear 网格搜索也只有 +0.25～0.47 dB 中位增益**，
纯 AWQ 方向 α=0.5 甚至是 **−0.85～−1.43 dB**。

机理由结构诊断给出：**激活是强通道结构化的（逐通道离散度 10–27×），权重几乎是平的
（1.04–1.77×）**。逐通道缩放只能在通道之间搬难度，而 int4 权重的难度是行内/组内的
无结构散布。

**但同一份数据正面支持 GPTQ**：无结构散布正是 GPTQ 的适用条件（逐列量化 + 把误差
补偿到未量化的权重上）。所以探针的价值不只是排除了 AWQ，还确认了 GPTQ 是对的工具形状。

**注意别读过头**：0.3–0.5 dB 在缺口是 9 个百分点时可以忽略，但完整 split 显示缺口
只剩 **1.56 个百分点**（D7），这时候它已经不可忽略了。结论是「不做主力」，不是「扔掉」。

数据：`fakequant_lab/results/awq_probe_gk1024.csv`（448 行）。

### 坑 · `load_state_dict(assign=True)` 会把权重包成 requires_grad=True 的 Parameter

调用方一旦忘了 `no_grad`，autograd 图会把**整条流水上所有层**的权重留住——诊断脚本
上实测 32 层后就 OOM（59 GB active）。`load_layer` 现在无条件
`layer.requires_grad_(False)`。

### 坑 · `with ... as cap` 结束后 `cap` 仍然绑定

`_Capture` 持有 layer 引用和捕获的激活，`with` 块结束不会释放，会多钉住一整层。
`__exit__` 里显式清掉。

### 坑 · NPU 的 `aclnnArange` 没有 int8 kernel

MSD 全值域自检要在 CPU 上造 `arange(-128,128)` 再搬到 NPU。

### 坑 · `torch.sum` 对整型会提升到 int64

`w_ksum` 要显式 `.to(torch.int32)`，kernel 要的是 int32。
