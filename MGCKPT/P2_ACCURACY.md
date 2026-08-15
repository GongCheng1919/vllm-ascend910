# P2 — 算法选型与精度门禁

**状态：待完成（暂缓，2026-08-13 起与 P3 并行）** · 最后更新 2026-08-13
对应 `MID_GROUP_ROADMAP.md` §6 · 代码 `vllm/fakequant_lab/` · 环境 `ASCEND_RT_VISIBLE_DEVICES` 见下

---

## 1. 当前状态

PPL 门禁（RTN）与 GPTQ 评测均已完成，GLUE 已出第一版结果，**验收门槛待定**（§3）。

- **PPL（完整 split）**：`w4a8-mg1024-asym`（RTN）相对 W8A8 **+3.52%**，过 ≤5% 门禁 ✅；
  `w4a8-pc` +33.8%、对称 +9.0% 出局。
- **GPTQ**：标定完成（asym/sym 各 64/64 层）并评完 PPL——**无端到端增益，非对称档反而更差**
  （+4.98% vs RTN 的 +3.52%），测试集逐层 SNR 增益 ≈0。**用户决定暂缓 GPTQ**（D7）。
- **GLUE（limit 500/任务，0-shot）**：w8a8 vs bf16 基本无损（−0.1 点）；**asym vs w8a8 平均
  −4.5 个百分点**（cola −11.7 / qqp −13.0 崩盘），未达路线图原定任务门槛（≤1 点）。
- **验收标准**：PPL 已过；任务门槛待用户按实测数据拍板（§3 第一个待办）。

### 一句话给接手的人

> 主线 = **GK=1024 + 非对称（RTN），已过 PPL 门禁**；GPTQ 这版不带来收益，暂缓。
> GLUE 显示任务退化比 PPL 严重（平均 −4.5 点，cola/qqp 崩），**门槛数字没定**。
> 剩下：定门槛 → GSM8K（batch）→ 一项中文 → 冻结验收标准。
>
> **2026-08-13：本阶段暂缓，P3 已并行开工（D8）。P2 不是"已通过"，是"没跑完且当前
> 唯一的任务指标没过"。** 谁来接手 P2，先读 D8 再读 §3。

---

## 2. 已完成

### 2.1 门槛变更（用户决定，2026-08-11）

PPL 门槛从 **≤2%** 放宽到 **≤5%**（相对 W8A8）。理由：+5% 对应 BF16 PPL 5.0 → 5.25，
对 W4A8 而言是可接受的工程标准。**记录在案是因为路线图 §6 原本写了「开跑前写死，
不得事后调整」——这是一次明示的、有理由的变更，不是事后找补。**

**但 PPL 不是唯一门禁**：路线图 §6 自己写了「若三项指标冲突，以任务评测为准」，
QwQ-32B 是推理模型，PPL 只测单步 next-token，测不出误差沿几千 token 推理链累积。
任务评测仍未完成（§3）。

### 2.2 GPTQ 实现（`gptq.py` / `run_gptq.py`）

逐列量化 + 逆 Hessian 误差补偿，group params 走 `quant.group_params`（bf16 scale
规则与 kernel 一致），支持非对称。**自检 T7**：合成数据 +3.22 dB、**H=I 时逐位退化为
RTN**、非对称再叠 +1.29 dB。

顺序标定：每层用**已量化的前序层**产出的激活标定；q/k/v 共享 Hessian，gate/up 共享；
逐层落盘可断点续跑。

**真实标定集上的增益：+1.3 ~ +6.1 dB**（层输出 SNR，相对 RTN）。
⚠️ 冒烟时（4 窗口）看到 +10~20 dB 是**假的**——8192 token 对 `down_proj` 的 K=27648
是秩亏 Hessian，那是样本内过拟合。**必须用 ≥128 窗口。**

**最终结果（128 窗口标定，完整 split PPL）**：asym-GPTQ +4.98% vs RTN +3.52%、sym-GPTQ +7.94% vs
RTN +9.09%——非对称档端到端反而变差，测试集逐层 SNR 增益 ≈0（+0.03 dB）。**GPTQ 暂缓（D7）**。

### 2.3 fake-quant 模式重新定义（用户决定）

原来默认 `exact`（复刻 kernel 整数算术）。用户指出 fake quant 只需算法一致，不必复刻
计算过程。现在：

| 模式 | 含义 | 成本 |
|---|---|---|
| **`dequant`（默认）** | 反量化成 bf16 + 单次 bf16 matmul，计算上就是 BF16 模型 | 1.1–2.5× bf16 |
| `kernel` | 复刻 AscendC 的分组整数算术，**降级为验证工具** | 8–19× bf16 |

**`dequant` 已在完整 split 上对拍通过**（§2.4）。旧名 `fast`/`exact` 保留为别名。

### 2.4 关键数据（完整 WikiText-2 test split，150 窗口 / 307k token）

`results/ppl_gk1024_full.json`（kernel）、`ppl_gk1024_full_dequant.json`（dequant）：

| 配置 | ΔPPL vs w8a8 (kernel) | ΔPPL vs w8a8 (dequant) | 差 |
|---|---:|---:|---:|
| `w8a8-mg1024` | −0.098% | −0.150% | 0.05 |
| `w4a8-mg1024` | +9.028% | +9.094% | 0.07 |
| **`w4a8-mg1024-asym`** | **+3.560%** | **+3.523%** | **0.04** |
| `w4a8-pc` | +33.85% | +33.68% | 0.17 |

门禁效应量是 1–9 个百分点，`dequant` 只移动 0.04–0.17——**可以放心用于任务评测**。

### 2.5 完整 split 的 GK 曲线（RTN）

`results/gksweep_full.json`：

| GK | 对称 | 非对称 |
|---:|---:|---:|
| 128 | +5.58% | +3.02% |
| 512 | +8.33% | **+6.04%** ⚠️ |
| 1024 | +9.03% | **+3.56%** |

**GK=512 非对称的反常是真的**，见 D2。

### 2.6 任务评测链路（`resident.py` / `run_tasks.py`）

- `resident.py`：accelerate `device_map` 跨卡常驻（62 GB 需 ≥2 张卡），权重就地换成
  量化-反量化版本，激活用 forward-pre-hook。**模块图不变**，`generate()` / KV cache /
  lm_eval 的 HF wrapper 全部原样可用。
- `run_tasks.py`：接 lm_eval，GLUE / GSM8K / C-Eval / CMMLU 走同一路径。
- **已端到端跑通**（bf16 + GSM8K，2 张卡）。
- **GLUE 实测**（2026-08-12，`results/tasks_glue_{bf16,w8a8,asym}.json`，limit 500/任务，0-shot）：
  w8a8 相对 bf16 平均 −0.1 个百分点（无损）；**asym 相对 w8a8 平均 −4.5 个百分点**（8 任务主指标：
  cola mcc −11.7、qqp acc −13.0 崩盘，其余 −0.3~−5）。结论：任务退化比 PPL 严格得多，
  门禁口径应以任务为准（路线图 §6 原话），门槛数字待定（§3）。

---

## 3. 待办

- [x] **GPTQ 标定 + PPL 评测**（2026-08-11 完成）——结论：无端到端增益，**暂缓**（D7）。
- [x] **GLUE**（limit 500/任务，bf16/w8a8/asym 三配置）——asym vs w8a8 平均 **−4.5 个百分点**。
- [ ] **定任务验收门槛**（用户原话"用任务分退化反过来定"）——GLUE 数据已到手，选项见
      `int4_cube_lab/P2_REPORT_W4A8_ACCURACY.md` §9，拍板后写进本文件并冻结。
- [ ] **GSM8K**（生成，对推理模型最有信息量）——**必须开 batch**：未批处理只有 4.2 token/s，
      200 题 × 1024 token 要 13 小时；`--batch-size 8~16` 应接近线性提速。
- [ ] **一项中文**（C-Eval / CMMLU，路线图 §6 要求）。
- [ ] 更新 `reports/midgroup/P2_accuracy.md` 与路线图 §6（把冻结的门槛写进去）。

---

## 4. 决策与坑

### D1 · PPL 门槛 2% → 5%（用户决定）

见 §2.1。**任务评测仍是最终判据**，门槛未定，正是待办里要测的。

### D2 · GK=512 非对称的非单调是**真实且可复现**的，不是 bug

排查过三件事，全部排除了实现问题：

1. **量化器本身逐层验证完全单调**：每层每个投影都满足
   asym128 (19.4–19.8 dB) > asym512 (17.7–18.3) > asym1024 (16.9–17.7)，且 asym > sym。
2. **runner 是确定性的**：`mg1024-asym` 在两次独立运行里 PPL 完全相同（6.9092）。
3. **换模式独立复现**：`mg512-asym` 在 kernel 模式 +6.039%、dequant 模式 +6.010%。

所以是真实现象：**量化器在 GK=512 上更优，端到端却更差**。机理未解释（大概率是误差
方向与网络的相互作用，SNR 只测幅度不测方向）。

**实用推论：不能用「组更细一定更好」来选 GK，必须实测。** 而 P0 唯一买得起的
**GK=1024 恰好是三个非对称测点里最好的**——P0 与 P2 之间原本的张力消失了。

### D3 · GPTQ 的 Cholesky 必然落到 CPU

`torch.linalg.cholesky` 在 torch_npu 里没有 kernel，会**静默 fallback 到 CPU**（只有一条
warning）。`inverse_hessian` 把它显式化，避免每次调用触发 fallback 抖动。
`down_proj` 的 K=27648 那三步 K³ 是单层最大开销（约 75 s）。

**坑**：`H.double()` 在 NPU 上会被静默降级回 float32（也只有 warning），所以必须
**先 `.cpu()` 再转 fp64**，否则 `--fp64-hessian` 根本不生效。

### D4 · 标定集用 C4，不是 wikitext-2 train

用 wikitext-2 train 标定再评 wikitext-2 test 是同域的，会美化结果——用它定验收门槛是
错的。`--calib c4`（默认）从 `/mnt/local_datasets/pretrain/c4/en` 随机取 2048-token 片段，
这也是 GPTQ/AWQ 论文的惯例。

### D5 · 不引入外部 GPTQ 实现（评估过，暂不做）

环境里没有可用的 NPU 端 GPTQ：CANN 8.5 的 `msmodelslim` 只是**空的命名空间目录**
（无子模块），也没装 AutoGPTQ/GPTQModel。GPTQ 是纯离线权重变换，去 GPU 端跑原理上
等价，但用现成工具有两个坑会让产物**直接不可用**：

- **`desc_act` / act_order 必须关掉**：它重排输入通道，而 kernel 每 GK 列冲刷一次 L0C，
  **要求 K 分组是连续切片**。用 desc_act 量出来的 checkpoint 我们没法用。
- **scale 必须 bf16**：标准工具存 fp16，会让 fake quant 与真实 kernel 对不上。

自研实现实测 ~215 s/层（两作业共享 CPU 时），全模型约 3.8 小时，与 32B 模型在 GPU 上
跑 GPTQ 同量级。**等需要频繁迭代 GPTQ 变体时再考虑引入依赖。**

### D6 · 生成类评测的吞吐

未批处理 **4.2 token/s**（2 张卡，62 GB bf16）。解码是访存瓶颈，必须开 batch。
`kernel` 模式在 M=1 时是 19× bf16，**绝不能用它做生成评测**——这也是 §2.3 改默认的
直接动机之一。

### D7 · GPTQ 无端到端增益（用户决定暂缓）

128 窗口标定（C4）跑完 PPL：asym-GPTQ +4.98% vs RTN +3.52%（变差）、sym-GPTQ +7.94% vs +9.09%
（改善但远不够）。测试集逐层 SNR：asym 增益均值 +0.03 dB（标定集声称 +1.3~6.1 dB）——样本外退化。
KL/logit SNR 变好但 PPL 变差，与 D2 同构（SNR 只测幅度不测方向）。可能 128 窗口对 32B 模型仍不足。
**决定：GPTQ 暂缓，主线维持 RTN 非对称；任务评测不再包含 GPTQ 配置。**

### D8 · P2 暂缓、P3 并行开工（用户决定 2026-08-13）

**决定**：把 P2 标为**待完成**，不再阻塞 P3；P3 以 `W4A8-mg1024-asym` 为基准开工。

**理由**（用户原话大意）：NPU 不适合做量化算法迭代，P2 剩下的都是慢实验
（GSM8K 生成、中文任务、门槛拍板），而 P3 是 kernel 工程，两者无依赖。且优化本来
就要针对非对称算法，没有必要等 P2 定案。

**这是对路线图 §6「P2 不过就不做 P3」的明示偏离**，与 D1 同性质：有理由、记录在案，
不是事后找补。风险与回退路径写在 `P3_KERNEL.md` D2。

**必须保留的判断**：本阶段**没有通过**。PPL 过了（+3.52% ≤ 5%），但路线图 §6 自己写了
「若三项指标冲突，以任务评测为准」，而唯一测过的任务指标 GLUE 是
**asym vs w8a8 平均 −4.5 个百分点**。**任何地方都不要把 P2 表述成"已通过"或
"精度已验证"** —— 它是被挂起的 go/no-go。

**恢复 P2 时的入口**：§3 的三项待办不变（定门槛 → GSM8K batch 8~16 → 一项中文）。
若最终判 no-go，回退目标是 **W8A8-mg（−0.10%）**，P3 的分组冲刷流水与激活量化 kernel
可原样复用。

### 坑 · lm_eval 的安装

直接 `pip install lm_eval` 会在构建 `rouge-score` 时失败。解法：
先 `pip install --upgrade setuptools wheel absl-py nltk six`，
再 `pip install --no-build-isolation rouge-score`，最后 `pip install lm_eval`。
装好是 **0.4.12**。

---

## 5. 作业状态（2026-08-12）

- GPTQ 标定（asym/sym 各 64/64 层）**已完成**，PPL 已评完（§2.2 / D7），**不再续跑**。
- GLUE 三配置评测**已完成**（`results/tasks_glue_*.json`）。
- 无在跑作业。下一步排队：定门槛 → GSM8K（batch 8~16）→ 一项中文。

（若需重跑 GPTQ：`run_gptq.py` 断点续跑——目录里已存在的 `layer_NNN.safetensors` 会被跳过。）

```bash
cd vllm && export ASCEND_RT_VISIBLE_DEVICES=4
python -u -m fakequant_lab.run_gptq --gk 1024 --asym --calib c4 --nsamples 128 \
    --out fakequant_lab/gptq_gk1024_asym
```

跑完后的第一件事：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python -u -m fakequant_lab.run_ppl --windows 0 \
    --gptq w4a8-mg1024-asym-gptq=fakequant_lab/gptq_gk1024_asym \
    --gptq w4a8-mg1024-sym-gptq=fakequant_lab/gptq_gk1024_sym \
    --out fakequant_lab/results/ppl_gptq_full.json
```

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| `fakequant_lab/results/ppl_gk1024_full.json` | 五组 + 非对称，kernel 模式，完整 split |
| `fakequant_lab/results/ppl_gk1024_full_dequant.json` | 同上，dequant 模式（模式对拍） |
| `fakequant_lab/results/gksweep_full.json` | GK ∈ {128,512,1024} × 对称/非对称，完整 split |
| `fakequant_lab/results/ppl_gk512_recheck.json` | GK=512 反常的独立复现 |
| `fakequant_lab/results/tier12_gk1024.csv` | Tier1/2，1792 行（P1） |
| `fakequant_lab/results/awq_probe_gk1024.csv` | 逐通道缩放探针，448 行（P1） |
| `fakequant_lab/gptq_gk1024_{sym,asym}/` | GPTQ 标定权重（已完成；结论见 D7，暂缓） |
| `fakequant_lab/results/ppl_gptq_{asym,sym}_dequant.json` | GPTQ 的 PPL 实测（dequant，完整 split） |
| `fakequant_lab/results/tasks_glue_{bf16,w8a8,asym}.json` | GLUE 任务评测（limit 500/任务，0-shot） |
