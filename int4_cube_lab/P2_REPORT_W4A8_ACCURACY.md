# P2 — W4A8 精度门禁报告（QwQ-32B · WikiText-2 完整 split）

日期: 2026-08-11 · 模型: QwQ-32B（64 层，hidden=5120）· 环境: Ascend 910B4 + CANN 8.5.0 + torch_npu 2.7.1
Lab: `vllm/fakequant_lab/` · 检查点: `MGCKPT/P2_ACCURACY.md`（本报告是它的正式版）· 路线图: `MID_GROUP_ROADMAP.md` §6

## 结论（一句话）

**`w4a8-mg1024-asym`（RTN，GK=1024 + 非对称 + zero point）相对 W8A8 的 PPL 退化只有 +3.52%，
已过 ≤5% 门禁**；per-channel（`w4a8-pc`）出局（+33.8%）；mid-group 对称也出局（+9.0%）。
GPTQ 标定正在跑（约 50% 层），用来把余量做厚并为下游任务评测留空间；最终验收门槛将由
任务评测的退化幅度反过来定（见 §6）。

## 1. 范围与方法

- **模型/数据**：QwQ-32B 全模型（64 层），WikiText-2 **test split 完整跑完**（150 窗口 / 307,200 token，
  seq=2048）。这是 P1 冒烟（4–16 窗口）之后的正式门禁数据。
- **配置**（同一份 checkpoint，一次 pass 全跑）：

| 配置 | W | A | 备注 |
|---|---|---|---|
| `bf16` | — | — | 对照 |
| `w8a8` | int8 per-channel | int8 per-token | **门禁基准** |
| `w8a8-mg1024` | int8 mid-group | int8 mid-group | |
| `w4a8-mg1024` | **int4** mid-group | int8 mid-group | 对称 |
| `w4a8-pc` | int4 per-channel | int8 per-token | |
| `w4a8-mg1024-asym` | int4 mid-group + **zero point** | int8 mid-group | 非对称 |

- **两种算术模式对拍**：`dequant`（默认）与 `kernel`（逐位复刻），量化算法两者完全相同——
  详见下面的「两种算术模式是什么」。
- **门禁口径**：ΔPPL vs `w8a8`（kernel 模式），门槛 **≤5%**（用户决定从 ≤2% 放宽，见 §7 D1）。

### 两种算术模式：dequant 与 kernel（为什么对拍可信）

**共同前提**：模式只影响"怎么算"，不影响"量成什么样"——位宽、分组边界（GK）、bf16 scale、
zero point 在两种模式下完全一样，**决定模型质量的是量化算法，不是模式**（`fake_linear.py` 头注释）。

**`dequant`（默认）——普通 fake quant**：

```python
y = F.linear(quantize(x, cfg.a).dequantize().to(torch.bfloat16), self.w_deq)   # fake_linear.py:147
```

激活量化成 int8/int4 后**反量化回 bf16**，权重在初始化时提前反量化成 bf16 存一次（`w_deq` buffer），
之后每个 forward 就是**一次普通 bf16 matmul**——计算上就是一个 BF16 模型（实测 1.1–2.5× bf16 成本），
解码速度正常，**任务评测用它**。它唯一引入的失真：反量化操作数要四舍五入到 bf16（真实 kernel 不
做这步），自检 T6 实测该噪声比量化噪声本身低 ~27 dB（约占总噪声功率 0.2%），可忽略。

**`kernel` —— 逐位复刻 AscendC kernel 的分组整数算术**：

```python
for g in range(G):                                  # 按 K 分组循环（kernel 每 GK 列冲刷一次 L0C）
    partial = A_q[:, g] @ W_q[:, g].T               # 精确整数和
    if w_zero is not None:
        partial -= A_q[:, g].sum(dim=1, keepdim=True) * w_zero[:, g]   # 非对称的 rank-1 修正
    acc += partial * as[:, g] * ws[:, g]            # 组末乘 scale，fp32 累加
```

保持**整数码**参与乘法、**每组累加完再乘 scale**——与真实 kernel 的流水一致。两个细节：

- 为什么 fp32 matmul 能模拟 int32 精确和：`|A| ≤ 127, |W| ≤ 8`，组内部分和 `|partial| ≤ GK·127·8 ≈ 1e6 < 2^24`，
  fp32 精确表示；自检 T4 用 CPU int64 参考核对过。
- 非对称的 `- wz·ΣA_q` 项对应 zero point 的 rank-1 贡献：`Σ A_q·(W_q−wz) = Σ A_q·W_q − wz[n,g]·Σ_k A_q[m,k]`，
  其中激活行和 `Σ_k A_q[m,k]` 由 per-token 量化 kernel 在写 int4 平面的同一趟里产出。

代价：每个 Linear 要做 **G 次 matmul**，实测 **8–19× bf16**（M=1 时 19×）→ **只当验证工具，
绝不用它做生成评测**。

| | dequant | kernel |
|---|---|---|
| 算法 | 反量化 → 一次 bf16 matmul | 分组整数 GEMM + fp32 重缩放（逐位同 kernel） |
| 失真 | 仅 bf16 舍入（噪声功率的 ~0.2%，T6） | 无（T4 对 CPU int64 核对） |
| 成本 | 1.1–2.5× bf16 | 8–19× bf16 |
| 用途 | PPL 与下游任务评测 | 验证 dequant 没有歪曲算法效果 |

**结论**：完整 split 上两模式 ΔPPL 只差 **0.04–0.17 个百分点**，而门禁效应量是 1–9 个百分点——
dequant 可以放心代表 kernel，后续 GLUE/GSM8K 任务评测全部走 dequant 模式。
（旧名 `fast`/`exact` 是这两个模式的别名，见 `fake_linear.py:46`。）

## 2. 完整 split PPL（RTN 基线，最终数据）

`results/ppl_gk1024_full.json`（kernel）· `ppl_gk1024_full_dequant.json`（dequant）：

| 配置 | PPL (kernel) | ΔPPL vs w8a8 | PPL (dequant) | ΔPPL (dequant) | 两模式差 |
|---|---:|---:|---:|---:|---:|
| `bf16` | 6.6605 | −0.17% | 6.6605 | −0.16% | 0.00 |
| `w8a8` | 6.6716 | — | 6.6713 | — | — |
| `w8a8-mg1024` | 6.6651 | −0.10% | 6.6613 | −0.15% | 0.05 |
| `w4a8-mg1024` | 7.2739 | **+9.03%** | 7.2780 | +9.09% | 0.07 |
| `w4a8-pc` | 8.9297 | **+33.85%** | 8.9181 | +33.68% | 0.17 |
| **`w4a8-mg1024-asym`** | **6.9092** | **+3.56%** ✅ | 6.9064 | +3.52% | 0.04 |

要点：

1. **当前最好一档 `w4a8-mg1024-asym` 已过 ≤5% 门禁**（+3.56%）。+5% 对应 BF16 PPL 5.0 → 5.25，
   对 W4A8 是可接受的工程标准。
2. **`w4a8-pc` 出局**：+33.8%。per-channel 每列一个 scale 在 int8 激活下不够——权重必须分组（mid-group）
   才能压住误差。这从算法侧印证了 P0 的结论：**mid-group 是 W4A8 的唯一现实选项**。
3. **非对称（zero point）是关键**：同一 GK=1024，加 zero point 把 +9.03% 压到 +3.56%。
4. **`dequant` 与 `kernel` 只差 0.04–0.17 个百分点**，而门禁效应量是 1–9 个百分点——
   两种模式等价，任务评测放心用便宜的 `dequant`。

## 3. 误差的细粒度信号（kernel 模式）

PPL 只反映均值；KL 与 logit SNR 看误差质量：

| 配置 | KL vs bf16 | logit SNR (dB) |
|---|---:|---:|
| `w8a8` | 0.064 | 14.12 |
| `w8a8-mg1024` | 0.018 | 19.37 |
| `w4a8-mg1024` | 0.242 | 8.60 |
| `w4a8-pc` | 0.493 | 5.12 |
| **`w4a8-mg1024-asym`** | **0.180** | **9.88** |

`w4a8-pc` 的 KL 是 `w4a8-mg1024-asym` 的 2.7 倍、logit SNR 低 4.8 dB——与 PPL 结论一致。
有趣的是 `w8a8-mg1024` 比 `w8a8` 还好（KL 0.018 vs 0.064，SNR 19.4 vs 14.1）：mid-group 的
per-group scale 对激活也适用，且在这里帮了忙（`w8a8` 的 A 是 per-token 量化）。

## 4. GK 曲线（RTN，完整 split）

`results/gksweep_full.json`（kernel 模式）：

| GK | 对称 | 非对称 |
|---:|---:|---:|
| 128 | +5.58% | **+3.02%** |
| 512 | +8.33% | **+6.04%** ⚠️ |
| 1024 | +9.03% | **+3.56%** |

1. **非对称全面优于对称**（每个 GK 都低 2.3–5.5 个百分点）。
2. **GK=512 非对称是非单调的**（+6.04% > GK=1024 的 +3.56%），且**真实可复现**（详见 §7 D2）：
   量化器本身逐层完全单调（asym128 > asym512 > asym1024 的 SNR），端到端却更差。
3. **实用推论：不能拿"组更细一定更好"来选 GK，必须实测。** 而 P0 在 kernel 侧唯一买得起的
   **GK=1024 恰好是三个非对称测点里最好的**——P0 与 P2 之间原本的张力消失，选型收敛。

## 5. GPTQ（已完成，结论：无端到端增益，暂缓）

- **实现**（`gptq.py` / `run_gptq.py`）：逐列量化 + 逆 Hessian 误差补偿；group params 走
  `quant.group_params`（**bf16 scale，与 kernel 规则一致**）；支持非对称；顺序标定（每层用已量化的
  前序层产出的激活），q/k/v 共享 Hessian、gate/up 共享；逐层落盘可断点续跑。
- **自检 T7 通过**：合成数据 +3.22 dB；**H=I 时逐位退化为 RTN**（实现正确性的锚点）；
  非对称再叠 +1.29 dB。
- **真实标定集上的增益：+1.3 ~ +6.1 dB**（层输出 SNR，相对 RTN）。
  ⚠️ 冒烟时 4 窗口看到 +10~20 dB 是**假的**——8192 token 对 `down_proj`（K=27648）是秩亏
  Hessian 的样本内过拟合，**必须用 ≥128 窗口**。
- **状态**（2026-08-11 晚）：标定完成——`gptq_gk1024_asym/` 64/64 层（13216 s）、`gptq_gk1024_sym/` 64/64 层
  （12513 s）。PPL 评测（dequant，完整 split，两卡并行各 ~9 min）结果见下。

### GPTQ 的 PPL 实测（dequant，完整 split，`ppl_gptq_{asym,sym}_dequant.json`）

| 配置 | PPL | Δ vs w8a8 | KL | logit SNR |
|---|---:|---:|---:|---:|
| `w4a8-mg1024-asym`（RTN） | 6.9064 | **+3.52%** | 0.180 | 9.89 |
| `w4a8-mg1024-asym-gptq` | 7.0036 | **+4.98%** | 0.164 | 10.31 |
| `w4a8-mg1024`（RTN 对称） | 7.2780 | +9.09% | 0.242 | 8.59 |
| `w4a8-mg1024-sym-gptq` | 7.2010 | +7.94% | 0.203 | 9.32 |

**结论：GPTQ 没有带来端到端收益，非对称档反而更差（+4.98% vs +3.52%）。** 测试集上
asym-GPTQ 的逐层 SNR 相对 RTN 几乎零增益（均值 +0.03 dB，而标定集声称 +1.3~6.1 dB）——
典型的样本外退化；KL/logit SNR 变好但 PPL 变差，与 D2（GK=512 反常）同构。
**用户决定：GPTQ 暂缓，不跑任务评测**，P2 主线维持 RTN 非对称。
（顺带修了 `run_ppl.py` 一个 bug：`--gptq` 是 `action="append"` 但 `default=""`，首次使用必崩，改 `default=None`。）

## 6. 任务评测（GLUE 已完成，GSM8K / 中文待跑）

- `resident.py`：accelerate `device_map` 跨卡常驻（62 GB 需 ≥2 卡），权重就地换量化-反量化版本，
  激活用 forward-pre-hook——**模块图不变**，`generate()` / KV cache / lm_eval 的 HF wrapper 原样可用。
- `run_tasks.py`：接 lm_eval（0.4.12），GLUE / GSM8K / C-Eval / CMMLU 同一路径。
- **计划顺序**：GLUE（loglikelihood，便宜）✅ → GSM8K（生成，对推理模型最有信息量）→ 一项中文。
- ⚠️ **GSM8K 必须开 batch**：实测未批处理只有 4.2 token/s（解码是访存瓶颈），200 题 × 1024 token
  要 13 小时；`--batch-size 8~16` 应接近线性提速。`kernel` 模式 M=1 是 19× bf16，绝不能用它做生成评测。
- **门禁**：用任务分的退化幅度**反过来定验收门槛**（用户原话）——GLUE 数据已到手，见 §6.1 与 §9。

### 6.1 GLUE 实测（2026-08-12，0-shot loglikelihood，limit 500/任务，dequant 模式，2 卡/配置）

`results/tasks_glue_{bf16,w8a8,asym}.json`（bf16 1621 s / w8a8 3934 s / asym 4106 s）：

| 任务（主指标） | bf16 | w8a8 | asym | asym−w8a8 |
|---|---:|---:|---:|---:|
| cola (mcc) | 0.3564 | 0.3501 | 0.2328 | **−0.117** |
| sst2 (acc) | 0.9080 | 0.9140 | 0.9000 | −0.014 |
| mrpc (acc) | 0.7843 | 0.7794 | 0.7819 | +0.003 |
| qqp (acc) | 0.8300 | 0.8380 | 0.7080 | **−0.130** |
| mnli (acc) | 0.5940 | 0.6180 | 0.5700 | −0.048 |
| qnli (acc) | 0.5440 | 0.5220 | 0.5160 | −0.006 |
| rte (acc) | 0.7762 | 0.7617 | 0.7401 | −0.022 |
| wnli (acc) | 0.8592 | 0.8592 | 0.8310 | −0.028 |
| **平均（8 任务主指标）** | | | | **−0.045（−4.5 个百分点）** |

要点：

1. **W8A8 基本无损**（vs bf16 平均 −0.1 个百分点，多个任务还更高），量化链路本身可靠。
2. **W4A8-asym 平均 −4.5 个百分点，分布不均**：6/8 任务只掉 0.3%~5%，但 **cola（mcc −11.7 点）与
   qqp（acc −13.0 点）明显崩盘**——这两个是对 logit 分布形状最敏感的判别任务，与 PPL 阶段
   logit SNR 比 w8a8 低 4 dB 的现象一致。
3. **任务评测比 PPL 严格**：PPL 门禁 +3.52% 看着安全，GLUE 平均却 −4.5 点（路线图原定任务门槛
   绝对分 ≤1 个百分点，**未过**）——这正是"若指标冲突以任务评测为准"要防的情况。
   500 例/任务是 caveat（cola 的 mcc 方差大），但 qqp 500 例的 −13 点不太像采样噪声。

## 7. 决策与坑

| # | 决策/坑 | 要点 |
|---|---|---|
| D1 | **PPL 门槛 2% → 5%**（用户决定） | +5% 对应 BF16 PPL 5.0→5.25，W4A8 可接受的工程标准；路线图要求"开跑前写死"，这是一次明示的、有理由的变更。任务评测仍是最终判据。 |
| D2 | **GK=512 非对称非单调是真实的** | 排除实现问题：量化器逐层单调（asym128 19.4–19.8 > asym512 17.7–18.3 > asym1024 16.9–17.7 dB）；runner 确定性（mg1024-asym 两次运行 PPL 完全相同 6.9092）；dequant 模式独立复现（+6.010%）。机理未解释（SNR 只测幅度不测方向）。 |
| D3 | **GPTQ Cholesky 必然落 CPU** | `torch.linalg.cholesky` 在 torch_npu 无 kernel，静默 fallback；`H.double()` 在 NPU 上静默降回 fp32，必须**先 `.cpu()` 再转 fp64**，否则 `--fp64-hessian` 不生效。 |
| D4 | **标定集用 C4，不用 wikitext-2 train** | 同域标定+评测会美化结果；C4 是 GPTQ/AWQ 论文惯例。 |
| D5 | **不引入外部 GPTQ 实现** | `desc_act` 必须关（kernel 每 GK 列冲刷 L0C，要求 K 分组连续切片）；scale 必须 bf16（标准工具存 fp16 会对不上）。自研 ~215 s/层，全模型 ~3.8 h，与 GPU 同量级。 |
| D6 | **生成评测吞吐** | 未批处理 4.2 token/s，必须 batch 8~16。 |
| D7 | **GPTQ 无端到端增益（已决定暂缓）** | 标定集 +1.3~6.1 dB，测试集 asym 几乎零增益（+0.03 dB）；PPL 从 +3.52% 恶化到 +4.98%。128 窗口对 32B 模型可能仍不足，或误差方向与网络交互不利（与 D2 同类）。用户决定暂缓，不跑 GPTQ 任务评测。 |
| — | lm_eval 安装坑 | 先 `pip install --upgrade setuptools wheel absl-py nltk six`，再 `--no-build-isolation rouge-score`，最后 `pip install lm_eval`（0.4.12）。 |

## 8. 产物与复现

| 路径 | 内容 |
|---|---|
| `fakequant_lab/results/ppl_gk1024_full.json` | 五组 + 非对称，kernel 模式，完整 split |
| `fakequant_lab/results/ppl_gk1024_full_dequant.json` | 同上，dequant 模式（模式对拍） |
| `fakequant_lab/results/gksweep_full.json` | GK ∈ {128,512,1024} × 对称/非对称，完整 split |
| `fakequant_lab/results/ppl_gk512_recheck.json` | GK=512 反常的独立复现（dequant） |
| `fakequant_lab/gptq_gk1024_{sym,asym}/` | GPTQ 标定权重（已完成 64/64 层；结论见 §5，暂缓） |
| `fakequant_lab/results/ppl_gptq_{asym,sym}_dequant.json` | GPTQ 的 PPL 实测（dequant，完整 split） |
| `fakequant_lab/results/tasks_glue_{bf16,w8a8,asym}.json` | GLUE 任务评测（limit 500/任务，0-shot） |
| `fakequant_lab/results/tier12_gk1024.csv` · `awq_probe_gk1024.csv` | P1 产物（tier 扫描 / AWQ 探针） |

复现命令（环境固定 `export ASCEND_RT_VISIBLE_DEVICES=1`，device 0 已挂死）：

```bash
cd vllm
# PPL 门禁（完整 split，~50 min/模式）
python -u -m fakequant_lab.run_ppl --windows 0 --out fakequant_lab/results/ppl_gk1024_full.json
# GPTQ 标定（断点续跑）
export ASCEND_RT_VISIBLE_DEVICES=4
python -u -m fakequant_lab.run_gptq --gk 1024 --asym --calib c4 --nsamples 128 \
    --out fakequant_lab/gptq_gk1024_asym
```

---

**状态行（2026-08-12）**：PPL 门禁 **已过**（`w4a8-mg1024-asym` RTN +3.52%，≤5%）；GPTQ 已评测并
**暂缓**（无端到端增益）；**GLUE 已出**（asym vs w8a8 平均 −4.5 个百分点，未达路线图原定 ≤1 点门槛）。
下一步：**定任务验收门槛（§9）→ GSM8K（batch）→ 一项中文**。

---

## 9. P2 验收标准（现状与待定）

**原定标准**（`MID_GROUP_ROADMAP.md` §6，开跑前写死）：

| 指标 | 原阈值 | 实测 | 状态 |
|---|---|---|---|
| WikiText-2 PPL（相对 W8A8） | ≤ 2%（**用户放宽为 ≤ 5%**，D1） | asym **+3.52%** | ✅ 过 |
| SNR Tier1/2/3 逐层 | ≥ 32 / 35 / 35 dB | P1 已产出 tier12 曲线；P2 未作为门禁用 | — |
| 任务评测（≥2 项，相对 W8A8 绝对分） | ≤ 1 个百分点 | GLUE 平均 **−4.5 点**（cola −11.7 / qqp −13.0） | ❌ 未过（按原标准） |

> 路线图 §6 写明"若三项指标冲突，以任务评测为准"；用户原话是**用任务分的退化幅度反过来定验收
> 门槛**。GLUE 数据已到手，最终门槛待用户拍板，确定后写进本文件与检查点并冻结。

**待拍板的选项**：① 以 GLUE 8 任务平均绝对分差 ≤5 个百分点为门槛（当前 −4.5 点，勉强过，需确认
cola/qqp 崩盘可接受）；② 以生成类任务（GSM8K）为主定门槛，loglikelihood 任务仅供参考；③ 平均 +
最差任务双约束；④ 门槛定不下来/过不了 → W4A8 需要更强的算法（AWQ / 更细 GK / GPTQ 再调），或宣告方案不成立。
