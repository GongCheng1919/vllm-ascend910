# MID-GROUP W4A8 路线图

**独立于 `ROADMAP.md`**（那份是 QwQ-32B vLLM 服务化基线的路线图，两者只在 P4 交汇）。

创建日期: 2026-08-10 · 目标模型: **QwQ-32B**（本地已有 `models/QwQ-32B`，BF16 62G）
硬件: Ascend 910B4 × 8 · CANN 8.5.0 · torch_npu 2.7.1/2.8.0

---

## 0. 一句话目标

把 **mid-group W4A8**（权重 int4、激活 int8，**两侧都按 K 方向大分组量化**）做成一条
从算法到 kernel 到 vLLM 的完整可用路径，并证明它在 QwQ-32B 上**精度接近 W8A8、
decode 吞吐显著优于 W8A8**。

### Definition of Done

1. 有一条 **SNR / PPL 关于 group size 的曲线**，据此选定唯一的 `GK`（并冻结）。
2. 该 `GK` 下 fake-quant 的精度**相对 W8A8 的退化在事先写定的阈值内**。
3. AscendC kernel 在该 `GK` 下正确（CPU 参考逐 shape 比对 + race 压测），
   且 decode 档（M ≤ 16）延迟**优于同结构 W8A8 kernel**。
4. 端到端接入 vLLM，对 `ROADMAP.md` Phase 3 冻结的基线给出可比数字。

**门禁在 P2。** P2 不过就不做 P3——kernel 开发太重，不能押在未验证的算法上。

> **2026-08-13 修订（用户决定）：P2 与 P3 改为并行，P2 暂缓。**
> 原判断的前提是"P3 的 kernel 开发很重"。P0 的探针（D2）为了让代价数字可信，
> 已经把完整的 mid-group 流水写完了，**P3 的边际成本远低于本文写作时的估计**，
> 这个前提不再成立。理由与回退路径见 `MGCKPT/P2_ACCURACY.md` D8 与
> `MGCKPT/P3_KERNEL.md` D2。
>
> **但 P2 的结论没有改变**：它是**被挂起的 go/no-go**，不是通过。PPL 过了（+3.52%），
> 而 §6 自己写的「以任务评测为准」这一条上，GLUE 是 **−4.5 个百分点**。
> P3 是 at-risk 施工，回退目标是 **W8A8-mg（−0.10%）**。

---

## 1. 方案定义

沿 K 方向把长度 `K` 切成 `K/GK` 个组，**A 和 W 用同一组边界**：

```
y[m,n] = Σ_g  as[m,g] · ws[n,g] · ( Σ_{k∈g} A_q[m,k] · W_q[n,k] )

A_q: [M,K] int8,  as: [M, K/GK] bf16     (per-token × per-group)
W_q: [N,K] int4,  ws: [N, K/GK] bf16     (per-channel × per-group)
```

W 和 A 用同一种量化方案（而不是"A per-token + W mid-group"混搭）是刻意的：

- 两侧在**同一个边界**做 rescale，kernel 里只有一套冲刷逻辑；
- 可以直接复用 `nitro` / `mid_group_gemm_fwd_lab` 里已经验证过的 mid-group INT8
  量化与 GEMM 结构，迁移量最小；
- 混搭方案在精度上没有已知优势，工程上却多一套代码路径。

### 与 W4A8 MSD 的关系

`int4_cube_lab` 已验证：910B4 的 cube 没有 s8×s4 混合位宽指令，int8 激活只能通过
MSD 拆分进 s4 通路（`REPORT_W4A8.md` §1）。**拆分沿"数值位"、分组沿 K，两者正交**，
所以 mid-group 不需要重新设计拆分：

```
A_q = 16·h + l + 8         h = A_q>>4, l = (A_q&15)^8, 均在 [-8,7]，对全 int8 精确

Σ_{k∈g} A_q·W_q = 16·S_hi[g] + S_lo[g] + 8·KS[n,g]
                                          └─ KS[n,g] = Σ_{k∈g} W_q[n,k]
```

唯一的变化：**`w_ksum` 从 `[N]` 变成 `[N, K/GK]`**，仍然是权重侧一次性预处理。

**MSD 拆分对精度零影响**（int32 内精确累加），所以 P1 的 fake quant **不需要建模
拆分**，按 int8 激活建模即可。

---

## 2. 既有资产盘点（P0 之前先确认，避免重复造）

| 资产 | 位置 | 用途 |
|---|---|---|
| mid-group INT8 SNR 三层门禁扫描 | `NPU-OP/snr_midgroup_sweep.py` | P1/P2 直接扩展，已有 Tier1/2/3 与"取最大通过档"策略 |
| mid-group INT8 GEMM (AscendC) | `NPU-OP/mid_group_gemm_fwd_lab/`, `mid_group_gemm_lab/` | P3 的分组冲刷结构参考 |
| mid-group INT8 Triton 实现 | `NPU-OP/triton_midgroup_int8.py` | 交叉验证参考实现 |
| **W4A8 MSD kernel（per-channel）** | `vllm/int4_cube_lab/kernels/perchannel_w4a8_gemm.inc` | **P3 的直接基础**，只需改累加粒度 |
| W8A8 / W4A4 对照 kernel | `vllm/int4_cube_lab/kernels/perchannel_lowbit_gemm.inc` | P0/P3 的受控基线 |
| QwQ-32B BF16 / W8A8 / W4A8-Random | `vllm/models/` | P1 校准与 P4 端到端 |
| vLLM 服务化基线（TP4/PP4 矩阵） | `vllm/results/`, `vllm/reports/` | P4 的对照点，已冻结 |

> P0 的第一件事是**读这些**，特别是 `mid_group_gemm_fwd_lab` 的分组冲刷是怎么写的，
> 以及 `snr_midgroup_sweep.py` 的三层判据能否原样搬到 W4。本路线图写作时尚未通读。

---

## 3. `GK` 候选集（由硬件与模型维度共同确定）

**候选集不是自由参数**——它同时受 kernel 流水边界和 QwQ-32B 的 K 整除性约束。

QwQ-32B 的 K 只有两个取值：`5120`（q/k/v/o/gate/up 的 K）与 `27648`（down 的 K）。

| GK | 整除 5120 | 整除 27648 | 对应 kernel 边界 | 结论 |
|---:|:--:|:--:|---|---|
| 256 | ✓ (×20) | ✓ (×108) | 需要 `INNER_K=256`（即 TILE_M=128 的几何）| 备选，流水改动较大 |
| 512 | ✓ (×10) | ✓ (×54) | **1 个 L0 slice**（小 M 档的自然边界）| 备选 |
| 1024 | ✓ (×5) | ✓ (×27) | **2 个 L0 slice** | **首选（最大）** |
| 2048 | ✗ (×2.5) | ✗ (×13.5) | 1 个 L1 group | **排除** |

所以 **P1/P2 只扫 {256, 512, 1024}**，且按"精度达标取最大"的既有策略，1024 是期望落点。

> **P0 实测已进一步收窄（2026-08-10，见 `MGCKPT/P0_COST.md` §2.5）：**
> mid-group 在 decode 档是 **AIV 每组固定指令发射瓶颈**（不是带宽、不是握手）。
> 相对 per-channel W4A8 的代价：**GK=1024 约 1.27–1.54×**（优化 scale 前导后可到
> ~1.12–1.2×），**GK=512 约 1.7–2.8×**（优化到底仍 ~2×），**GK=256 约 2.7–5.1×**。
> 因此：**GK=1024 是唯一现实选项；GK=512 需要 P2 证明精度收益足够大才值得；
> GK=256 实质出局。** P2 的曲线请优先把 1024 这一档做扎实。

> 若 P2 发现 1024 精度不够而 512 够，kernel 代价按 P0 的表评估是否仍划算；
> 若连 256 都不够，则 mid-group W4A8 方案不成立，回退到 per-channel W4A8
> （已实现、已验证）或 W8A8。

> **P1 实测已推翻「缩小 GK 能救精度」这个前提（2026-08-11，见 `MGCKPT/P1_HARNESS.md`）：**
> 完整 WikiText-2 test split 上，RTN 对称 `W4A8-mg1024` 相对 W8A8 是 **+9.03%**；
> GK 缩到 kernel 根本买不起的 128 也只赚回约 1/3 的缺口。归因：**mid-group 对激活
> 值钱（down_proj +8.5 dB），对 int4 权重几乎无效**（per-channel 13.8 dB →
> GK=128 只到 18.3 dB），W4A8 的 GEMM 噪声被权重侧钉死。
>
> **但方案没有死**：加上**非对称权重（每组零点）**后是 **+3.56%**，距门槛只差
> 1.56 个百分点，而 kernel 只多一个每组的激活行和（远便宜于缩小 GK）。因此
> **§6 的阶梯从第 2 级（GPTQ）起步，且非对称应设为默认起点；GK 细化和 AWQ 都不是杠杆**
> （AWQ 式逐通道缩放实测只值 0.3–0.5 dB，因为 QwQ-32B 的 int4 权重难度不是通道
> 结构化的——而这恰恰是 GPTQ 的适用条件）。
>
> 两条影响 §6 方法学的：
> 1. **SNR 三层判据在现成 QwQ-32B 上作废**——连 `w8a8` 自己都过不了 32/35 dB
>    （T2 最差 9.40 dB），可它 PPL 无损。门禁改用 PPL / 任务评测、以 W8A8 为参照系；
>    Tier1/2 保留为定位问题层的诊断工具。
> 2. **门禁数字必须在完整 test split 上取**：8 窗口的子集**系统性高估退化近一倍**
>    （+18.67% vs +9.03%），因为开头那段文本明显更容易。这是偏差不是方差。
>
> 另：§3 末尾写的回退目标 `W4A8-pc` 实测 **+33.85%**，不是可用的回退点；真正的回退是
> **W8A8-mg**（**−0.10%**，比 per-token W8A8 还略好）。

---

## 4. P0：可行域与代价（kernel 侧先出成本列）

**目的：在算法侧动手之前，先把"每个 GK 值多少钱"量出来，作为 P2 选型的输入。**

mid-group 相对现在的 per-channel W4A8，唯一的结构变化是 **L0C 不再跨整个 K 累加**，
而是每 `GK` 冲刷一次：

| 变化项 | per-channel（现状） | mid-group GK |
|---|---|---|
| Fixpipe 次数 / 输出 tile | 1 | `K/GK` |
| CrossCore 握手对 / 输出 tile | 1 | `K/GK` |
| workspace 往返字节 / 输出 tile | `2·TILE_M·TILE_N·4` | `(K/GK)×` 同上 |
| AIV 每 tile 的 pass 数 | 1 | `K/GK`（累加到 UB 的 fp32 累加器）|

workspace 往返大概率**命中 L2**（每 tile 每组只有 16–128 KB，写完立刻读，L2 有 168 MiB），
所以主要代价预计不是 HBM 带宽，而是**握手次数与 AIV pass 数**。但这是推测，P0 要实测。

### 工作

1. 读既有 mid-group INT8 kernel 的分组冲刷实现，确认可复用的结构。
2. 在 `int4_cube_lab` 里加一个**只改流水、不改数学**的探针 build：
   把 K 循环按 `GK` 切开，每组做一次 Fixpipe + AIV pass，**scale 全部取 1**，
   结果与现有 per-channel kernel 逐位一致即为正确。
3. 在 QwQ-32B 的真实形状上扫 `GK ∈ {256, 512, 1024, ∞}` × `M ∈ {1,4,16,64,512,1024}`，
   cold 测量。

QwQ-32B 形状（TP=1）：`gate_up N=2×27648=55296, K=5120`；`down N=5120, K=27648`；
`qkv N=5120+1024+1024=7168, K=5120`；`o N=5120, K=5120`。

### 交付

`int4_cube_lab/results/midgroup_cost.csv` + 一张表：**每个 GK 相对 `GK=∞` 的延迟增幅**，
decode 档和 prefill 档分列。这张表交给 P2 做取舍。

### Exit Criteria

- 探针 build 与 per-channel kernel 结果逐位一致（证明只改了流水）。
- 三个 GK 各有一组 cold 数字，且能回答："若 GK 从 1024 降到 512，多付多少百分比"。

---

## 5. P1：fake-quant harness

**目的：一套能在 QwQ-32B 上跑 BF16 / W8A8 / mid-group W4A8×3 档的伪量化插桩。**

### 量化定义（与 §1 严格一致）

- **W**：int4 对称，per-(output channel, K-group)，scale = `max|w|/7`（bf16 存储）。
- **A**：int8 对称，per-(token, K-group)，**动态**（运行时按当前 token 的组内 max 计算），
  scale = `max|a|/127`（bf16 存储）。
- 不建模 MSD 拆分（§1 已论证其精确）。
- scale 用 **bf16** 存储并在计算前 round 到 bf16——kernel 里就是 bf16，
  fake quant 不得用 fp32 scale，否则会系统性高估精度。

### 对照组（必须同框，同一 harness、同一评测集）

| 组 | W | A |
|---|---|---|
| BF16 | — | — |
| W8A8 | int8 per-channel | int8 per-token |
| W8A8-mg | int8 mid-group GK | int8 mid-group GK |
| **W4A8-mg** | **int4 mid-group GK** | **int8 mid-group GK** |
| W4A8-pc（对照） | int4 per-channel | int8 per-token |

> `W8A8-mg` 这一组用来把"mid-group 本身的影响"和"W 降到 4bit 的影响"分开，
> 否则曲线上两个变量纠缠在一起。`W4A8-pc` 是已经实现的 kernel 对应的算法配置，
> 作为下界参照。

### 覆盖范围

先只量化 **Linear 层的 W 和 A**（qkv / o / gate_up / down），
**不动** embedding、lm_head、norm、attention 内部的 QK/PV matmul。这与 kernel 的
作用域一致；扩大范围放到 P2 之后再议。

### Exit Criteria

- harness 能对同一份权重产出 5 组配置的输出，且 BF16 组与原模型逐位一致（插桩无副作用）。
- 单层 GEMM 级的 SNR 可导出（复用 `snr_midgroup_sweep.py` 的 Tier 1/2）。

---

## 6. P2：算法选型与精度门禁 ← **决策点**

### 曲线（本阶段的核心交付）

对 `GK ∈ {256, 512, 1024}` × 上述 5 组配置，产出两条曲线：

1. **SNR vs GK**：沿用 `snr_midgroup_sweep.py` 的三层判据
   （Tier1 单张量 ≥ 32 dB / Tier2 单 GEMM ≥ 35 dB / Tier3 单层 ≥ 35 dB），
   逐层给出，重点看最差层。便宜、快，用来先筛掉明显不行的档。
2. **PPL vs GK**：WikiText-2，QwQ-32B 全模型。慢，只对通过 SNR 筛选的档跑。

### 量化算法阶梯

按成本从低到高，**能过就停**：

1. **RTN**（round-to-nearest）——先跑，作为下界。若 RTN 在 GK=1024 就够，后面全省。
2. **GPTQ**——若 RTN 不够。注意 GPTQ 的 Hessian 是按组更新的，`group_size` 直接设成 `GK`。
3. **AWQ**——若 GPTQ 仍不够。
4. 都不够 → 见 §3 的回退分支。

### 通过标准（**开跑前写死，不得事后调整**）

| 指标 | 阈值 |
|---|---|
| WikiText-2 PPL | `W4A8-mg` 相对 **W8A8** 的退化 ≤ **2%**（相对 BF16 ≤ 4%）|
| SNR Tier1/2/3 | 分别 ≥ 32 / 35 / 35 dB，**逐层**，最差层也要过 |
| 任务评测 | 至少 2 项（建议 GSM8K + 一项中文），相对 W8A8 绝对分下降 ≤ 1 个百分点 |

> QwQ-32B 是推理模型，输出长、对累积误差敏感，任务评测比 PPL 更能反映真实退化。
> 若三项指标冲突（PPL 过、任务不过），以**任务评测**为准。

### 决策

在通过阈值的档里，**结合 P0 的代价表取性价比最高的一档**（默认取最大的通过档），
写入本文件并**冻结**。此后 kernel 只实现这一个 `GK`。

### Exit Criteria

- 两条曲线 + 一张决策表落盘到 `reports/`。
- `GK` 已冻结并记录理由。
- 若无档通过，**明确宣告方案不成立**并记录回退选择，不进入 P3。

---

## 7. P3：kernel 实现

> **2026-08-13 修订：基准冻结为 `W4A8-mg1024-asym`（GK=1024 + 非对称权重）。**
> 起点不再是 `perchannel_w4a8_gemm.inc`，而是 **P0 的探针 `midgroup_w4a8_gemm.inc`**
> ——下面 1./2. 的流水改动它已经做完并压测通过。实际剩余工作、非对称的数值契约
> 与 AIV 增量见 `MGCKPT/P3_KERNEL.md`。
>
> 非对称使 §1 的式子多一个 rank-1 项（A 侧仍是对称 int8）：
>
> ```
> y[m,n] = Σ_g as[g,m]·ws[g,n]·( 16·S_hi + S_lo + 8·w_ksum[g,n] − wz[g,n]·a_ksum[g,m] )
> ```
>
> 即每组多一个**激活行和** `a_ksum[g,m]` 和一个**权重零点** `wz[g,n]`。
> `w_ksum` 仍用原始码值、**不减零点**（它是 MSD 的 `+8` 偏置，与非对称无关）。
> int4 码值仍是有符号 nibble `[-8,7]`，**存储格式与 cube 通路不变**。

基础是 `int4_cube_lab/kernels/perchannel_w4a8_gemm.inc`，**stacked-A 布局不动**
（hi/lo 沿 M 叠成 `2·TILE_M` 的 cube tile，一条 Mmad、一次 Fixpipe）。改动集中在三处：

1. **L0C 冲刷粒度**：从"整个 K 一次"改成"每 `GK` 一次"（P0 探针里已经验证过流水）。
2. **AIV 累加**：每组做 `(16·C_hi + C_lo + 8·KS[n,g]) · ws[n,g] · as[m,g]`，
   累加进 UB 的 fp32 累加器；最后一组之后才 Cast 成 bf16 写出。
   UB 里的 fp32 累加器已经存在（`ubAcc`），不新增预算。
3. **scale / ksum 的加载**：`ws` 从 `[N]` 变 `[N, K/GK]`，`as` 从 `[M]` 变 `[M, K/GK]`，
   `w_ksum` 从 `[N]` 变 `[N, K/GK]`。每组一次小 DataCopy。
   （P0 实现为 **group-major** `[K/GK, N]` / `[K/GK, M]` 并批量 staging `BATCH_G` 组，
   见 `MGCKPT/P0_COST.md` D4；非对称再加 `wz[G,N]` 与 `a_ksum[G,M]` 两个张量。）

配套还需要一个 **per-token mid-group 量化 kernel**：吃 bf16 激活，吐 `a_hi` / `a_lo`
两个 int4 平面 + `as[M, K/GK]`（非对称基准下再加 `a_ksum[M, K/GK]`，逐组 max 的 reduce
顺带求和，几乎免费）。这是 elementwise，字节总量与吐一份 int8 相同，
但**必须实现**，否则 §4 的性能数字不含这一步（`REPORT_W4A8.md` §7 已记此缺口）。

### 交付与 Exit Criteria

- `GK` 为编译期常量，**只出必要的 TILE_M 档**（预期 16 / 64 / 128 三档，视 P2 结论裁剪）。
- CPU 参考逐 shape 比对（A 按全 int8 范围、W 按 [-8,7] 生成），SNR ≥ 40 dB。
- race 压测：tile 数 > blockDim 的形状连跑 30 次全过。
- QwQ-32B 全部四种 Linear 形状 × `M ∈ {1,4,16,64,128,512,1024}` cold 扫描，
  **decode 档延迟低于同结构 W8A8 kernel**。
- 量化 kernel 的开销单独测量并计入端到端账。

---

## 8. P4：端到端

1. 用 P2 选定的算法与 `GK` 产出**真实校准的** QwQ-32B W4A8 checkpoint
   （替换 `models/QwQ-32B-W4A8-Random` 那份随机权重）。
2. AscendC kernel 经 `op-plugin` / `TORCH_LIBRARY_FRAGMENT` 包成 `torch.ops.npu.*`。
3. 接入 vLLM 的 Linear 层，替换掉现在的 `npu_weight_quant_batchmatmul` 路径
   （该路径实测是 weight-only bf16 反量化，激活的 int8 从未进入 GEMM，
   见本轮分析与 `reports/W4A8_PERGROUP_ANALYSIS.md`）。
4. 对 `ROADMAP.md` Phase 3 冻结的 TP4 矩阵重跑，报 BF16 / W8A8 / W4A8-mg 三方对比。

### Exit Criteria

- 功能：与 `ROADMAP.md` Phase 2 相同的 smoke test 全过。
- 性能：decode 密集档（1024 in / 256 out，并发 1–64）相对 W8A8 有可测提升。
- 质量：P2 的任务评测在端到端上复现（fake quant 与真实 kernel 结论一致）。

---

## 9. 风险与未决问题

| 风险 | 影响 | 缓解 |
|---|---|---|
| GK=1024 精度不够，只有 256 能过 | P0 的代价表决定是否还划算；可能整个方案不成立 | P0 先把代价量出来，P2 决策时手里有数 |
| workspace 往返未命中 L2，成为 HBM 瓶颈 | mid-group 的优势被吃掉 | P0 探针实测，不靠推测 |
| 每组一次 CrossCore 握手，decode 档延迟被同步支配 | 小 M 收益消失 | P0 实测；必要时探索一次握手覆盖多组 |
| A 的动态 mid-group 量化 kernel 开销未知 | 端到端收益打折 | P3 单独测量，计入账 |
| QwQ-32B 激活存在离群点，int8 per-group 不稳 | 精度门禁不过 | P2 先看逐层 SNR 的最差层定位问题层 |
| `snr_midgroup_sweep.py` 的 INT8 判据未必适用于 W4 | 门禁失准 | P1 先用 W8A8-mg 组复现已知结论，验证判据可迁移 |

---

## 10. 产物契约

```text
reports/midgroup/
|-- P0_cost.md              GK 代价表（kernel 侧）
|-- P1_harness.md           fake-quant 定义与自检
|-- P2_accuracy.md          SNR/PPL 曲线 + 决策表 + 冻结的 GK
|-- P3_kernel.md            正确性、压测、性能扫描
`-- P4_e2e.md               端到端对照

int4_cube_lab/results/
|-- midgroup_cost.csv
`-- midgroup_w4a8_qwen.csv
```

每份报告都要分开陈述**执行成功**与**性能/精度结论**——kernel 跑通不等于方案成立。

---

## 附：阶段依赖

```
P0 代价表 ──┐
            ├──> P2 决策（门禁）──> P3 kernel ──> P4 端到端
P1 harness ─┘
```

P0 与 P1 可并行（一个在 kernel 侧、一个在算法侧，互不阻塞）。
**P2 是唯一的 go/no-go 点。**
