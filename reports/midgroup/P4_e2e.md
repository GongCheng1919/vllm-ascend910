# P4 — mid-group W4A8 端到端接入 vLLM

**日期** 2026-08-14 · **硬件** Ascend 910B4 · CANN 8.5.0 · `ASCEND_RT_VISIBLE_DEVICES=1`
**软件** vLLM v0.13.0 + vllm-ascend · torch 2.8.0 + torch_npu 2.8.0.post2（`.venv`）
**代码** `vllm/npu_ops/` · **执行记录** `MGCKPT/P4_E2E.md`（CKPT-A ~ CKPT-E）
对应 `MID_GROUP_ROADMAP.md` §8

> 本报告按产物契约分开陈述**执行成功**与**性能结论**。
>
> **前置声明一：P4 是 at-risk 施工。** P2 精度门禁**没有通过**——PPL 过了（+3.52%），
> 但唯一测过的任务指标 GLUE 是 **asym vs W8A8 平均 −4.5 个百分点**
> （cola −11.7 / qqp −13.0）。P2 / P3 / P4 并行是 2026-08-13 的明示决策
> （`MGCKPT/P2_ACCURACY.md` D8）。**本报告的性能数字不构成方案可用的证明。**
>
> **前置声明二：全部为随机权重（`load_format=dummy`）。** 生成的文本按构造就是
> 乱码，本报告**只测延迟，不测质量**。真实 checkpoint 的导出器（路线图第 1 步）
> 尚未产出。
>
> **前置声明三：本报告的每一个比值都带 batch 档。** 本期最大的教训是
> weight-only / mid-group / per-channel 三条路的**排序会随 M 换位**——
> 不带 M 的「X 比 Y 快」在这个课题里是错误陈述（§7.2）。

---

## 1. 交付了什么

| 项 | 状态 |
|---|---|
| P3 的两个 kernel 包成 `torch.ops.npu.midgroup_{w4a8_gemm,quant_a}` | ✅ |
| 在**真 vLLM 引擎**里被 ACL 图捕获并 replay，四种投影全部接管，端到端能生成 | ✅ |
| 加载期转换器（BF16 → kernel 布局），64 个 linear 全部转换，0 跳过 | ✅ |
| 五路对照基准（原生 BF16 / 原生 W8A8 / 原生 W4A8 / 我们 W8A8 / 我们 W4A8） | ✅ |
| batch 1→256 八档扫描 | ✅ |
| 逐位回归（对改动前的 `.so`，含 ragged M） | ✅ |
| 真实校准 checkpoint（路线图第 1 步） | ❌ 未做 |
| TP4 矩阵（路线图第 4 步） | ❌ 未做 |

---

## 2. 口径——**先读这一节，否则数字会读错**

本期三分之二的工作量花在发现「上一版口径是错的」。有三个口径陷阱，
每一个都足以让结论反号：

| 陷阱 | 错在哪 | 正确做法 |
|---|---|---|
| **eager 计时** | batch ≤16 量的是 python 下发，不是设备；量化路径每投影多 3–4 次下发，**必然显得更慢** | 图捕获后 replay |
| **手搭图 ≠ 引擎图** | `torch.npu.NPUGraph` 不经过 dynamo，量不到 python 分支被折掉的问题 | 在真引擎里量 |
| **默认图模式** | vllm-ascend 默认 `PIECEWISE`，每层被 attention 切开、段间跑 python，**白亏 2 倍** | 显式传 `FULL_DECODE_ONLY` |

**本报告的口径**：QwQ-32B 几何、TP=1、`FULL_DECODE_ONLY`、
`prompt_len=128 / out_len=64`、随机权重。

**per-layer 斜率法**：跑 `--layers 4` 和 `--layers 16`，差分除以 12。
这样扣掉每步 ~4.1 ms 的固定开销——其中大头是 **lm_head（152064×5120 bf16
= 1.56 GB / 解码步）**，它不被任何 arm 接管，留在分母里会把所有比值往 1 拉。
**因此本报告的比值是每层比值，不是端到端比值**：16 层实测端到端是 1.19×，
外推 64 层约 1.26×，而每层斜率是 1.32×（同一次测量）。

---

## 3. 执行成功（正确性）

### 3.1 逐位回归

补齐从 python 搬进算子后（§4.1），对**改动前的 `.so`** 逐位对拍：

| 检查 | 覆盖 | 结果 |
|---|---|---|
| torch 侧对拍 | 3 形状 × 14 个 M（1/2/3/15/16/17/33/64/65/100/128/129/200/256）| **42/42 逐位相同** |
| lab `--check` | M = 1/3/16/17/64/100/128/1024，K=5120 | 8/8 **byte mismatches = 0** |
| 图捕获整层无回归 | batch=1 | 376.6 µs vs 冻结值 378.4（重复区间 376.7–382.4）|

脚本 `npu_ops/python/test_pad_regress.py`。

### 3.2 期间修掉的一个真 bug

`midgroup_quant_a.inc` 里零填充 `ubX`（V 管道）与装载 `ubX`（MTE2）之间只有
`PipeBarrier<PIPE_V>`——**PIPE_V 只排 V 对 V**，零填充会把刚装载的行冲掉。

**它被一个调用方不变量藏了整整一期**：python 和 lab harness 都先把 M 补成
kBr 的倍数再进 kernel，`rows < kBr` 那条分支从来没执行过。补齐搬进算子、
M=1 的 ragged job 上了解码热路径，它当场现形——**恰好 (M mod 16) 行不对**。
已加 `V_MTE2` flag 并提到无条件执行（同类 hazard 还有第二处：上一个 job 对
`ubX` 的 V 读 vs 本 job 的 MTE2 写）。

---

## 4. 性能结论

### 4.1 两项改动，各自值多少

| 改动 | 每层 µs（batch=1）| 对 BF16 |
|---|--:|--:|
| 接入引擎，原样（CKPT-B）| 882.1 | 1.34× |
| ＋ 补齐搬进 `midgroup_quant_a`（CKPT-C）| 878.5 | 1.41× |
| ＋ 图模式换 `FULL_DECODE_ONLY`（CKPT-D）| **385.8** | **2.82×** |

- **补齐**：引擎 profiler 里 **PadV3 3840 次 / MemSet 3840 次 → 全部归零**，
  投影的每层设备时间 **568 → 321 µs（−44%）**。
  但墙上时间几乎没动——**当时引擎是主机受限的**（设备利用率 36%）。
- **图模式**：`Computing` 几乎不变（462.6 → 458.9 ms），
  **`Free` 掉了 481 ms**（807.1 → 326.0）。空隙就是 PIECEWISE 的分段边界。

**两件事缺一不可**：只改补齐是 1.41×，只换图模式（补齐仍回退）约 1.78×。

### 4.2 交叉验证：引擎口径与手搭口径终于一致

| | 手搭图（`bench_decode_graph.py`）| 引擎（`bench_engine_decode.py`）|
|---|--:|--:|
| 我们 W4A8，batch=1 | 376.6 µs/层 | 385.8 µs/层 |
| vs 我们 W8A8 | 1.33× | 1.32× |

**两个完全独立的口径给出同一个数。** PIECEWISE 下它们差 2.3 倍、无法互证；
FULL 模式下 CKPT-A 那套手搭数字可以当引擎的预测用了。

### 4.3 五路对照 × 八档 batch（每层斜率 µs）

| arm | 1 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 原生 BF16 | 1072 | 1295 | 1372 | 1584 | 1776 | 1709 | 2163 | 2377 |
| 原生 W8A8 | 792 | 970 | — | 1209 | — | 1223 | — | — |
| 原生 W4A8 | 441 | 922 | 1873 | 1991 | 1967 | 2236 | 2363 | 2679 |
| 我们 W8A8 | 485 | 595 | 687 | 830 | 885 | 933 | 1240 | 1496 |
| **我们 W4A8** | **408** | **520** | **676** | **731** | **1008** | **986** | **1699** | **1816** |

| 我们的 W4A8 vs | 1 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **原生 W4A8** | 1.08× | 1.77× | **2.77×** | 2.72× | 1.95× | 2.27× | 1.39× | 1.48× |
| 我们 W8A8 | 1.19× | 1.15× | 1.02× | 1.14× | **0.88×** | **0.95×** | **0.73×** | **0.82×** |
| 原生 BF16 | 2.63× | 2.49× | 2.03× | 2.17× | 1.76× | 1.73× | 1.27× | 1.31× |
| （原生 W4A8 vs BF16）| 2.43× | 1.40× | **0.73×** | 0.80× | 0.90× | 0.76× | 0.92× | 0.89× |

**两个 arm 的定义差别很重要，不能混读：**

- **原生 W8A8 / 原生 W4A8** 加载厂商 checkpoint、走 vllm-ascend 自己的量化路径
  （`quantization="ascend"`）。它们回答的是**「货架上的产品有多快」**。
- **我们 W8A8 / 我们 W4A8** 从同一份 BF16 权重、在同一个加载钩子上、对同一批
  64 个 linear 转换。这一对回答的是**「隔离掉 checkpoint 和转换路径后，
  GEMM 谁快」**。

**口径限制**：曲线拼自两轮（1/16/64/128 一轮，32/96/192/256 另一轮）。
每轮内部单调，但第二轮系统性偏高（最大 batch 不同 → 捕获的图和 KV 布局不同）。
**同一列内跨 arm 的比值干净；跨列的绝对值和精确交叉点不干净。**

---

## 5. 三条稳的结论

### 5.1 对货架上的 W4A8：**全程赢，1.08×（M=1）到 2.77×（batch=32）**

厂商路径是 **weight-only + group_size=128**：`npu_weight_quant_batchmatmul`
把 int4 反量化成 bf16 再算，激活的 int8 从不进 cube。

- **M=1 它几乎不吃亏**（2.43× BF16）——解码档是**带宽受限**的，
  weight-only 照样只从 HBM 读 int4，反量化在片上。它放弃的是**算力**收益，
  而 M=1 根本不缺算力。
- **batch≥32 它崩掉**（0.73–0.92× BF16，**比 BF16 还慢**）——cube 成为瓶颈后
  它得按 bf16 算力算，还要多付反量化。

g128 不是它的疏忽而是它的约束：每 128 个元素就要冲刷一次 L0C 并施加组 scale，
真 W4A8 的 cube 路径基本走不通。**这也正是我们把 GK 定到 1024 的原因。**

### 5.2 对我们自己的 W8A8：**小 batch 赢 1.15–1.19×，batch≥96 输到 0.73–0.88×**

交叉点在 32 和 96 之间（受口径限制，暂时钉不死）。

**输的是分组冲刷，不是 MSD。** P3 的 kernel 级数据给了干净的对照——
`per-channel int4`（同样做 MSD 拆分，只是没有分组冲刷）：

| M | 1–16 | 64 | 128 | **512** | **1024** |
|---|--:|--:|--:|--:|--:|
| mid-group 相对 per-channel int4 的税 | 1.05–1.12× | 1.10–1.25× | 1.05–1.20× | **1.34–1.42×** | **1.42–1.47×** |

换算过去：M=1024 时 mid-group 对 W8A8 是 0.76×（qkv），
但 **per-channel int4 对 W8A8 是 1.12×，仍然赢**。
**int4 cube 的 2 倍吞吐确实抵掉了 MSD 的两次 MAD——架构是对的**，
输的是每个量化组一次 workspace 往返（G = K/1024 = 5~27 次，per-channel 只有 1 次）。
小 M 时 DMA 受限把它藏住，大 M 时 cube 忙起来就露出来。

**因此：应当修 kernel，而不是做按 M 分派。** 分派需要同时保留两份权重，
与「转换后丢掉 bf16」的内存收益直接冲突；修冲刷则同时改善对 W8A8 和对
原生 W4A8 的比值。

### 5.3 对 BF16：**全程 ≥1.27×**，batch=1 时 2.63×

---

## 6. 已知边界与未验证项

| 项 | 状态 |
|---|---|
| **精度** | **P2 门禁未过（GLUE −4.5）。性能数字不构成方案可用的证明。** |
| 真实 checkpoint | 未产出，全部为随机权重 |
| TP > 1 / PP | **未验证**。源码里没有禁止 FULL + TP>1 的分支，SP 的 capture-size 调整对两种模式一视同仁，但「代码没禁止」≠「验过」——全图意味着集合通信被录进图 |
| 全 64 层 | 未跑。本报告是 4/16 层取斜率 |
| `FULL_DECODE_ONLY` 的生产风险 | vllm-ascend 自己打红色警告：*early experimental… capturing too many batch sizes can lead to **OOM errors or inference hangs*** |
| prefill 回归 | 未量。`FULL_DECODE_ONLY` 的 prefill 走 eager |
| 精确交叉点 | 需在一个进程里跑完所有 batch 档 |
| batch=1 跑间波动 | 同一对 arm 不同轮次给出 1.19× / 1.32×，引用需给区间 |

**关于图模式能否上生产**（源码依据）：

- 上游 vLLM v1 的默认值就是 `FULL_AND_PIECEWISE`，docstring 称其
  "the most performant mode for most models and is the default"；
- vllm-ascend 把它降级成 PIECEWISE，注释是
  `# TODO: Full graph is fully supported later, and the default value will be set to full graph`；
- **坑**：显式传 `FULL_AND_PIECEWISE` 会被**静默降级成 PIECEWISE**，
  NPU 上必须写 `FULL_DECODE_ONLY` 或 `FULL`；
- 硬限制：pooling 模型、encoder-decoder 强制 PIECEWISE，`enforce_eager` 强制 NONE
  ——都与 QwQ-32B 无关。

---

## 7. 本期推翻的判断

### 7.1 「补齐修掉后引擎应到 1.5–1.8×」——错，实际 1.41×

预测基于「设备时间省 249 µs/层」，但**当时引擎是主机受限的**（利用率 36%），
省下的设备时间只是变成了更多空闲。**拿设备时间推墙上时间，中间少了一步
「设备是不是瓶颈」。** 判据：`step_trace_time.csv` 的 `Computing` vs `Free`。

### 7.2 「D3 说厂商 W4A8 比 BF16 还慢，是错的」——我改过头了，D3 在它自己的口径下是对的

我拿 **M=1** 的一个点（2.43× BF16）去推翻 D3 在**大 M** 下的观察。
扫描显示 batch≥32 确实比 BF16 慢（0.73–0.92×）。**D3 没错。**

**同一个错误在本期犯了两次，都是单工作点外推。** 由此立下的规矩：

> **任何「X 比 Y 快/慢」的结论必须带 M 档。**
> 本课题里 weight-only / mid-group / per-channel 三条路的排序**在 M 上会换位**。
> 消融同理：**M=16 的消融不能说明 M=512**。

### 7.3 「自定义 transformer 层是下一步」——被一个参数取代

本来的理由是压主机开销（每层 ~390 µs 空隙）。`FULL_DECODE_ONLY` 一个参数
把它压到 ~19 µs（385.8 wall vs 367 device）。**主机侧没有空间了**，
自定义层要重开必须给出新的、设备侧的理由。算子融合同理（只值 −21 µs/层）。

### 7.4 「P5 只剩 25 µs，收工」——只对解码档成立

P5 的消融**全部在 M=16 做的**（那里 DMA 受限）。大 M 的 mid-group 税是
**40–47%**，量级完全不同，而且是 §5.2 高并发劣势的直接原因。
**P5 应当重开，从 M=512 的消融开始。**

---

## 8. 复现

```bash
# 单 arm（图模式必须显式给，默认 PIECEWISE 白亏 2 倍）
ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_engine_decode.py \
    --arm w4a8 --layers 16 --batch 1 --out-len 64 \
    --cudagraph-mode FULL_DECODE_ONLY --out results/engine.csv
# arm ∈ {bf16, w8a8-native, w4a8-native, w8a8, w4a8}
# 每层斜率 = (step_us@16层 − step_us@4层) / 12

# 逐位回归（需先备份改动前的 .so 作为 golden）
.venv/bin/python npu_ops/python/test_pad_regress.py --mode golden --lib <old.so> --out g.npz
.venv/bin/python npu_ops/python/test_pad_regress.py --mode new --ref g.npz
```

| 数据 | 路径 |
|---|---|
| 五路 × 八档扫描 | `int4_cube_lab/results/engine_batch_sweep_5arm.csv` |
| 五路对照（3 次重复）| `int4_cube_lab/results/engine_decode_5arm.csv` |
| FULL 模式斜率 | `int4_cube_lab/results/engine_decode_full.csv` |
| 图捕获整层无回归 | `int4_cube_lab/results/e2e_decode_graph_ckptc.csv` |
| kernel 级 M 扫描 | `int4_cube_lab/results/midgroup_w4a8_qwen.csv` |

---

## 9. 结论

**执行成功**：mid-group W4A8 kernel 已在真 vLLM 引擎里被 ACL 图捕获、replay、
端到端生成，四种投影全部接管，数值与 P3 逐位一致。

**性能结论（每层斜率，QwQ-32B / TP=1 / FULL_DECODE_ONLY / 随机权重）**：
对 BF16 全程 **≥1.27×**（batch=1 时 2.63×）；对货架上的 W4A8 全程赢，
**1.08×（M=1）到 2.77×（batch=32）**；对一个精简实现的 W8A8，
**小 batch 赢 1.15–1.19×、batch≥96 输到 0.73–0.88×**——原因已定位为分组冲刷，
且已确认**不是架构上限**（同样做 MSD 的 per-channel int4 在 M=1024 仍赢 W8A8 1.12×）。

**但**：P2 精度门禁未过，权重是随机的，TP>1 与全 64 层未验证。
**本报告是工程验证，不是方案可用的结论。**
