# P4 — 端到端接入 vLLM

**状态：本期冻结于 CKPT-E（§0''''），报告已产出，转 `P5_KERNEL_OPT.md`**
**（第 2、3 步完成；第 1、4 步未做）** · 最后更新 2026-08-14
**口径已换代**：eager 基准（§7）作废，图捕获基准（§8）为准 — D7 / D8
对应 `MID_GROUP_ROADMAP.md` §8 · 代码 `vllm/npu_ops/`（新建）· 环境 `ASCEND_RT_VISIBLE_DEVICES=1`

---

## 0. 基线冻结 · CKPT-A（2026-08-13）

**这一节是记号，不要再改数字。** 后续 kernel 优化（`P5_KERNEL_OPT.md`）的一切
「快了多少」都对这里比。复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_decode_graph.py \
    --batches 1 16 64 128 --out int4_cube_lab/results/e2e_decode_graph.csv
```

| batch | BF16 | W8A8 | **w4a8-all4** | vs BF16 | vs W8A8 |
|--:|--:|--:|--:|--:|--:|
| 1 | 953.7 | 492.3 | **378.4** | 2.52× | 1.30× |
| 16 | 1179.3 | 600.1 | **500.6** | 2.36× | 1.20× |
| 64 | 1655.9 | 994.7 | **877.2** | 1.89× | 1.13× |
| 128 | 1970.1 | 1276.6 | 1361.1 | 1.45× | 0.94× |

口径：NPU 图捕获 replay，层内用 `npu_rms_norm` / `npu_fused_infer_attention_score` /
`npu_swiglu`，QwQ-32B TP=1，kv_len=1024，随机权重。**别拿 §7 的 eager 数字比**（D7/D8）。

**冻结时的代码状态**

| 项 | 状态 |
|---|---|
| `npu_ops/kernel/*` | 与 `int4_cube_lab/kernels/*` **逐字节相同**（已核对 3 个文件）|
| `vllm_midgroup_linear.py` | 含 D9 的常驻缓冲补齐；对 `F.pad` 版本逐位相同 |
| `torch.ops.npu.midgroup_*` | 可被 `torch.npu.NPUGraph` 捕获 |
| 精度 | 与 `fakequant_lab` 逐位一致（P3）；**P2 门禁仍未过**（D1）|

**已知的下一步瓶颈**：GEMM 的 GM→L1 权重搬运（**不是**分组冲刷握手——
§8.5 的诊断已被 `P5_KERNEL_OPT.md` §3' 的消融推翻）。冷口径下值 ~25 µs / batch=1。
**转 `P5_KERNEL_OPT.md`。**

---

## 0'. CKPT-B（2026-08-13）· **第 3 步完成：kernel 已在真 vLLM 引擎里跑**

**里程碑**：`torch.ops.npu.midgroup_*` 在 vLLM v1 + vllm-ascend 的
**ACL 图捕获**里正常 replay（日志 `Replaying aclgraph`），四种投影全部接管，
端到端能生成。这是 P4 §2 的第 3 步，此前所有数字都来自手搭的层。

### 0'.1 数据（`results/engine_decode.csv`，QwQ-32B 几何，TP=1，ACL 图开）

用两个层数取斜率，扣掉每步固定开销（调度/采样/embedding 约 3.8–4.1 ms）：

| arm | 4 层 µs/step | 16 层 µs/step | **斜率 = µs/层** | 固定开销 µs |
|---|--:|--:|--:|--:|
| BF16 | 8817.5 | 22985.3 | **1180.7** | 4095 |
| **W4A8-mg** | 7362.1 | 17947.0 | **882.1** | 3834 |

**引擎口径 1.34× / 层**（吞吐 43.5 → 55.7 tok/s @16 层）。

### 0'.2 复现

```bash
ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_engine_decode.py \
    --arm w4a8 --layers 16 --batch 1 --out-len 64 --out int4_cube_lab/results/engine_decode.csv
```

`--layers` 用 `hf_overrides` 砍层数、`load_format=dummy` 不读盘，所以一轮 1–2 分钟。
**per-layer 才是可迁移的量，绝对时间不是**（少了 48 层）。

### 0'.3 为什么是 1.34× 而不是 CKPT-A 的 2.52×——**已定位，不是 kernel 的问题**

profiler（`llm.start_profile()`）打出来，解码期设备时间的构成：

| kernel | 次数 | ms | 占比 |
|---|--:|--:|--:|
| midgroup_w4a8_gemm_m16 | 960 | 69.9 | 38.9% |
| **MemSet** | **3840** | **25.8** | **14.4%** |
| **PadV3** | **3840** | **34.0** | **19.0%** |
| aclnnMatmul（prefill）| 16 | 19.8 | 11.0% |
| midgroup_quant_a | 1024 | 7.3 | 4.1% |

**33% 的解码设备时间花在补齐上**。pad 的形状是 `[1,2560]→[16,2560]`、
`[5,1]→[5,16]`——**M=1 没补齐就进了 GEMM host**，于是 host 对
`a_hi/a_lo/a_scale/a_ksum` 各补一次（每次调用 4 个 pad × 960 = 3840），
正是 **D6 记录的反模式**，而 D6 的修法（在量化之前只补 `x` 一个张量）
写在 python 里，**在图捕获下没有生效**。

**根因（D10）**：`MidGroupW4A8Linear.__call__` 里的 `if mp != m:` 是个
**数据无关但符号化的分支**，dynamo 追踪时把它折成 False。**eager 下 profiler
里一个 PadV3 都没有**（分支真的执行了），图捕获下 3840 个——这也是 CKPT-A
量不到的原因：`bench_decode_graph.py` 用的是 `torch.npu.NPUGraph` 直接捕获，
**不经过 dynamo**，python 分支照常执行。

试过但**无效**的两招：`m = int(x2.shape[0])` 强制特化、`VLLM_DISABLE_COMPILE_CACHE=1`
排除编译缓存。**正确的修法是把补齐做进算子内部**（见 §4 待办），
让 python 侧不再有任何分支。

**预期**：拿掉这 33%，引擎口径应到 **1.5–1.8× / 层**。

### 0'.4 接入方式与它的代价

`npu_ops/python/vllm_engine_patch.py` 在**加载期**替换
`AscendUnquantizedLinearMethod.process_weights_after_loading`——必须在
`LLM(...)` **之前**打补丁，因为 ACL 图是在引擎初始化时捕获的，之后再换算子
捕获的图还指向 BF16 matmul（而且指向一块已被释放的权重）。

**它从 BF16 模型转，不用 `models/QwQ-32B-W4A8-Random`**：那个 checkpoint 是
厂商工具链 group_size=128 的产物，盘上已经是 int4，和我们 GK=1024 的契约对不上；
从它反解回 bf16 再量化是二次量化，精度比两边都差。从 BF16 转则与
`fakequant_lab` 逐位一致。**代价是加载期要放得下 BF16 权重**——这正是
第 1 步（权重导出器）要解决的。

**转换开销**：CPU 侧 `fakequant_lab` 量化，大层约 1.3–2.5 s，16 层 ≈ 20 s，
满 64 层预计 ~5 分钟（一次性）。

---

## 0''''. CKPT-E（2026-08-14）· **P4 冻结 · 转 P5**

**这一节是记号，不要再改数字。** P5 的一切「快了多少」都对这里比。
**报告已产出：`reports/midgroup/P4_e2e.md`。**

### E.1 冻结的基线（每层斜率 µs，QwQ-32B / TP=1 / **FULL_DECODE_ONLY** / 随机权重）

| arm | 1 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 原生 BF16 | 1072 | 1295 | 1372 | 1584 | 1776 | 1709 | 2163 | 2377 |
| 原生 W8A8 | 792 | 970 | — | 1209 | — | 1223 | — | — |
| 原生 W4A8 | 441 | 922 | 1873 | 1991 | 1967 | 2236 | 2363 | 2679 |
| 我们 W8A8 | 485 | 595 | 687 | 830 | 885 | 933 | 1240 | 1496 |
| **我们 W4A8** | **408** | **520** | **676** | **731** | **1008** | **986** | **1699** | **1816** |

复现（**图模式必须显式给**）：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_engine_decode.py \
    --arm w4a8 --layers 4  --batch 1 16 64 128 --out-len 64 \
    --cudagraph-mode FULL_DECODE_ONLY --out int4_cube_lab/results/engine_batch_sweep_5arm.csv
# 再跑 --layers 16，斜率 = (step@16 − step@4) / 12
```

### E.2 冻结时的代码状态

| 项 | 状态 |
|---|---|
| `npu_ops/kernel/*` | 与 `int4_cube_lab/kernels/*` **逐字节相同**（2026-08-14 核对 quant_a 两份）|
| `midgroup_quant_a` | **自己产出 Mp 行**（补齐在算子内），含 D11 的 `V_MTE2` 修复 |
| `midgroup_w4a8_gemm` | **本期未改动**；其 `Pad2D` 在 vLLM 路径上是 no-op |
| `vllm_midgroup_linear.py` | **零形状分支**（quant → gemm → `y[:rows]`）|
| `bench_engine_decode.py` | 五 arm + `--cudagraph-mode`，CSV 记录图模式 |
| 精度 | 与 `fakequant_lab` 逐位一致；**P2 门禁仍未过**（D1）|
| 两个 `.so` | `build/`（torch 2.7.1）与 `build-venv/`（torch 2.8.0）**均已重编**（D5）|

### E.3 P5 要打的靶子（**唯一的下一步**）

**batch≥96 我们输给自己的 W8A8（0.73–0.88×），原因是分组冲刷，不是 MSD。**

| 证据 | 出处 |
|---|---|
| mid-group 相对 per-channel int4 的税：M≤128 约 10%，**M≥512 涨到 1.34–1.47×** | `results/midgroup_w4a8_qwen.csv`（§0'''.67）|
| per-channel int4 在 M=1024 **仍赢 W8A8 1.12×** → 架构没问题 | 同上换算 |
| 引擎侧高并发劣势 0.73–0.88× | §0'''.65 |

**目标**：减少每个输出 tile 的 workspace 往返次数（现在 G=5~27 次，per-channel 1 次）。
**起点**：`P5_KERNEL_OPT.md` §4 第 2 步已作废——**从 M=512 的消融开始**，
不是 msprof（D4）。**先消融再改**（P5 D2）。

**硬门禁不变**：任何改动先过逐位对拍（P3 接缝测试 + `test_pad_regress.py`），
再看性能。

---

## 0'''. CKPT-D（2026-08-14）· **PIECEWISE 就是那个空隙 · 引擎口径 vs W8A8 1.32×**

**一句话**：把图模式从 vllm-ascend 默认的 `PIECEWISE` 换成 `FULL_DECODE_ONLY`，
**每层 878.5 → ~391 µs，vs BF16 1.41× → 2.78×，vs W8A8 第一次量到 1.32×**，
吞吐 47.4 → **96.5 tok/s**。**一个参数，不用写自定义 transformer 层。**

### 0'''.0 图模式：FULL 与 PIECEWISE 的区别，以及能不能上生产

| | PIECEWISE | FULL |
|---|---|---|
| `splitting_ops` | attention 类算子（含 `vllm::mla_forward`）| **空** |
| 捕获粒度 | 两个 attention 之间一段，各一张图 | 整个 forward **一张图** |
| attention | **图外 eager** | 图内 |
| 段间 | python | 无 |

`FULL_DECODE_ONLY` = decode 走 FULL、prefill/混合批走 eager。适合我们，
因为 decode 形状齐整。

**上生产的依据与风险，全部有源码出处**：

- **上游 vLLM v1 的默认值是 `FULL_AND_PIECEWISE`**，docstring 原话
  "the most performant mode for most models and is the default"。
- vllm-ascend 把它降级成 PIECEWISE，注释是
  `# TODO: Full graph is fully supported later, and the default value will be set to full graph.`
  ——**临时保守默认**，方向是转全图。
- **坑**：显式传 `FULL_AND_PIECEWISE` 会被**静默降级成 PIECEWISE**
  （`platform.py`）。NPU 上要全图**必须写 `FULL_DECODE_ONLY` 或 `FULL`**。
- 厂商自己的红色 WARNING：*early experimental stage… capturing too many batch
  sizes can lead to **OOM errors or inference hangs***。缓解手段是限制
  `cudagraph_capture_sizes` / 调低 `gpu_memory_utilization`。
- 代码里的硬限制：pooling 模型、encoder-decoder 强制 PIECEWISE，
  `enforce_eager` 强制 NONE。**都与 QwQ-32B 无关。**

**没验过的（不要当成已验证）**：TP>1、全 64 层、真实并发下的 capture size 数量、
prefill 回归。全部记在 §4 待办里，与第 4 步合并做。

**规矩：引用任何引擎数字必须带图模式**——同一套 kernel 两种模式是 1.41× 和 2.78×。

### 0'''.1 数据（`--layers 4/16` 取斜率，batch=1）

| 图模式 | BF16 µs/层 | W4A8-mg µs/层 | vs BF16 | 16 层吞吐 |
|---|--:|--:|--:|--:|
| PIECEWISE（默认）| 1234.5 | 878.5 | 1.41× | 38.1 → 57.7 tok/s |
| **FULL_DECODE_ONLY** | **1086.6** | **385.8** | **2.82×** | **47.4 → 97.4 tok/s** |

复现：`bench_engine_decode.py` 加 `--cudagraph-mode FULL_DECODE_ONLY`。

### 0'''.2 机制已证实：设备时间没变，空隙没了

同一个 arm、同样 16 层，只换图模式：

| 图模式 | Computing | Free | 设备利用率 |
|---|--:|--:|--:|
| PIECEWISE | 462.6 ms | 807.1 ms | 36% |
| **FULL_DECODE_ONLY** | **458.9 ms** | **326.0 ms** | **58%** |

**Computing 几乎不动（462.6 → 458.9），Free 掉了 481 ms**（7.5 ms/step）。
所以那 ~390 µs/层**全部是 PIECEWISE 的分段边界**——每层被 attention 切开、
段间跑 python——**与我们的算子无关**，也不是 D10 的 pad 在 host 侧的残留
（pad 本来就被捕获进图，每步不付 host 成本）。

### 0'''.3 两个交叉验证

- **引擎口径与手搭图口径终于对上了**：W4A8 每层 **385.8 µs（引擎）
  vs 376.6 µs（`bench_decode_graph.py`）**。CKPT-A 那套手搭数字从今天起
  可以当引擎的预测用了——**在 FULL 模式下**。PIECEWISE 下它们差 2.3 倍，
  这正是 D10 的「手搭图捕获 ≠ 引擎图捕获」在另一个维度上的复现。
- **CKPT-C 的补齐修复到这里才兑现**：pad 还在的话设备时间是 ~610 µs/层，
  FULL 模式下只能到 ~1.78×。**两件事叠起来才是 1.34× → 2.82×**，
  单独任何一件都不够。

### 0'''.4 W8A8 arm（引擎口径第一次量到）· **1.32× · 目标达成**

`bench_engine_decode.py --arm w8a8`：per-channel int8 权重 + per-token 动态激活
（`npu_dynamic_quant` + `npu_quant_matmul`），**从同一份 bf16 权重、在同一个
加载钩子上、对同一批 64 个 linear 转换**——所以这个比值隔离的是 GEMM，
不是 checkpoint、不是层数、不是转换路径。它与 `bench_decode_graph.py` 的
`w8a8` 是同一对算子，所以引擎口径和手搭口径可比。

**不是 `models/QwQ-32B-W8A8`**：那个 checkpoint 自带量化器和自己的层集合，
量它回答的是「厂商的构建快不快」，不是「我们的 GEMM 快不快」。是另一个问题。

| arm | 斜率 µs/层（3 次）| 均值 | 跑间离散 |
|---|---|--:|--:|
| BF16 | 1086.6 | 1086.6 | — |
| W8A8 | 518.8 / 513.6 / 510.2 | **514.2** | 1.7% |
| **W4A8-mg** | 378.5 / 400.1 / 394.5 | **391.0** | 5.5% |

**vs W8A8 = 1.32×（逐次 1.28 / 1.37 / 1.29，区间 1.28–1.37×）**，
vs BF16 = 2.78×。16 层吞吐 80.9 → 96.5 tok/s。

**与手搭口径对得上**：CKPT-A batch=1 是 w8a8 499.3 / w4a8 376.6 = 1.33×，
引擎 514.2 / 391.0 = 1.32×。**两个独立口径给出同一个数**，这是目前对
「1.3× 是真的」最强的一条证据。

**边界**：batch=1、16 层、TP=1、FULL_DECODE_ONLY。batch=128 那一档 CKPT-A 是
0.94×（进 cube-bound，D2 的按 M 分派仍然需要），**引擎里还没量过**。

### 0'''.5 五路对照（含 vllm-ascend 原生 W8A8 / W4A8）· **口径要说全**

用户 2026-08-14 要求把厂商原生路径也作为 baseline。`--arm` 现在有五个：

| arm | 是什么 | checkpoint |
|---|---|---|
| `bf16` | 原生 BF16 | `models/QwQ-32B` |
| `w8a8-native` | **vllm-ascend 自己的 W8A8** | `models/QwQ-32B-W8A8` |
| `w4a8-native` | **vllm-ascend 自己的 W4A8** | `models/QwQ-32B-W4A8-Random` |
| `w8a8` | 我们的 per-channel int8 | 从 `QwQ-32B` bf16 转 |
| `w4a8` | 我们的 mid-group W4A8 | 从 `QwQ-32B` bf16 转 |

两个 `-native` 走 `quantization="ascend"`（两个 config.json 都没有
`quantization_config`，方法写在 `quant_model_description.json` 里）。

**结果**（batch=1，FULL_DECODE_ONLY，4 层/16 层取斜率，n=2~3）：

| arm | µs/层 | 跑间离散 | vs BF16 | vs 源W8A8 | vs 我们W8A8 | **vs 源W4A8** | 64 层外推 step |
|---|--:|--:|--:|--:|--:|--:|--:|
| BF16 | 1080.4 | 1.1% | 1.00× | 0.74× | 0.48× | 0.41× | 72.8 ms |
| 源 W8A8 | 799.8 | 3.3% | 1.35× | 1.00× | 0.64× | 0.56× | 55.0 ms |
| 源 W4A8 | 446.0 | 0.4% | 2.42× | 1.79× | 1.15× | 1.00× | 32.8 ms |
| 我们 W8A8 | 514.2 | 1.7% | 2.10× | 1.56× | 1.00× | 0.87× | 37.0 ms |
| **我们 W4A8** | **391.0** | 5.5% | **2.76×** | **2.05×** | **1.32×** | **1.14×** | **29.2 ms** |

### 0'''.6 三条必须一起说的解读

**① 对「用户真能部署的东西」，M=1 上我们只快 1.08–1.14×。**
1.32× 的分母是**我们自己写的** W8A8；厂商的 W4A8 在这一档只慢一成左右。
**但这是最不利的一档**——§0'''.65 的扫描显示 batch=64 上是 **2.72×**。
**引用任何比值都必须带 M 档。**

**② 我们的 W8A8 比厂商 W8A8 快 1.56×，所以它是个偏乐观的分母。**
我们的实现很精简（`npu_dynamic_quant` + `npu_quant_matmul`，对称 per-channel，
不处理 offset / quant bias / NZ 转换），厂商那套做的事更多。
好的一面：拿它当分母，1.32× 是**保守**的；坏的一面：它不是部署形态。

**③ 但 1.14× 是 M=1 的数，不能外推成产品结论。**
（用户 2026-08-14 指出，本条据此改写；原文把它写成「战略风险」是下早了。）
**group_size=128 恰恰意味着对方走不了 cube**：每 128 个元素就要冲刷一次 L0C
并施加一次组 scale——这正是我们把 GK 定到 1024 的原因。所以厂商那条路
**只能反量化成 bf16**，它在 M=1 不吃亏纯粹是因为**带宽受限**。
**到了大 M，weight-only 要按 bf16 的算力去算，int8 的 2× 和 int4 的 4× cube
优势全部丢掉。** 我们的优势应该在那里显现。**§4 的 batch 扫描就是这一仗。**

**精度的事要单独记，别和上面混在一起**（原文混了）：我们的 GK=1024 没过 P2
门禁（GLUE −4.5）是**我们自己的**问题。它不构成竞争层面的风险——如果细粒度
本来就逼着对方走 weight-only，对方并没有拿到「又准又快」的免费午餐。

### 0'''.65 batch 扫描 · **用户的判断被证实：weight-only 在大 M 崩掉**

用户 2026-08-14 指出 §0'''.6 的 1.14× 不能外推：g128 逼着厂商只能反量化成 bf16，
M=1 不吃亏纯粹是带宽受限。**扫描证实了，而且幅度比预期大。**

每层斜率 µs（FULL_DECODE_ONLY，4/16 层取斜率）。
**batch 1/16/64/128 来自第一轮，32/96/192/256 来自第二轮**：

| arm | 1 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| BF16 | 1072 | 1295 | 1372 | 1584 | 1776 | 1709 | 2163 | 2377 |
| 源 W8A8 | 792 | 970 | — | 1209 | — | 1223 | — | — |
| 源 W4A8 | 441 | 922 | 1873 | 1991 | 1967 | 2236 | 2363 | 2679 |
| 我们 W8A8 | 485 | 595 | 687 | 830 | 885 | 933 | 1240 | 1496 |
| **我们 W4A8** | **408** | **520** | **676** | **731** | **1008** | **986** | **1699** | **1816** |

| 我们的 W4A8 vs | 1 | 16 | 32 | 64 | 96 | 128 | 192 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **源 W4A8** | 1.08× | 1.77× | **2.77×** | **2.72×** | 1.95× | 2.27× | 1.39× | 1.48× |
| 我们 W8A8 | 1.19× | 1.15× | 1.02× | 1.14× | **0.88×** | **0.95×** | **0.73×** | **0.82×** |
| BF16 | 2.63× | 2.49× | 2.03× | 2.17× | 1.76× | 1.73× | 1.27× | 1.31× |
| 源 W4A8 vs BF16 | 2.43× | 1.40× | **0.73×** | 0.80× | 0.90× | 0.76× | 0.92× | 0.89× |

> **⚠ 两轮之间不可直接比绝对值。** 每一轮内部单调（已核对），但第二轮
> 系统性偏高（例：BF16 在 batch=96 是 1776，在 batch=128 却是 1709）——
> 第二轮的最大 batch 是 256，捕获的图更多、KV 布局不同。
> **同一列内跨 arm 的比值是干净的**（同一轮、同一配置，只换 arm），
> **跨列的绝对值和精确交叉点不是**。要钉死交叉点需要**一个进程里跑完所有档**。

**读这张表只读三条稳的**：
1. **对源 W4A8 我们全程赢**，峰值在 batch 32–128（1.95–2.77×），
   到 192–256 回落到 ~1.4×；
2. **对我们自己的 W8A8，小 batch 赢（1.15–1.19×）、大 batch 输**
   （96 以上 0.73–0.88×），**交叉点在 32 和 96 之间**；
3. **对 BF16 全程 ≥1.27×**。

原始每层斜率（第一轮四档，供对照）：

| arm | batch=1 | batch=16 | batch=64 | batch=128 |
|---|--:|--:|--:|--:|
| BF16 | 1072.4 | 1295.0 | 1583.6 | 1708.6 |
| 源 W8A8 | 791.6 | 969.7 | 1208.9 | 1223.2 |
| **源 W4A8（weight-only）** | **441.4** | **922.3** | **1990.7** | **2235.8** |
| 我们 W8A8 | 485.2 | 595.3 | 830.4 | 933.0 |
| **我们 W4A8** | **407.8** | **519.6** | **731.0** | **986.5** |

比值：

| 我们的 W4A8 vs | batch=1 | batch=16 | batch=64 | batch=128 |
|---|--:|--:|--:|--:|
| **源 W4A8** | 1.08× | **1.77×** | **2.72×** | **2.27×** |
| 我们 W8A8 | 1.19× | 1.15× | 1.14× | 0.95× |
| BF16 | 2.63× | 2.49× | 2.17× | 1.73× |

**源 W4A8 相对 BF16：2.43×（1）→ 1.40×（16）→ 0.73×（32）→ 之后一直 0.76–0.92×。**
**在 batch 16 和 32 之间穿过 BF16，之后再也没回来。** 这正是 weight-only 的
特征曲线：M=1 带宽受限时它拿满收益，M 一大就得按 bf16 的算力算，
反量化本身还要额外开销。

**结论：我们相对货架产品的优势是 1.08×（M=1）到 2.77×（batch=32），不是 1.14×。**
1.14× 是最不利的那一档。

### 0'''.66 D3 我改过头了 —— 原文在它自己的口径下是对的

§0'''.6 里我写「D3 的性能结论错」，**那句话本身只在 M=1 成立**。
D3 说「W4A8 在 Phase 3 基线里比 BF16 还慢」——扫描显示 batch ≥64 确实如此
（0.80× / 0.76×）。**D3 没错，是我拿 M=1 一个点去推翻一个不同 M 下的观察。**
两次都是同一个错误：**单工作点外推**。

**规矩：任何「X 比 Y 快/慢」的结论必须带 M 档。** 这个项目里
weight-only、mid-group、per-channel 三条路的排序**在 M 上是会换位的**。

### 0'''.67 batch=128 输给 W8A8 的原因 · **是分组冲刷，不是 MSD**

engine 扫描显示 batch=128 我们对自己的 W8A8 是 0.95×。**先别急着做按 M 分派**——
P3 的 kernel 级数据（`results/midgroup_w4a8_qwen.csv`）已经指出原因了：

**mid-group 相对 per-channel int4（同样是 MSD 拆分，只是没有分组冲刷）：**

| M | qkv | o | gate_up | down |
|--:|--:|--:|--:|--:|
| 1–16 | 1.09–1.10× | 1.08–1.10× | 1.12× | 1.05–1.06× |
| 64 | 1.17× | 1.13× | 1.25× | 1.10× |
| 128 | 1.05× | 1.05× | 1.07× | 1.20× |
| **512** | **1.42×** | **1.40×** | **1.34×** | **1.40×** |
| **1024** | **1.47×** | **1.45×** | **1.42×** | **1.43×** |

（>1 = mid-group 更慢。）**解码档 ~10% 的税，到 M≥512 涨成 40–47%。**

**换算一下就知道架构没问题**：M=1024 时 mid-group 对 W8A8 是 0.76×（qkv），
但 per-channel int4 对 W8A8 是 0.76 × 1.47 = **1.12×**，仍然赢。
所以 **int4 cube 的 2 倍吞吐确实抵掉了 MSD 的两次 MAD**——
输的不是 MSD，是**每个量化组一次 workspace 往返**（G = K/1024 = 5~27 次冲刷，
per-channel 只有 1 次）。小 M 时 DMA 受限把它藏住了，大 M 时 cube 忙起来就露出来。

**这同时说明 P5 停早了**：它的消融只在 **M=16** 做过（那里删掉 AIV / Mmad /
Fixpipe 时间不变，所以判定 DMA-bound、只剩 25 µs）。**大 M 从没看过**，
而那里的可回收量是 40%+。`P5_KERNEL_OPT.md` 的「暂停」结论只对解码档成立。

**引擎侧独立印证了同一件事**：§0'''.65 的密扫描显示我们对自己的 W8A8
在 batch≥96 掉到 **0.73–0.88×**。两条独立证据（kernel 级的 mid-group 税、
引擎级的高并发劣势）指向同一个机制。

**推论：不需要按 M 分派，需要的是大 M 上少冲刷几次。** 这也顺带解掉了
「分派要留两份权重 vs 转换后丢 bf16 省内存」的矛盾——根本不用分派。
**动手前先做 M=512 的消融**（P5 D2：别猜，做消融）。

### 0'''.7 这对「自定义 transformer 层」的影响

**主机侧的收益已经被这个参数拿走了大部分**：W4A8 每层 385.8 µs 对设备时间
367 µs，**只剩 ~19 µs 的空隙**。自定义 transformer 层原本的目标（压主机开销）
**没有空间了**，不要再按那个理由做。

**新的瓶颈回到设备侧**，也就是说：
- P5 的 kernel 优化重新变得有意义（那 25 µs 现在能到墙上时间了）；
- 算子融合的 −21 µs 同理，但它仍然是小项；
- **最该先做的是把 W8A8 arm 加进引擎基准**——「超越 W8A8」到现在还没在
  引擎口径下量过一次，而 2.82× 是对 BF16 的。

---

## 0''. CKPT-C（2026-08-14）· **D10 已修，但引擎是主机受限的**

**一句话**：补齐已经搬进算子，PadV3 / MemSet 在引擎 profiler 里**归零**，
投影的每层设备时间 **568 → 321 µs（−44%）**；但引擎口径只从 1.34× 走到
**1.41×**，因为**这台引擎在 16 层 / batch=1 下是主机受限的**——设备利用率
w4a8 只有 36%。**下一步不再是 kernel，也不是算子融合，是把每层的主机开销压下去。**

### 0''.1 做了什么

`midgroup_quant_a` 的 host 自己按 `TileMFor(M)` 算 `Mp`，**直接分配 Mp 行的四个
输出**，kernel 对 `row >= M` 的行按零输入量化。于是：

- GEMM host 的四个 `Pad2D` 天然 no-op（`TileMFor` 两边是同一个函数）；
- python 侧**没有任何形状分支**了（`vllm_midgroup_linear.py` 只剩 quant → gemm →
  `y[:rows]`，切片是 view，不是 kernel），D10 的坑从结构上消失；
- 不需要按文档原方案改 GEMM 的 group-major stride，**GEMM 一行没动**。

### 0''.2 顺带挖出一个真 bug（D11）

kernel 里零填充 `ubX`（V）与随后 `DataCopyPad` 装载 `ubX`（MTE2）之间**只有
`PipeBarrier<PIPE_V>`**，缺 V→MTE2 同步，零填充会把刚装载的行冲掉。
**旧设计一直把它藏着**：调用方总把 M 补成 kBr 的倍数，`rows < kBr` 那条分支
永远不执行。补齐搬进算子、M=1 的 ragged job 上了解码热路径，它立刻暴露成
**恰好 (M mod 16) 行不对**。已加 `V_MTE2` flag，并把它提到无条件执行——
同一个 hazard 还有第二处（上一个 job 对 `ubX` 的 V 读 vs 本 job 的 MTE2 写）。

### 0''.3 数值门禁（全过）

| 检查 | 结果 |
|---|---|
| 对**改动前的 .so** 逐位对拍（3 形状 × 14 个 M，含 1/17/33/65/100/129/200）| **42/42 逐位相同** |
| lab `--check`（M=1/3/16/17/64/100/128/1024，K=5120）| 8/8 **byte mismatches=0** |
| 图捕获整层（CKPT-A 口径，batch=1）| 376.6 µs vs 冻结值 378.4（重复区间 376.7–382.4）**无回归** |

对拍脚本的坑：两次进程用 `hash(str)` 生成输入——**python 对字符串哈希每进程加盐**，
于是两边喂的根本不是同一组数据，第一版 42/42 全 FAIL。改成固定 seed 才有效。

### 0''.4 引擎口径（`--layers 4/16` 取斜率，batch=1，ACL 图开）

| arm | 4 层 µs/step | 16 层 µs/step | **斜率 µs/层** | vs BF16 |
|---|--:|--:|--:|--:|
| BF16 | 8567.1 | 23381.3 | **1234.5** | — |
| W4A8-mg（CKPT-B，补齐回退）| 7362.1 | 17947.0 | 882.1 | 1.34× |
| **W4A8-mg（本次）** | 6804.0 | 17345.5 | **878.5** | **1.41×** |

吞吐 38.1 → 57.7 tok/s @16 层（另一次跑到 62.7，**同配置跑间波动 ~8%**，
引用时请给区间）。

### 0''.5 为什么设备省了 44%、墙上只涨了 0.07×——**引擎主机受限**

profiler（63 个解码步 × 16 层）：**PadV3 = 0，MemSet = 0**（CKPT-B 是 3840 + 3840）。

| kernel | 次数 | µs/层/步 |
|---|--:|--:|
| midgroup_w4a8_gemm_m16 | 4032 | 284.0 |
| midgroup_quant_a | 4096 | 37.2 |
| FusedInferAttentionScore | 1024 | 20.4 |
| AddRmsNormBias / SwiGlu / RoPE / 其余 | — | ~25 |
| **解码设备合计** | | **~367** |

（`quant_a` 从 4.8 → 9.2 µs/次：它现在真的算 16 行而不是 1 行。
用 37 µs 换掉 249 µs 的 pad，净赚。`MatMulV2` 64 次 × 1234 µs 是 **lm_head**，
1.56 GB bf16，每个解码步一次——**不在层里，但它是每步固定开销的大头**。）

`step_trace_time.csv`：

| arm | Computing | Free | 设备利用率 |
|---|--:|--:|--:|
| BF16 | 1150.6 ms | 523.1 ms | **69%** |
| **W4A8-mg** | **462.6 ms** | **807.1 ms** | **36%** |

扣掉 profiler 自身的放大（用非 profile 的 step 时间算）：**两个 arm 的主机空隙都是
~8.5 ms/step**。拿 4 层和 16 层的差分拆开：**~390 µs/层是主机空隙，~367 µs/层是设备**。
**大约五五开。**

**推论（这条决定后面所有优先级）**：即使 GEMM 免费，每层也只能从 878 掉到 ~490 µs，
引擎口径封顶在 ~2.5×。**剩下的一半必须从主机侧拿**，而这正是自定义 transformer 层
要解决的问题——**不是算子融合能解决的**（融合只值 21 µs，见 §8.2）。

**注意 ACL 图确实在生效**，不要往「图没开」的方向查：`--eager` 下 w4a8 是
32311 µs/step，图捕获下 15943 µs/step，**2.03×**。日志有 `Replaying aclgraph`，
64 个 linear 全部转换。空隙是 **PIECEWISE 分段**留下的——每层被 attention 切开，
段与段之间是 python。

**下一步的两个候选，先做便宜的那个**：

1. **先试 FULL / `FULL_DECODE_ONLY` 图模式**（vllm-ascend 默认 PIECEWISE，
   平台日志明写 `PIECEWISE compilation enabled on NPU`）。整模型一张图的话
   每层的 python 胶水直接归零，**改一个 `compilation_config` 就能量**。
2. **自定义 transformer 层**（用户 2026-08-14 定的方向）：如果 1 不成立或不够，
   就自己写整层，把 op 数和分段数一起压下去。**目标是主机空隙，不是设备时间。**

---

## 1. 当前状态

> **2026-08-14：本节写于 P4 启动时，已过期。当前状态看 CKPT-E（§0''''）。**
> 下面的「一句话给接手的人」是**滚动更新的，仍然有效**。

（原文）刚启动。P3 已交付经验证的 kernel（`MGCKPT/P3_KERNEL.md`）：GEMM 与激活量化都对
`fakequant_lab` 逐位/逐字节一致，decode 档相对 W8A8 **1.51–1.53×**（含量化开销）。

### 一句话给接手的人

> **P4 是 at-risk 交付，和 P3 同理**：P2 精度门禁没过（GLUE −4.5 点），
> 所以端到端的性能数字只能算工程验证，**不能作为方案可用的结论**（D1）。
>
> 四步里第 2 步（包成 `torch.ops.npu.*`）有 nitro 的现成模板，是最先动的。
> 第 1 步（产出真实 checkpoint）不依赖 P2 的未决项——主线算法已定为 RTN 非对称。
> **第 3 步已跑通（CKPT-B / §0'）**，**D10 的补齐回退已修（CKPT-C / §0''）**，
> **图模式换成 `FULL_DECODE_ONLY`（CKPT-D / §0'''）**，五路对照已跑完。
> **口径要说全，否则会高估自己**：
> - vs BF16 **2.76×**、vs 我们自己的 W8A8 **1.32×**；
> - **vs 厂商 W4A8：全程赢**，1.08×（M=1）→ 峰值 **2.77×（batch=32）**
>   → 1.4–1.5×（192–256）（§0'''.65）。厂商那条是 weight-only + g128，
>   只能反量化成 bf16：M=1 带宽受限时它拿满收益，**batch≥32 起比 BF16 还慢**。
>   **M=1 是我们最不利的一档，别拿它当结论；也别拿 2.77× 当结论。**
> - **vs 我们自己的 W8A8：小 batch 赢 1.15–1.19×，batch≥96 输到 0.73–0.88×。**
>   原因是**分组冲刷**，不是 MSD（§0'''.67）——per-channel int4 在 M=1024
>   仍然赢 W8A8 1.12×。**所以修 kernel，不要做按 M 分派。**
> - vs BF16 全程 ≥1.27×。
>
> **跑引擎基准一律带 `--cudagraph-mode FULL_DECODE_ONLY`**，默认的
> PIECEWISE 会白亏 2 倍；**引用任何引擎数字都要带图模式和分母**。
> **下一步是 batch 扫描**：D3 的修正给出可证伪推论——我们的优势应在大 M 才显现，
> 而 CKPT-A 说 batch=128 对 W8A8 是 0.94×。**这一仗决定方案的上限。**
>
> **看数字只看 §8，别看 §7。** §7 的 eager 基准是 host-bound 的（D7），
> 还用了会把 KV head 展开 5 倍的 attention（D8），两个坑都往「量化没用」的方向偏。
> 图捕获口径下是 **vs BF16 1.45–2.52×、vs W8A8 0.94–1.30×**（batch=1 的 1.30×
> 含 D9 的补齐修复）。§8.5 猜的「分组流水」已被 P5 的消融证伪：GEMM 是纯
> GM→L1 搬运受限，剩余空间只有 ~25 µs（`P5_KERNEL_OPT.md` §3'）。

---

## 2. 路线图 §8 的四步，按依赖排序

| 步 | 内容 | 阻塞情况 |
|---|---|---|
| 2 | AscendC kernel → `torch.ops.npu.*` | **不阻塞**，nitro 有现成模板（§3）|
| 1 | 产出真实校准的 W4A8 checkpoint | **不阻塞**，主线算法 = RTN 非对称 GK=1024 |
| 3 | 接入 vLLM 的 Linear，替掉 `npu_weight_quant_batchmatmul` | 依赖 1、2 |
| 4 | 对 `ROADMAP.md` Phase 3 冻结的 TP4 矩阵重跑 | 依赖 3 |

---

## 3. 第 2 步的模板：nitro 已经把同类 kernel 接进 torch 了

`nitro-workspace/nitro/nitro/csrc/` 是**在本环境里验证过的**完整链路：

- `host/*.cpp`：每个算子一个 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`
- `CMakeLists.txt`：`ascendc_library()` 编 kernel + 一个 `.so` 聚合 host 绑定
- `setup.py`：条件构建，产出 `libnitro_npu_ops.so`，import 时注册 `torch.ops.npu.*`
- **`host/mid_group_gemm.cpp` 就是同构算子的绑定**（int8 mid-group GEMM），
  连 host 侧对 M/N 补到 128、K 补到组倍数再切回来的做法都可以直接抄。

**决定：新建 `vllm/npu_ops/`，照 nitro 的结构建，不往 nitro 里加。**
理由：nitro 是训练侧项目，这两个 kernel 是推理侧的；混在一起会让两边的构建门禁纠缠。

---

## 4. 待办

- [x] `vllm/npu_ops/` 脚手架 + 两个 host 绑定，**`.so` 构建通过、算子注册成功**
      （`torch.ops.npu.midgroup_{w4a8_gemm,quant_a}` 可调用并返回）。
- [x] **segfault 已定位并修复**（D4）——`utils.h` 的 UB，只在 Release 下发作。
- [x] 用 `torch.ops.npu.*` 与 `fakequant_lab` 对拍：M=64/N=512/K=5120
      **32767/32768 逐位相同，SNR 174.79 dB**，与 standalone 接缝测试同一个数。
- [ ] **第 1 步**：权重导出器——把 QwQ-32B 按 RTN 非对称 GK=1024 量化，落成 kernel
      要的布局：`w_q` packed int4、`w_scale[G,N]` bf16、`w_zero[G,N]` bf16、
      `w_ksum[G,N]` int32（group-major）。替掉 `models/QwQ-32B-W4A8-Random`。
- [x] **整层 decode 基准**（`npu_ops/python/bench_decode_arms.py`）——四 arm 对照。
      ~~这是本轮最重要的产出~~ **它的口径是错的（D7、D8），结论见 §7 的作废标记。**
- [x] **图捕获口径重测**（`bench_decode_graph.py` + `bench_proj_graph.py`）——
      **vs BF16 1.45–2.52×、vs W8A8 0.94–1.30×**，见 §8。自定义 W4A8 算子
      **可被 `torch.npu.NPUGraph` 捕获**（已验证），这是能这么测的前提。
- [x] **覆盖面结论**：qkv/o **应该开**（§8.3），不是回退；`QKV_O_MAX_M` 保持不动，
      按 §8.1 改成「batch ≤64 全开、≥128 回退」即可。
- [x] **去掉 `F.pad`**（D9）——batch=1 整层 492.9 → **378.4 µs**，
      vs W8A8 从 1.03× 到 **1.30×**，逐位不变。**当前 e2e 最好的一次单项改动。**
- [~] **GEMM 分组流水** —— **转 `P5_KERNEL_OPT.md`，且假设已被推翻**：消融显示
      AIV / Mmad / Fixpipe 全删掉时间不变，是纯 GM→L1 搬运受限；冷口径下
      可回收量从 −62 µs 修正为 **−25 µs**。
- [x] **主机空隙探针** —— `--cudagraph-mode FULL_DECODE_ONLY` 一个参数，
      **每层 878.5 → 385.8 µs，1.41× → 2.82×**（CKPT-D）。Computing 不变、
      Free 掉 481 ms，证实空隙就是 PIECEWISE 的分段边界。
- [~] **自定义 transformer 层** —— **暂缓，理由已变**。它原本的目标是压主机
      开销，而 FULL 模式已经把每层空隙压到 ~19 µs（385.8 wall vs 367 device），
      **没有空间了**。要重开必须先给出新的、设备侧的理由。
      同理**算子融合的 −21 µs 仍然是小项**，不是第一优先级。
- [x] **W8A8 arm 加进 `bench_engine_decode.py`** —— 已完成（§0'''.4）。
      **引擎口径 vs W8A8 = 1.32×（1.28–1.37×，3 次），vs BF16 2.78×**，
      与手搭口径的 1.33× 一致。**「超越 W8A8」在引擎里第一次拿到证据。**
- [x] **五路对照**：加了 `w8a8-native` / `w4a8-native` 两个 arm（§0'''.5）。
      **关键发现：对厂商 W4A8 我们只快 1.14×**，不是 1.32×（那个分母是我们
      自己的 W8A8）。D3 的性能结论已修正。
- [x] **batch 扫描** —— 已完成（§0'''.65）。**推论成立**：vs 厂商 W4A8
      1.08× → **2.72×（batch=64）**；weight-only 在 batch≥64 比 BF16 还慢。
      同时 vs 我们自己的 W8A8 在 batch=128 是 0.95×（与 CKPT-A 的 0.94× 一致）。
- [~] ~~按 M 分派接进 `vllm_engine_patch`（D2）~~ —— **先别做**。§0'''.67 显示
      batch=128 输给 W8A8 的原因是**分组冲刷**，不是 MSD：同样做 MSD 的
      per-channel int4 在 M=1024 仍然赢 W8A8 1.12×。**修 kernel 比分派更值**，
      而且不用面对「留两份权重 vs 丢 bf16 省内存」的矛盾。
- [ ] **P5 重开，但这次在 M=512 做消融（现在的第一优先级）**：
      P5 的「只剩 25 µs、暂停」是**只在 M=16 消融**得出的（那里 DMA-bound）。
      大 M 的 mid-group 税是 **40–47%**，量级完全不同。
      目标：减少每个输出 tile 的 workspace 往返次数。**先消融再改**（P5 D2）。
- [ ] **一个进程里跑完所有 batch 档，钉死交叉点**：现在的曲线拼自两轮，
      第二轮系统性偏高（最大 batch 不同 → 捕获的图和 KV 布局不同），
      所以**同列跨 arm 的比值可信，跨列的绝对值和精确交叉点不可信**。
      现在只能说交叉点在 32 和 96 之间。
- [ ] **batch=1 的跑间波动要收敛**：同一对 arm 在不同轮次给出 1.19× / 1.32×
      （w8a8 斜率 485–514、w4a8 391–408）。大 M 的趋势是压倒性的，不受影响；
      但**引用 batch=1 的数必须给区间或多跑几轮**。
- [ ] **`--cudagraph-mode` 的生产验证**：TP>1、全 64 层、真实并发下的
      capture size 数量（厂商点名的 OOM/hang 触发条件）、prefill 回归
      （`FULL_DECODE_ONLY` 的 prefill 走 eager）。**目前一条都没验过**，
      与第 4 步（TP4 矩阵）合并做。
- [x] **第 3 步 · 已跑通**（CKPT-B / §0'）：`vllm_engine_patch.py` 在加载期接管
      linear，ACL 图正常 replay，端到端能生成。**引擎口径 1.34× / 层**。
      顺带修了两个卡点：算子缺 fake/meta 实现（dynamo 追踪报
      `Operator does not support running with fake tensors`）、
      补丁要挂在 `AscendUnquantizedLinearMethod` 而不是 vLLM 基类。
- [x] **补齐做进算子内部** —— 已完成（CKPT-C）。`midgroup_quant_a` 直接产出
      Mp 行，python 侧零分支，**PadV3 / MemSet 在引擎 profiler 里归零**，
      投影每层设备时间 568 → 321 µs。GEMM 一行没动（不需要改 group-major stride）。
      **但引擎口径只到 1.41×，不是预期的 1.5–1.8×**——预期错在假设引擎是设备受限的，
      实际是主机受限（CKPT-C §0''.5）。顺带修了 kernel 的 V→MTE2 竞态（D11）。
- [ ] **`patch_w4a8_dynamic()`**（走 vllm-ascend 的 W4A8 量化路径）仍未验证；
      现在的接入走的是 BF16 转换路径，两者是不同的入口。
- [ ] **第 4 步**：TP4 矩阵重跑，出 BF16 / W8A8 / W4A8-mg 三方对比。

---

## 5. 决策与坑

### D1 · P4 同样是 at-risk 交付

P2 没过（PPL 过、GLUE −4.5 点），P3、P4 都是 2026-08-13 并行决策的产物
（`P2_ACCURACY.md` D8）。**P4 的端到端数字是工程验证，不是方案结论。**
路线图 §8 的 Exit Criteria 第三条「P2 的任务评测在端到端上复现」**现在无法核对**，
因为 P2 的任务门槛还没定。这条要么等 P2，要么在验收时显式标注为未核对。

### D4 · `.so` 里所有算子 segfault —— **已解决：`utils.h` 的 UB 在 -O3 下发作**

**现象**：`.so` 构建通过、`torch.ops.npu.*` 注册成功、调用能返回，但随后
`torch.npu.synchronize()`（或 `ASCEND_LAUNCH_BLOCKING=1` 时的调用本身）**段错误**。
**plog 里没有任何运行时错误、没有 AICore 异常**——是纯主机侧崩溃。

**对照组成立**：nitro 自己的 `torch.ops.npu.mid_group_gemm` 在同一进程、同一环境下
**正常跑通**。所以 torch_npu / CANN / 加载时机 / `TORCH_LIBRARY` 机制都没问题。

**已排除**（逐项实测）：

| 假设 | 结果 |
|---|---|
| cxx11 ABI 不一致（nitro 注释警告过的坑）| 排除，两边都是 1，与 torch 一致 |
| kernel 设备端二进制没生成 | 排除，静态库大小与 `int4_cube_lab` 的一致（1.30 MB vs 1.30 MB）|
| 用带 `..` 的路径引用 kernel 源码 | 排除，拷到本地目录重建后仍崩 |
| `TORCH_LIBRARY_FRAGMENT` 放在具名命名空间导致 ODR 冲突 | **本身是真 bug，已修**，但不是崩溃主因 |
| 对 `ascendc_library` target 调 `target_include_directories` | 排除，去掉后仍崩 |
| 核任务类型没声明 | GEMM **已经**声明了 `MIX_AIC_1_2` 却同样崩，所以至少不是唯一原因 |

**二分实验定了案**：把 nitro 的 `rstd_reduce.cpp`（在 nitro 自己的 `.so` 里跑得好好的）
拷进 `npu_ops/kernel/`，用我这套 CMake + host 绑定 launch——**它也崩**。
于是问题与我的 kernel 无关，锁定在 `.so` 组装。范围一缩小，差异就一眼可见：

**根因**：`utils.h` 的 `EXEC_KERNEL_CMD` 里那个 lambda 声明 `-> int` 却**没有 return 语句**
（编译时一直在报 `warning: no return statement in function returning non-void`）。这是 UB：

- **nitro 用 Debug 构建**（它的 CMakeLists 里 `CMAKE_BUILD_TYPE "Debug" FORCE`），
  `-O0` 下只是返回寄存器里的垃圾值，恰好无害；
- **我用 Release**，`-O3` 下 GCC 把 fall-through 当**不可达**处理，
  launch 直接变成主机侧 SEGV——**没有 ACL 错误码、plog 里什么都没有**，
  这正是最难查的形态。

**修法**：让 lambda 显式返回 launch 的状态（已改 `npu_ops/host/utils.h`）。
改完 nitro 的 kernel 和我的两个 kernel 在 Release 下全部正常。

**这是 nitro `utils.h` 里的潜伏 bug**，它自己没暴露只因为一直是 Debug 构建。
若 nitro 哪天切 Release 会遇到同样的坑——值得回传。

**方法论**：这一轮我先猜了五次（ABI、kernel 二进制、源码路径、ODR、include 目录），
全错；二分实验一次定位。**又一次印证 P3 D9：别猜，做消融/二分。**

### D11 · 「调用方总是补齐」把 kernel 里的一个竞态藏了整整一期

`midgroup_quant_a.inc` 的零填充（`Duplicate`，V 管道）与随后装载 `ubX` 的
`DataCopyPad`（MTE2）之间只有 `PipeBarrier<PIPE_V>`。**PIPE_V 只排 V 对 V**，
MTE2 可以在填充还在飞的时候就开始写，于是零把刚装载的行冲掉。

**它为什么一直没被发现**：调用方（python 和 lab harness 两边）总是先把 M 补到
kBr 的倍数再进 kernel，`rows < kBr` 那条分支**从来没执行过**。把补齐搬进算子、
M=1 的 ragged job 上了解码热路径，它当场现形——而且现形得很干净：
**恰好 (M mod 16) 行不对**，M=17/33/65/129 各错 1 行，M=100 错 4 行，M=200 错 8 行。

**判据**：错的行数等于 `M mod TILE`，就该怀疑 ragged 分支，而不是算法。

**同类第二处**：上一个 job 对 `ubX` 的 V 读 vs 本 job 的 MTE2 写，也只有
MTE3_V 挡着（挡不住）。所以 `V_MTE2` flag 提到了**无条件执行**，两处一起覆盖。

**规矩**：**一条「上层保证了 X」的不变量，会让 kernel 里依赖 !X 的那条路径永远
不被测试。** 搬动这类不变量之前，先想清楚哪条死代码要活过来。

### D10 · python 里的形状分支在 dynamo 追踪下会被折掉 —— **已从结构上消除**

> **修法（2026-08-14, CKPT-C）**：补齐已经搬进 `midgroup_quant_a` 的 host，
> python 侧不再有任何形状分支，PadV3 / MemSet 在引擎 profiler 里归零。
> 下面的记录保留，因为**判据和规矩仍然有效**。


`MidGroupW4A8Linear.__call__` 的 `if mp != m:`（把 M 补到 TILE_M）**在 eager 下
正常执行，在 vLLM 的图捕获下不执行**。后果不是错误——GEMM host 的 `Pad2D`
兜住了正确性——而是**悄悄退化成 D6 明确要避免的四张量补齐**，占掉解码期
33% 的设备时间，且**只在引擎里发生**。

**判据**：同一个 arm 分别用 `--eager` 和图捕获跑 profiler。eager 下 PadV3 = 0，
图捕获下 PadV3 = 3840。**两边算子构成不一样，就说明有 python 逻辑没进图。**

**试过无效**：`int(x2.shape[0])` 强制特化、`VLLM_DISABLE_COMPILE_CACHE=1`。

**规矩**：**给引擎用的算子包装里不要放依赖形状的 python 分支。**
形状相关的决策要么在 C++ host 里做（运行期一定执行），要么做成算子参数。
`vllm_midgroup_linear.py` 里那个分支的注释已标注此坑。

**更普适的一条**：CKPT-A 的手搭基准用 `torch.npu.NPUGraph` 直接捕获，
**不经过 dynamo**，所以量不到这个问题。**手搭图捕获 ≠ 引擎图捕获**，
接引擎前不要把前者当成"已经验过图捕获了"。

### D9 · `F.pad` 在解码档是 13 µs 的定额，换成常驻缓冲的 `copy_` 就没了

M=1 要补齐到 kernel 的 TILE_M=16。原来用 `torch.nn.functional.pad`，
逐算子成本表（§8.4）显示它 **K=5120 和 K=27648 都是 12.1–12.5 µs**——
**与数据量无关，是下发/固定成本**（160 KB 算 13 µs 只有 12 GB/s，不可能是带宽）。
换成「常驻零缓冲 + `buf[:m].copy_(x)`」后测不出来（0.2–0.6 µs）。

四个投影合计 **−53 µs**，但整层实测从 492.9 → 378.4 µs（**−114 µs**，重复 3 次稳定）。
多出来的一半应该是少了每次调用的临时分配和随之而来的依赖，没有再细拆。

**输出逐位不变**：3 组形状 × 7 个 M（1/3/16/17/64/100/128）对 `F.pad` 版本
`torch.equal` 全 True，且连续两次调用结果一致（共享缓冲不会漏旧行）。

缓冲按 `(mp, K)` 全模型共享，不是每层一份：它在同一条流里立刻被 `midgroup_quant_a`
消费，流内不重排，所以一份就够；每层一份的话光 `down` 就是 0.9 MB × 64 层。
只写 `[0, m)`，尾部永远是零。

**推广**：解码档的小算子在这台机器上普遍是定额成本——kernel 下发 ~4 µs、
单算子图 replay ~65 µs、`F.pad` ~13 µs、`copy_` ~0。**能预分配就别现分配。**

### D7 · **eager 口径的整层基准全是 host-bound，数字不能用**

`bench_decode_arms.py` / `bench_decode_layer.py` 在 batch ≤16 时量的**不是设备时间**。
判据很简单——把「主机下发完 N 次迭代」和「设备排空」分开计时
（`issue` vs `wall`，脚本见 §6）：

| batch | arm | issue µs | wall µs | 判定 |
|--:|---|--:|--:|---|
| 1 | nonproj（四投影全换成常量）| 925.4 | 929.6 | **HOST** |
| 1 | bf16 | 1125.8 | 1148.0 | **HOST** |
| 1 | ours-w4a8 | 1387.6 | 1390.4 | **HOST** |
| 64 | bf16 | 990.4 | 3303.0 | dev |

**一个投影都不算的空层，主机也要 925 µs 才下发得完。**§7 那 854 µs 的
「非投影占比」量的是 python 下发，设备几乎全程闲着。这也解释了 §7 里对不上的账：
逐投影表（bf16 batch=1 合计 850 µs）+ 非投影 854 µs > 整层 1159 µs——
**分量加起来超过总量，就是在说它们重叠**，那一刻就该停下来查口径。

危害是**有方向性**的，不是噪声：量化路径每个投影多 3–4 次下发（pad / quant /
gemm / slice），所以 host-bound 下**它必然显得更慢**，正好把真实收益抹平甚至反号。

**规矩：torch_npu 上任何整层/多算子基准，一律图捕获后 replay 计时**，
或至少报 `issue` 与 `wall` 两个数。单算子基准也要注意：图 replay 本身有
~60 µs 下发地板，比这更快的算子量不出来（§8.4 的括号）。

### D8 · SDPA 的 `enable_gqa` 把分母吹大 3 倍

§7 的层里用 `F.scaled_dot_product_attention(..., enable_gqa=True)`，
它把 8 个 KV head 展开成 40 个再算；vllm-ascend 走的是
`npu_fused_infer_attention_score`，原生 GQA：

| batch | SDPA(enable_gqa) | fused_infer_attention | KV 字节 | 带宽下界 |
|--:|--:|--:|--:|--:|
| 1 | 68.2 | 111.0 | 4.2 MB | 2.6 µs |
| 64 | 805.1 | 271.5 | 268 MB | 168 µs |
| 128 | 1570.5 | 548.7 | 537 MB | 336 µs |

（batch=1 两列都被下发地板压住，不代表设备时间。）
batch ≥64 那一档 **3 倍的 attention 开销直接进了整层分母**，
让所有量化 arm 的比值往 1 靠。**基准里的非投影部分必须用引擎真用的算子**，
否则量的是自己搭的那套的缺点。

### D5 · 两套 torch 环境，`.so` 必须各编一份

系统 python 是 torch 2.7.1，vLLM 在 `.venv` 里是 **torch 2.8.0 + torch_npu 2.8.0.post2**。
按另一个 torch 编的 `.so` **能加载但 dispatch key 会错**——nitro 那份在 venv 里
把 impl 注册到了 `MAIA:` 而不是 `PrivateUse1:`，于是 npu 张量被当成 CPU 张量拒绝。
`w4a8_ops._lib_path()` 按运行解释器的 torch 版本自动选 `build/` 还是 `build-venv/`。

### D6 · 别在 host 侧补齐量化后的四个张量

第一版在 GEMM host 里对 `a_hi/a_lo/a_scale/a_ksum` 各做一次 `Pad2D`，M=1 时四次
`constant_pad_nd` 比 GEMM 本身还贵（195 µs vs M=16 的 92 µs，**同样的填充后工作量**）。
改成在**量化之前**只填充 `x` 一个张量，M=1 降到 164 µs。

### D2 · 按投影分派（P3 §4.7 的交代）

P3 最终扫描发现 **prefill 档 `qkv`/`o` 用 mid-group 比 W8A8 慢**（M=1024 为
0.76×/0.79×），合计 >1 全靠 `gate_up`/`down` 撑。**vLLM 接入时应按投影和 M 档分派**：
decode 全用 mid-group；prefill 的 qkv/o 走 per-channel W4A8 或 W8A8。
不分派的话 prefill 会白亏。

### D3 · 现有 vLLM W4A8 路径不是真 W4A8 —— **机制对，性能结论错（2026-08-14 修正）**

`reports/W4A8_PERGROUP_ANALYSIS.md`：现在的 `npu_weight_quant_batchmatmul` 是
**weight-only、反量化成 bf16 再算**，激活的 int8 从未进 GEMM。所以第 3 步是**替换**
而不是复用它。

**机制在 v0.13.0 仍然成立**（`vllm_ascend/quantization/w4a8_dynamic.py:163`
还是 `npu_weight_quant_batchmatmul`）。

> **「所以它比 BF16 还慢」只在大 M 成立——但那正是 D3 当初的口径，所以 D3 没错。**
> batch 扫描（§0'''.65）实测 `w4a8-native` vs BF16：
> **2.43×（M=1）→ 1.40×（16）→ 0.80×（64）→ 0.76×（128）**。
>
> **机制**：M=1 decode 是**带宽受限**的，weight-only int4 照样只从 HBM 读
> int4——反量化在片上。它放弃的是**算力**收益，而 M=1 根本不缺算力。
> M 一大，cube 成为瓶颈，它就得按 bf16 的算力算，还要多付反量化，于是穿过 BF16。
>
> **我们相对它：1.08× → 1.77× → 2.72× → 2.27×。**
> **教训**：我 2026-08-14 先拿 M=1 一个点「推翻」了 D3，又被扫描推翻回来。
> **单工作点外推，在这个项目里已经错了两次**（另一次是 §0''.5 拿设备时间推
> 墙上时间）。

---

## 7. 整层 decode 实测

> **§7.1–7.3 的 eager 口径数据已作废**，结论被 §8 的图捕获口径推翻。
> 保留原文是为了让「错在哪」可追溯——**引用数字请一律用 §8。**

### 7.1 数据（µs / 层，QwQ-32B TP=1，kv_len=1024，随机权重）— **已作废，见 §8**

`results/e2e_decode_arms_cold.csv`。`ours-w4a8` 只在 gate_up/down 上用 kernel，
qkv/o 走 BF16。`ours-prequant` = 量化结果预先算好、只计时 GEMM，**是融合的上界**。

| batch | BF16 | vLLM W8A8 | nitro 融合 W8A8 | ours-w4a8 | ours-prequant | 融合上界 | pq/bf16 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1185.4 | 1466.3 | ✗ | 1410.7 | 1150.6 | **1.23×** | 1.03× |
| 16 | 1738.0 | 1226.7 | ✗ | 1130.5 | 1047.0 | **1.08×** | 1.66× |
| 64 | 3319.6 | 2730.9 | ✗ | 2566.7 | 2582.7 | **0.99×** | 1.29× |
| 128 | 5215.0 | 4581.1 | 6803.6 | 4653.0 | 4597.2 | **1.01×** | 1.13× |

### 7.2 被推翻的判断（都是我先说错、再被实验纠正的）

**① 「整层收益低是因为没融合」——错。** ~~（本条本身也是错的，见 §8.2）~~
`ours-prequant` 把量化变成免费，天花板只有 batch=1 的 1.23×，batch≥64 是 0。
真实分母问题是**非投影部分占整层 49–75%**（attention + 2×RMSNorm + RoPE + residual），
量化对那部分毫无帮助。P3 的 1.51× 是纯 GEMM 口径，整层稀释是数学必然。
**推论：融合 kernel 的优先级应该低于扩大覆盖面。**

**② 「拿 nitro 当 W8A8 融合基线」——不成立。**
nitro 的融合 W8A8 在 batch=128 是 6803.6 µs，**比 vLLM 不融合的 W8A8 慢 1.5×**，
也比 BF16 慢。原因是它做 dual quant（row + col.T），列量化走 AIC cube 转置——
那是反向传播才需要的，推理侧纯浪费，其开销超过融合收益。
另外它的 B1 **并没有减少下发次数**（rstd + norm⊕quant + gemm 三次，与 vLLM 相同），
融合省的是**访存**不是下发。它还要求 M 是 128 的倍数（`silu_rowq`），
batch<128 直接报错。**nitro 的价值是结构模板，不是可复用的推理基线。**

**③ 「qkv/o 输是因为 L2 让 BF16 占了便宜」——错。**
qkv(73 MB)/o(52 MB) 确实能装进 168 MB 的 L2，但**同一次前向里 gate_up(566 MB) +
down(283 MB) 已经把 L2 冲干净**，下一轮的 qkv 本来就是冷的。加了权重轮换后
数字纹丝不动（bf16 batch=1: 1155→1185）。轮换代码保留（口径更严谨），结论不变。

### 7.3 覆盖面：qkv/o 为什么现在回退 — **已作废，见 §8.3**

投影级实测（`results/e2e_decode_layer.csv`，含量化的完整调用路径）：

| | M=1 | M=16 | M=128 | M=256 |
|---|--:|--:|--:|--:|
| qkv vs BF16 | 0.23× | 0.44× | 0.58× | 1.01× |
| o vs BF16 | 0.17× | 0.34× | 0.35× | 0.51× |
| gate_up vs BF16 | 2.68× | 3.09× | 2.09× | 1.78× |
| down vs BF16 | 2.27× | 4.53× | 2.52× | 1.34× |

qkv M=1 的 GEMM 本身只要 31.9 µs（P3），整条路径 164 µs。**qkv/o 的 N、K 都小，
GEMM 太便宜，摊不掉固定开销**；gate_up/down 计算量大，收益完整兑现。
`vllm_midgroup_linear.should_use_midgroup` 里的 `QKV_O_MAX_M=256` 是照 P3 的
kernel 级数据写的，**按这份路径级数据应该收紧到「qkv/o 全档回退」**——尚未改，
因为覆盖面工作可能会改变结论。

---

## 8. 图捕获口径的整层实测（**手搭图口径；e2e 结论以 CKPT-E 为准**）

> **2026-08-14**：本节曾标注「当前唯一有效的 e2e 数字」，**已过期**。
> 它是**手搭 `torch.npu.NPUGraph`** 的口径，不是引擎口径。
> **引擎数字看 CKPT-E（§0''''）。** 本节仍然有用的地方：在
> `FULL_DECODE_ONLY` 下它与引擎口径**一致**（376.6 vs 385.8 µs/层，
> 1.33× vs 1.32×），所以可以当引擎的快速预测用——**但只在 FULL 模式下**。

用户 2026-08-13 提出「w4a8 在 e2e 上优势似乎很少」。查下来**优势一直在，是 §7 的
测量口径把它吃掉了**：eager 口径同时踩了两个坑（D7、D8）。换成 vLLM 真实解码所用的
**ACL 图捕获 + 融合算子**口径后，数字翻倍。

### 8.1 数据（µs / 层，`npu_ops/python/bench_decode_graph.py`，`results/e2e_decode_graph.csv`）

层内算子换成 vllm-ascend 解码实际调用的那套：`npu_rms_norm` / `npu_swiglu` /
`npu_fused_infer_attention_score`；整层 `torch.npu.NPUGraph` 捕获后 replay 计时。
**两个 `-pq` 列是各自的融合上界**（激活在图外量化好，只计时 GEMM），W8A8 也给了
同样的待遇，所以两边可比。

| batch | nonproj | BF16 | W8A8 | W8A8-pq | w4a8-mlp | **w4a8-all4** | w4a8-pq | vs BF16 | vs W8A8 | 融合上界 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 78.7 | 953.7 | 492.3 | 475.3 | 432.1 | **378.4** | 342.3 | **2.52×** | **1.30×** | 1.11× |
| 16 | 134.1 | 1179.3 | 600.1 | 567.6 | 554.4 | **500.6** | 467.4 | **2.36×** | 1.20× | 1.07× |
| 64 | 401.9 | 1655.9 | 994.7 | 927.8 | 916.6 | **877.2** | 828.6 | **1.89×** | 1.13× | 1.06× |
| 128 | 711.3 | 1970.1 | 1276.6 | 1204.3 | 1390.0 | 1361.1 | 1301.6 | 1.45× | **0.94×** | 1.05× |

**batch=1 这一行含 D9 的修复**（`F.pad` → 常驻缓冲 `copy_`）：修复前 492.9 µs / 1.03×，
修复后 378.4 µs / 1.30×。重复 3 次为 376.7 / 378.0 / 382.4 µs，vs W8A8 1.29–1.31×。
batch ≥16 不经过补齐分支，数字与修复前一致。

对照 §7.1 的同一件事：**BF16 口径从 1.03–1.66× 变成 1.45–2.36×**；
W8A8 口径从 0.98–1.08× 变成 0.95–1.21×。**结论：优势少不是方案的问题，是量的问题。**

### 8.2 §7.2 ① 是错的：非投影不是 49–75%，融合也不是低优先级

图捕获下 `nonproj`（四个投影换成常量）只有 batch=1 的 **72 µs**，占 BF16 整层 7%、
占 w4a8 整层 15%——不是 49–75%。eager 下那 854 µs 里**几乎没有设备时间**，
是 python 下发（D7）。

融合上界同样翻转：batch=1 从 1.23× 变成 1.43×（492.9 → 344.4 µs）。
那 150 µs 里**大头不是量化 kernel，是 M=1→16 的 `pad`**——拆开来看是
`pad` 12.3 µs × 4 vs `quant_a` 4.8 µs × 4（§8.4）。**`pad` 已按 D9 去掉**，
融合上界随之回落到 **1.11×**（378.4 → 342.3），只剩 21 µs 的量化 kernel。
batch ≥16 更少，只值 1.05–1.07×。**所以融合不是第一优先级，GEMM 流水才是（§8.5）。**

### 8.3 §7.3 也是错的：qkv/o 应该开，不是关

图捕获下 `w4a8-all4` 在 batch ≤64 **一律不慢于** `w4a8-mlp`（492.9/501.9/877.7 vs
496.2/553.8/919.1），batch=16 差 9%。§7.3 那张「qkv 0.23×、o 0.17×」的表量的是
**每次调用的 python 下发成本**，不是 GEMM。`QKV_O_MAX_M` 不该收紧到全档回退；
按 §8.1，**batch ≤64 全开、batch ≥128 回退**才是数据支持的分派。

### 8.4 逐算子成本表（batch=1，`bench_op_costs.py`，`results/op_costs_m1.csv`）

**测法**：单算子图 replay 有 ~65 µs 地板，量不了解码尺度的算子。把同一算子在**一个图里
重复 R=20 次**再减地板，`(T_R − T_0)/R` 就能量到远低于地板的量级。
`graph_wall(lambda: None)` = 64.3 µs 就是那个地板本身。

| proj | w4 权重 MB | w8 权重 MB | `F.pad` | `copy_` | quant_a | dyn_quant | GEMM4 | GEMM8 | **w4 合计** | **w8 合计** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| qkv | 18.6 | 36.7 | 12.1 | 0.6 | 4.8 | 0.4 | 25.8 | 27.8 | 33.4 | 29.4 |
| o | 13.3 | 26.2 | 12.3 | 0.4 | 4.8 | 0.5 | 21.9 | 19.5 | 29.4 | 22.1 |
| gate_up | 143.8 | 283.1 | 12.3 | 0.3 | 4.9 | 0.4 | 139.4 | 225.9 | **147.6** | **228.8** |
| down | 71.9 | 141.6 | 12.5 | 0.2 | 6.8 | 3.8 | 73.2 | 104.8 | **82.9** | **110.7** |
| **合计** | 247.6 | 487.6 | | | | | 260.3 | 378.0 | **293.3** | **391.1** |

（`F.pad` 一列是**修复前**的成本，现在走 `copy_`；D9。）

**qkv/o 输、gate_up/down 赢**：qkv/o 的权重才 13–19 MB，2× 的字节优势换不回
每次调用的固定开销；gate_up/down 分别赢 1.55× / 1.34×。但 §8.1 显示
**全开仍然最快**（378 vs 只做 MLP 的 432）——因为 qkv/o 只输 4–7 µs，
而把它们留在 BF16 要多付 54 µs。

### 8.5 GEMM 还差多少，以及卡在哪（profiler 实证）

单算子极限尺寸（N=128,K=1024）下：我们的 GEMM **4.0 µs**、`npu_quant_matmul` **5.2 µs**、
`quant_a` 4.5 µs。**所以下发地板不是我们的问题**（我们比 W8A8 还低），
差距全在 kernel 内部。

`torch_npu.profiler` + `AiCMetrics.PipeUtilization`（M=16 padded，10 次取中位数）：

| kernel | proj | µs | 有效 GB/s | aic_mte2% | aic_mac% | aic_mte1% | **aic_scalar%** | aiv_vec% |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| midgroup_w4a8 | qkv | 26.9 | 693 | 45.5 | 13.7 | 25.4 | **81.2** | 21.3 |
| midgroup_w4a8 | o | 21.9 | 608 | 41.8 | 12.0 | 22.2 | **80.8** | 18.8 |
| midgroup_w4a8 | gate_up | 136.2 | 1056 | 76.0 | 19.2 | 33.9 | **84.7** | 29.0 |
| midgroup_w4a8 | down | 73.1 | 984 | 44.0 | 17.3 | 34.8 | **70.1** | 25.9 |
| QuantMatmulV4 | qkv | 38.1 | 1320 | 77.3 | 11.1 | 36.5 | 83.7 | — |
| QuantMatmulV4 | o | 33.1 | 1344 | 69.8 | 9.1 | 24.6 | 53.2 | — |
| QuantMatmulV4 | gate_up | 232.5 | 1253 | **96.0** | 12.7 | 34.1 | 43.6 | — |
| QuantMatmulV4 | down | 119.5 | 1351 | **92.2** | 12.6 | 36.5 | 34.2 | — |

> **⚠ 本节的诊断已被 `P5_KERNEL_OPT.md` §3' 推翻。** 消融显示把 AIV、Mmad、
> Fixpipe 全删掉时间不变——是纯 GM→L1 搬运受限，不是握手。且 −62 µs 的估计
> 受 L2 命中污染，冷口径下实际只有 ~25 µs。**下面的推理保留以便追溯。**

**结论：W8A8 是干净的访存瓶颈（mte2 92–96%），我们不是。**
我们的 MTE2 只有 42–76%，MAC 只有 12–19%，**没有任何一条流水打满**——
是**依赖/控制瓶颈**，不是吞吐瓶颈。同时 `aic_scalar` 70–85%，在 `down`
（27 个 group，group 循环次数最多）上 mte2 只有 44%，而同形状 W8A8 是 92%。
指向 **mid-group 的分组冲刷握手（2 个 workspace slot + 两对 flag）挡住了 cube**：
每个 group 都要和 AIV 同步一次，group 越多越亏。

**剩余可回收的量（batch=1，共 293 µs 的投影时间）：**

| 项 | 值多少 | 难度 |
|---|--:|---|
| ~~GEMM 打到 W8A8 的 mte2 水平（1250 GB/s）~~ **已被 P5 修正为 −25 µs** | ~~−62 µs~~ | ~~加深分组流水~~ **消融证明流水不是瓶颈** |
| `quant_a` 融进 `npu_rms_norm` / `npu_swiglu` | −21 µs | 新 kernel，P3 D7 早有此定位 |
| qkv/o 每核 tile 太少（56/40 个 tile 摊到 20 核）| 含在第一项里 | 需要更细的 N 切分或 K 切分 |

全吃到 = 投影 ~210 µs、整层 ~290 µs → **vs W8A8 ≈ 1.7×、vs BF16 ≈ 3.3×**。

batch=128 的 0.94× 是另一回事：那一档进 cube-bound，MSD 的双 MAD 把算力翻倍，
与 D2 的「按 M 档分派」一致，**不用修，回退即可**。

---

## 6. 产物

| 路径 | 内容 | 状态 |
|---|---|---|
| `vllm/npu_ops/` | torch 算子包（CMake + host 绑定 + kernel） | **已产出** |
| `npu_ops/build.sh` | 按指定解释器构建（`PYTHON=` / `BUILD_DIR=`）| 已产出 |
| `npu_ops/build_nitro_for_venv.sh` | 按 venv torch 重编 nitro，**不动其原树** | 已产出 |
| `npu_ops/python/w4a8_ops.py` | 算子加载 + 权重量化到 kernel 布局 | 已产出 |
| `npu_ops/python/vllm_midgroup_linear.py` | vLLM linear method + 按投影分派 | 已产出（未接引擎）|
| `npu_ops/python/bench_decode_arms.py` | 四 arm 整层基准 | 已产出，**口径作废（D7/D8）** |
| `npu_ops/python/bench_decode_graph.py` | **图捕获整层基准（当前口径）** | **已产出** |
| `npu_ops/python/bench_proj_graph.py` | 图捕获单投影 + 有效带宽 | **已产出** |
| `npu_ops/python/bench_op_costs.py` | **逐算子成本（R 次重复摊掉 replay 地板）** | **已产出** |
| `npu_ops/python/bench_host_vs_device.py` | **host/device 判据（D7 的证据）** | **已产出** |
| `int4_cube_lab/results/op_costs_m1.csv` | §8.4 数据 | **已产出** |
| `int4_cube_lab/results/e2e_decode_graph.csv` | §8.1 数据 | **已产出** |
| `int4_cube_lab/results/proj_graph.csv` | §8.4 数据 | **已产出** |
| `npu_ops/python/vllm_engine_patch.py` | **引擎接入（加载期替换 linear）** | **已产出，CKPT-B** |
| `npu_ops/python/bench_engine_decode.py` | **引擎口径基准 + profiler 开关** | **已产出** |
| `int4_cube_lab/results/engine_decode.csv` | §0'.1 数据 | **已产出** |
| `npu_ops/python/test_pad_regress.py` | **补齐搬进算子的逐位回归**（对改动前的 .so 比）| **已产出，42/42 PASS** |
| `int4_cube_lab/results/engine_decode_ckptc.csv` | §0''.4 数据 | **已产出** |
| `int4_cube_lab/results/engine_decode_full.csv` | §0'''.1 FULL 模式数据 | **已产出** |
| `int4_cube_lab/results/engine_decode_w8a8.csv` | §0'''.4 W8A8 对照（3 次重复）| **已产出** |
| `int4_cube_lab/results/engine_decode_5arm.csv` | §0'''.5 五路对照（含两个原生 arm）| **已产出** |
| `int4_cube_lab/results/engine_batch_sweep_5arm.csv` | §0'''.65 五路 × 四档 batch 扫描 | **已产出** |
| `int4_cube_lab/results/e2e_decode_graph_ckptc.csv` | §0''.3 无回归证据 | **已产出** |
| `int4_cube_lab/results/e2e_decode_arms_cold.csv` | §7.1 数据 | 已产出，作废 |
| `int4_cube_lab/results/e2e_decode_layer.csv` | 投影级数据 | 已产出，作废 |
| `reports/midgroup/P4_e2e.md` | **端到端对照报告（五路 × 八档）** | **已产出（CKPT-E）** |
