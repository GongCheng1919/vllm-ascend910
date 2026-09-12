# P10 — 迁移到 Qwen3.8-27B(与 vllm-ascend 基线升级)

**状态：Phase A + B 已交付，本期收工（2026-09-04）** · 上游 `P7_LOWDIM_KERNEL.md` D2
环境固定：`export ASCEND_RT_VISIBLE_DEVICES=1`（device 0 已挂死；多卡从 1 号起排）

---

## ★ 一句话给接手的人

**主管方要求换掉 QwQ-32B（太老）并把基线从 vllm-ascend 0.13.0 抬到最新。
换模型这件事无法绕开环境升级——但升级的代价在 CANN，不在我们的代码，
所以 P10 先做一件完全不需要 vLLM 的事：在现有 CANN 8.5 上量 Qwen3.8-27B 的
算子级收益。那批数字决定要不要花升级那笔钱。**

---

## 0. 主管方的三条意见（2026-09-04）

| # | 意见 | P10 的去向 |
|---|---|---|
| 1 | 对比的是 vllm-ascend 0.13.0，应该跟最新版的原生 W4A8/W8A8 比 | ✅ 接受。**最新是 v0.23.0**（2026-08-16），不是 0.22.0。并入 Phase B |
| 2 | Deck 没把 `load→Cube→L0C-GM-UB 中转→Vector→store` 这条流水当成贡献 | ✅ **已完成**（见 §2 第 1 项）|
| 3 | QwQ-32B 太老，换 Qwen3.8-27B 稠密模型（或 MoE）| ✅ 稠密接受，**MoE 明确不做**（见 D3）|

---

## 1. 当前状态

- **Deck 已改**：`reports/QwQ32B_W4A8_DECK.html` 45 → 46 页，新增
  **方法·四「CV 流水」**（第 20 页），「四个设计决策」→「五个设计决策」，
  挑战 B2/B3 两页加了前向指针。
- **Phase A 已开跑**：`int4_cube_lab/scripts/sweep_p10_qwen38.sh`，
  产物 `int4_cube_lab/results/p10_qwen38_shapes.csv`。**零 vLLM 依赖。**
- **Phase B / C 未开工**，且**是否开工由 Phase A 的结果决定**。

---

## 2. 已完成

### 1. Deck 的 CV 流水页（主管方意见 2）

新第 20 页把这条流水作为**第 ⑤ 个设计决策**陈述，四个设计点：

| 设计 | 为什么非这样不可 |
|---|---|
| stacked-A | MSD 的 h/l 两平面沿 M 叠成 `2·TILE_M` 共享同一个 B ⇒ **一条 Mmad、一个 L0C、一次 Fixpipe** |
| A 与 W 共用组边界 | 一次冲刷同时服务两侧 dequant scale；不对齐则同一块 L0C 要冲两次 |
| GM 中转 + 双 slot | **L0C 读不到 Vector 侧**，而分组累加必须在 fp32 做 ⇒ 部分和只能经 GM 交给 AIV；2 slot + 4 个 CrossCore flag 让 Cube 的下一组与 Vector 的当前组重叠 |
| 中转全程 int32 | 与 CPU 参考逐位一致，门禁才能用位比较；`BF16_FLUSH` 省一半字节但不逐位一致 ⇒ 不设默认 |

**同页强制带代价模型**（黄框）。理由见 D1。

### 2. Qwen3.8-27B 的架构与形状（已核对 HF 原始 config.json）

`Qwen3_5ForConditionalGeneration`，**多模态**（vision + text + 1 层 MTP draft head）。
text：hidden **5120**、intermediate **17408**、**64 层**，`full_attention_interval=4`
⇒ **16 层全注意力 + 48 层线性注意力（Gated DeltaNet）**；
全注意力 24 Q heads × head_dim **256**（`attn_output_gate`）/ 4 KV heads；
线性注意力 16 key heads × 128、48 value heads × 128；vocab **248320**；
vision 27 层 / hidden 1152 / intermediate 4304 / out_hidden 5120。

| proj | 层数 | 并行 | N | K | TP 切法 | `N%128` |
|---|--:|---|--:|--:|---|---|
| qkv | 16 | 列 | 14336 | 5120 | N/TP | ✅ |
| o | 16 | 行 | 5120 | **6144** | K/TP | ✅ |
| gdn_in_qkvz | 48 | 列 | 16384 | 5120 | N/TP | ✅ |
| **gdn_in_ba** | 48 | 列 | **96** | 5120 | — | ❌ **跳过** |
| gdn_out | 48 | 行 | 5120 | **6144** | K/TP | ✅ |
| gate_up | 64 | 列 | 34816 | 5120 | N/TP | ✅ |
| down | 64 | 行 | 5120 | **17408** | K/TP | ✅ |
| vision(27 层) | — | — | 4304 | 1152 | — | ❌ **跳过** |

**跳过的两处不是妥协，是正确选择**：`in_ba` 是递归衰减门，量化误差沿序列**累乘**
（softmax 注意力每步重读精确 KV 没有这个性质），它只有 0.49 M/层；vision 的
K=1152 本来就在 `K*` 以下，压了也不赚。⇒ **可量化线性权重覆盖 24.33 B / 27.31 B。**

### 3. 内存账（算出来的，非实测）

| | 权重 | 备注 |
|---|--:|---|
| BF16 | **50.9 GiB** | 单卡 60.96 GiB 塞得进但 KV 无处可放 |
| W8A8 | 28.2 GiB | |
| **我们的 W4A8** | **17.0 GiB** | **3.00×**，单卡剩 **44 GiB** |

- **KV 64 KiB/token**（QwQ 是 256 KiB，**4× 更省**）——只有 16 层全注意力持有 KV。
- **GDN 状态 144 MiB/序列，与长度无关**——这是并发的新约束，不是长度的约束。
- ⇒ 我们那条「长上下文的下一个瓶颈是 KV 带宽（拐点 ~37.5K）」的曲线会**大幅右移**，
  权重带宽（我们赢的地方）在更长的上下文里仍是瓶颈。**待实测，不是结论。**

### 4. 环境依赖面（已实测本机）

| | 现状 | vllm-ascend v0.23.0 要求 |
|---|---|---|
| vllm / vllm-ascend | 0.13.0 / 0.13.0 | **0.23.0** |
| torch / torch_npu | 2.8.0+cpu / 2.8.0.post2 | **2.10.0 / 2.10.0.post4** |
| triton-ascend | 3.2.0 | **3.2.2** |
| CANN | **8.5.0**（`/usr/local/Ascend/cann-8.5.0`，V100R001C25SPC001B232）| **Stable CANN 9.1.0** |
| driver / HDK | **25.5.1** / V100R001C23SPC006B220 | 安装页写 **HDK 26.0.RC1**（**不在**官方兼容矩阵里，见 D7）|

> ⚠ 2026-09-04 更正：早先记的「CANN 9.0.1 / torch_npu 2.10.0.post2 / triton 3.2.1」
> 是从搜索摘要抄错的。**以 vllm-ascend `community/versioning_policy` 的发布兼容矩阵为准**，
> v0.23.0 那一行是上表右列。

**我们对 vLLM 的接触面只有 3 个文件 7 个符号**：

| 符号 | 位置 | 风险 |
|---|---|---|
| `LinearBase` / `LinearMethodBase` / `UnquantizedLinearMethod` | `vllm.model_executor.layers.linear` | 低 |
| `register_quantization_config` / `QuantizationConfig` | `vllm...quantization` | 中（抽象方法签名会变）|
| `current_platform` | `vllm.platforms` | 低 |
| `AscendUnquantizedLinearMethod` | `vllm_ascend.ops.linear` | **高**（已有 try/except 回落上游）|
| `AscendW4A8DynamicLinearMethod.apply` monkeypatch | `vllm_ascend.quantization.w4a8_dynamic` | **高**；仅「与厂商 W4A8 同路对比」才需要 |
| `llm.llm_engine.engine_core.engine_core...model` | 私有路径 | **高**（V1 引擎内部）|
| `compilation_config={"cudagraph_mode": FULL_DECODE_ONLY}` | | 中（已知会静默降级）|

⇒ 改动量级**几十到一两百行**。而 `int4_cube_lab/` 与 `fakequant_lab/`
**对 vLLM 零 import**（已 grep 核实）。

---

## 3. 下一步（按「挡住多少结论」排序）

| # | 事项 | 它挡住了什么 |
|---|---|---|
| 1 | **预期跳过清单** | W4A8 的 decode / prefill 吞吐——**挡住结论最多的一条** |
| 2 | **64 层单卡** | 把 3.0× 从推算变成实测；对外最有说服力的单一数字 |
| 3 | **四条 arm 对照矩阵** | bf16 / w8a8-native / w8a8(我们) / w4a8(我们)。⚠ A2 上**没有官方 Qwen3.8 W4A8 checkpoint**，要不要用 ModelSlim 自转一份当分母**需要拍板** |
| 4 | **拆掉私有属性链**（D8）| 迁移里最脆的一环，已被一个默认值变化打断过一次 |
| 5 | **精度** | 真实权重（51.7 GiB 未下载）+ **GDN 递归路径的敏感性分析**；P2 仍是挂起的 go/no-go |
| 6 | 端到端相对 BF16 的加速比 | **48/64 层的 GDN 向量工作我们不加速**，QwQ 的 2.0–2.2× 一定被压低，**压多少只能测**（D4）|

---

## 3'. 原三段式排期（已完成 A 与 B，留档）

### Phase A — 算子级收益（**进行中，零升级**）

`sweep_p10_qwen38.sh`：Qwen3.8 六个形状 + QwQ 四个形状（**同一 session 的对照**，
因为本机时钟跑间漂 ~14%），`M ∈ {1,2,4,8,16,64}`、`TP ∈ {1,2,4}`，
mid-group W4A8 vs 我们的 per-channel W8A8，`--cold --profile --nosync`、
median-of-3、warmup 10 / repeat 50。

- **M 档特意从 1 起并覆盖 2/4**：Qwen3.8 自带 1 层 MTP draft head，
  batch=1 的解码是**验证 M=2..4**，不是 M=1。
- 无 BF16 arm：lab 侧没有 bf16 GEMM，BF16 对照只能在引擎级做（Phase C）。
- **顺带把 P7 欠的重复性门禁做在这批新形状上**（新数据，不用补测旧的）。

**判据**：TP=1/2 下六个形状的 `mg_vs_w8` 若普遍 < 1，Phase B 值得买；
若 M=2..4 已经打平，则先回 P7 打固定项，不要动环境。

### Phase B — 环境升级（**由 A 决定是否开工**）

**B0（先做，不需要任何人）**：容器内把 CANN 9.1.0 装到 `$HOME/Ascend`，
`source` 它的 `set_env.sh`，重编 kernel + 跑门禁。
这一步是**唯一能判定「宿主机 driver 25.5.1 到底服不服 CANN 9.1.0」的实验**，
而且不动现有环境（这个容器就此保持为 P0–P8 的冻结复现环境）。
- **过** ⇒ 驱动根本不用碰，Phase B 只剩 `torch_npu 2.10.0.post4` + `vllm-ascend 0.23.0`
  的 pip 层，全在我们手里。
- **不过** ⇒ 拿到的是一条**具体的运行时报错**，那才是去找宿主机管理员时该带的东西
  （能说清是 driver 还是 firmware，比「文档写要 HDK 26.0.RC1」有说服力得多）。

**B1（只有 B0 不过才需要，且要找人）**：HDK 升级在宿主机上，要重启，
**影响同机所有容器** ⇒ 排期/权限问题，不是技术问题。届时更省事的做法是让管理员
直接起一个官方 `vllm-ascend:v0.23.0`（A2 tag）容器、把 repo bind-mount 进去
——但**那绕不开同一个宿主 driver**，所以顺序仍然是先 B0。

**B2**：验三件事 ① kernel 能编、门禁能过（**这是 CANN 的账不是 vLLM 的**）
② 转换计数 + `apply_model` 逐 rank 查类名两道证据仍能拿到 ③ `FULL_DECODE_ONLY` 仍在。

### Phase C — 引擎级矩阵

四条 arm：`bf16` / `w8a8-native` / `w8a8(我们)` / `w4a8(我们)`。
⚠ **缺 `w4a8-native`**：A2 上没有官方 Qwen3.8-27B W4A8 checkpoint
（只有 BF16 / W8A8 / W8A8-MXFP8(950) / W8A8-310p）。
要不要用 ModelSlim 自转一份当分母，**需要拍板**。

---

## 4. 产物

```text
int4_cube_lab/scripts/sweep_p10_qwen38.sh     Phase A 扫描脚本(含 flock 互斥)
int4_cube_lab/results/p10_qwen38_shapes.csv   Phase A 数字,180 行,零 NA
int4_cube_lab/results/p10_qwen38.log          运行日志
reports/QwQ32B_W4A8_DECK.html                46 页,新增方法·四
```

## Phase A 结论（2026-09-04，180 行全通，零 NA）

### 结论 1：TP=1/2 全赢，解码档 1.33–1.50×，且 **M=1..16 完全平**

按层数加权的「一层解码全部 Linear 的总 GEMM 时间」，分母是**我们自己的
per-channel W8A8**（不是 BF16 —— lab 侧没有 bf16 GEMM，那条只能进 Phase C）：

| TP | M=1 | M=2 | M=4 | M=8 | M=16 | M=64 |
|---|--:|--:|--:|--:|--:|--:|
| **1** | **1.49×** | **1.50×** | **1.50×** | **1.49×** | **1.50×** | 1.27× |
| 2 | 1.35× | 1.34× | 1.34× | 1.34× | 1.33× | 1.16× |
| 4 | 1.18× | 1.19× | 1.20× | 1.19× | 1.22× | 1.08× |

逐层比值（`mg_vs_w8`，<1 = 我们赢），TP=1：
`gate_up` 0.628–0.636 · `down` 0.651–0.659 · `gdn_in_qkvz` 0.687–0.695 ·
`qkv` 0.701–0.707 · `o` 0.775–0.795 · `gdn_out` 0.770–0.806。
TP=2 全赢（0.685–0.892）。**TP=4 只有 `o` / `gdn_out` 输**（K=1536 → 1.04 / 1.09），
其余仍赢 0.75–0.89。

**M=1→16 那条平线（TP=1 是 0.665–0.670）直接回答了 MTP 那个问题：
draft head 把解码的 M 推到 2–4 不花我们一分钱。**

### 结论 2：`K*` 成本模型在一个全新模型上独立复现

P7 D2 是在 QwQ 的 `o_proj`（K∈{640,1280,2560,5120}）上拟的。这次换模型、换 K
集合（Qwen3.8 的 `o`/`gdn_out`，N=5120 固定，K∈{6144,3072,1536}）重拟：

| M | 斜率比 | 固定差 | K\* |
|--:|--:|--:|--:|
| 1 | 0.58 | +3.19 µs | 1723 |
| 2 | 0.61 | +2.05 µs | 1285 |
| 4 | 0.53 | +3.72 µs | 1830 |
| 8 | 0.57 | +3.24 µs | 1809 |
| 16 | 0.61 | +2.15 µs | 1346 |
| 64 | 0.62 | +5.67 µs | 3050 |

斜率比 **0.53–0.62**（理论 0.5；P7 在 QwQ 上 M=16 是 0.60），`K*` 落在
**1300–1830**（P7 在 QwQ 上 M=16 是 1665）。⇒ **`o/gdn_out @ TP=4` 的 K=1536
正好骑在 `K*` 上，所以它就是 1.00–1.09，不多不少。**

**这比任何单点加速都值钱**：同一个成本模型、不同模型、不同 K 集合，仍然预测得住。
报告里应该以这个身份出现，而不是以「又一个加速数字」的身份。

### 结论 3（更正）：Qwen3.8 的算子级收益**略低于** QwQ，不是更高

同一个 session 的对照（这就是为什么脚本里带 QwQ 行）：

| TP | Qwen3.8 | QwQ-32B | 相对差 |
|---|--:|--:|--:|
| 1 | 1.49–1.50× | 1.55–1.56× | **−3.0 ~ −4.0%** |
| 2 | 1.33–1.35× | 1.37–1.39× | −2.1 ~ −3.6% |
| 4 | 1.18–1.22× | 1.21–1.23× | −0.5 ~ −2.4% |

原因是 **FFN 变瘦**：QwQ 的 `down` K=27648（0.627）、`gate_up` N=55296（0.603），
Qwen3.8 只有 17408 / 34816；`o_proj` 从 5120 涨到 6144 那点便宜补不回来。

⚠ **单点都在 ±2.6% 噪声地板附近**，但 18 个点里 17 个同号 ⇒ **方向可信、
量级 ~3%**。措辞应为「**打平或略低**」，不得写成「明显更差」，也**不得像本期
开工时那样写成「Qwen3.8 的形状更适合我们」——那句话在 GEMM 层面是错的**。

**真正的优势不在算子比值，在部署形态**：17.0 GiB 装得进一张卡 ⇒ **TP=1 可用**，
而 TP=1 是 1.50×、TP=4 只有 1.19×。**这是部署论证，不是 kernel 论证**，报告里
必须这么写，否则会被问穿。

### 判据

**Phase A 通过 ⇒ 建议开 Phase B。** 三条限定不得省：
① 这是 **GEMM-only 的天花板**，D4 仍然成立（48/64 层 GDN 的向量工作不加速，
引擎级比值一定更低，低多少只能测）；② **没有 BF16 对照**；
③ **单次扫描 median-of-3**，P7 欠的重复性门禁仍然欠着——M=1..16 那条平线
是稳定性的旁证，不是门禁。

---

## Phase B0 结论（2026-09-04）：**宿主机 driver 服 CANN 9.1.0，不用升驱动**

容器内把 CANN 9.1.0 装进 `$HOME/Ascend`（toolkit 3.5 G + 910b-ops 合计 9.0 G，
nnal 1.2 G），全程 **没碰 `/usr/local/Ascend`、没碰 `.venv`**。四条证据：

| # | 证据 | 结果 |
|---|---|---|
| 1 | **声明兼容** | CANN 9.1.0 自报 `compatible_version=[C15],[C21],**[C23]**,[C10]`；本机 driver `Innerversion=V100R001C23SPC006B220` ⇒ **C23 在列**。安装页那句 HDK 26.0.RC1 是「验证过的组合」，不是最低要求 |
| 2 | **编译零改动** | 55/55 目标干净全量编过（`MAKE_RC=0`，零 `error:`）。`.inc` 一行没动 ⇒ `MmadParams` / `FixpipeParamsV220` / CrossCore flag 在 8.5→9.1 之间无破坏性变更 |
| 3 | **正确性** | `check_ragged.sh` **67 PASS / 0 FAIL / ALL PASS**（m16+m64+m128+sym+其它 GK+量化器逐字节）|
| 4 | **性能不退化** | 三个 Qwen3.8 形状 **+0.4% / +1.3% / +2.9%**，全在 lab 噪声地板 ±2.6% 内 ⇒ **Phase A 那 180 行在新 CANN 下依然成立** |

| M | N | K | CANN 8.5 | CANN 9.1 | 差 |
|--:|--:|--:|--:|--:|--:|
| 16 | 5120 | 6144 (`o`) | 29.21 µs | 30.05 µs | +2.9% |
| 16 | 34816 | 5120 (`gate_up`) | 99.59 µs | 100.03 µs | +0.4% |
| 1 | 5120 | 17408 (`down`) | 53.49 µs | 54.17 µs | +1.3% |

⇒ **B1（找宿主机管理员升 HDK）取消。** 剩下的 Phase B 全是 pip 层、全在我们手里：
`torch 2.10.0` + `torch_npu 2.10.0.post4` + `vllm/vllm-ascend 0.23.0` +
`triton-ascend 3.2.2`，装进新的 `.venv-023`，现有 `.venv`（0.13.0）原样保留。
**风险从「驱动权限」变回「我们那 7 个 vLLM 符号在 0.23 里还在不在」——那是能自己改的活。**

### B0 踩的两个坑（都不是 CANN 的问题）

1. **`910b-ops` / `nnal` 不吃 `--full`**（只有 toolkit 吃）。传了 `--full` 它们
   **打印 help 就退出，exit code 仍是 0** ——又一个「没干活但看起来成功」的形状。
   正确是 `--install --quiet --install-path=`。
2. **ascendc 的 preprocess 是原地链接**（`ld.lld X.o -o X.o`）。先单编一个 target、
   再跑全量，第二遍读到已链过的 object ⇒ `unknown file type`。
   **同一个 build 目录不要先单编再全量**，或者先删干净。
   `build.sh` 已加 `JOBS` 旋钮（本机 96 核，`-j96` 让这类原地步骤更容易撞车）。

### 新增的环境约定

```bash
source /home/gongcheng/Ascend/cann-9.1.0/set_env.sh    # CANN 9.1.0 探针环境
export ASCEND_RT_VISIBLE_DEVICES=1
BUILD_DIR=build-cann91 bash int4_cube_lab/scripts/build.sh   # 不碰 CANN 8.5 的 build/
BUILD_DIR="$PWD/build-cann91" bash int4_cube_lab/scripts/check_ragged.sh
```

`build.sh` 与 `check_ragged.sh` 都新增了 `BUILD_DIR` 支持：**cmake cache 记住
toolkit 路径，一个 `build/` 伺候不了两个 CANN**，而 P0–P10 PhaseA 的全部数字都是
在 CANN 8.5 的 `build/` 里测的。

## Phase B（pip 层）结论（2026-09-04）：环境建成，**8 个符号里 7 个原地不动**

`.venv-023` 已建成并全部 import 通过。现有 `.venv`（vllm 0.13.0）**原样保留**。

| 包 | 版本 |
|---|---|
| vllm / vllm_ascend | **0.23.0 / 0.23.0** |
| torch / torch_npu | **2.10.0+cpu / 2.10.0.post4** |
| torchvision / torchaudio | 0.25.0 / 2.10.0 |
| triton_ascend / transformers / numpy | 3.2.2 / **5.5.4** / **1.26.4** |

### ★ 对主管方意见 1 的直接回答：**最新版的原生 W4A8 在 A2 上仍然是 weight-only**

`vllm_ascend/quantization/methods/w4a8.py:178` 的 `apply` 里写着
`# NOTE: activation \`x\` is not quantized`，调的是 `npu_weight_quant_batchmatmul`
——**和 0.13.0 完全同一性质**。⇒ 把基线抬到 0.23.0 之后，
**`w4a8-native` 这条 arm 的性质没有改变，我们此前的对照结论不因换版本而失效。**
（0.19–0.23 那些 W4A8 投入落在 `w4a8_mxfp4.py` = Ascend 950 的 MXFP4 路径上，
不在 A2。另有 `w4a4_flatquant` / `w4a4_mxfp4` / `w4a16` 等新方法可作为新对照组候选。）

### 符号审计（P10 §2 第 4 项那张表的实测结果）

| 符号 | 0.23.0 |
|---|---|
| `LinearBase` / `LinearMethodBase` / `UnquantizedLinearMethod` | ✅ 原地 |
| `register_quantization_config` / `QuantizationConfig` | ✅ 原地 |
| `vllm.platforms.current_platform` | ✅ → **`NPUPlatform`**（插件自动激活）|
| `vllm_ascend.ops.linear.AscendUnquantizedLinearMethod` | ✅ 原地，**`apply(self, layer, x, bias=None)` 签名不变** ⇒ **我们的主路径钩子零改动** |
| `vllm_ascend.quantization.w4a8_dynamic.AscendW4A8DynamicLinearMethod` | ❌ **模块搬家** → `vllm_ascend.quantization.methods.w4a8`。**类名与 `apply(self, layer, x, bias=None, tp_rank=None)` 签名完全一致**，只需改 import 路径；基类换成新的 `AscendLinearScheme` |
| `CUDAGraphMode` | ✅ `FULL / FULL_AND_PIECEWISE / FULL_DECODE_ONLY / NONE / PIECEWISE` 全在 |

**还没验的一条**：私有路径
`llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model`
——要起真引擎才能验，是 Phase B 的下一步。

### 两条新情报

1. **Ascend 现在默认禁用 breakable(PIECEWISE) cudagraph**：启动日志
   `Breakable cudagraph is force disabled on Ascend because DeepSeek V4 PIECEWISE
   cudagraph is not supported yet`。P4/P6 测到的「默认 PIECEWISE 白亏 2 倍」在
   0.23.0 上**可能已不成立**，重测时要重新确认默认值。
2. `ERROR ... Failed to import Triton kernels ... cannot import name
   'constexpr_function' from 'triton.runtime.jit'` —— `triton 3.5.0`（由
   `triton-ascend 3.2.2` 拖入）与 vLLM 期望的版本不一致。目前**不影响任何 import**，
   记录待观察。

### Phase B 踩的坑：**pip 会在 Ascend 机器上装 CUDA，且和有没有卡无关**

`pip install vllm==0.23.0` 会拉进整套 CUDA 13（`nvidia-cudnn-cu13` 单个 433 MB）。
原因**不是它检测到 GPU**——pip 从不看硬件，只看依赖元数据 + 「OS + CPU 架构」标记：

- `torch==2.10.0` 的 `nvidia-*` 依赖全部带 `platform_machine == "x86_64"` ⇒ aarch64 **一条不触发**；
- `torch==2.11.0` 把标记**放宽到了 aarch64**（NVIDIA GB200/GH200 sbsa 是 ARM），于是我们这台 aarch64 的 910B4 被误伤；
- 而拽进 `torch 2.11.0` 的是 **`vllm==0.23.0` 的精确 pin**，不是 vllm-ascend（后者要 `torch==2.10.0`）。**两者在 torch 上直接冲突**，官方那条「先 vllm 再 vllm-ascend」是靠第二步降级 torch 收场的。

⇒ 装法改为：**先装 torch 2.10 栈 → `vllm --no-deps` → `vllm-ascend` → 从 vllm 的
METADATA 补齐其余依赖**（脚本见 `scratchpad/install023.sh`）。
仍有 **21 个 CUDA 包**混进来，来源是 vLLM 两个**名字里不带 nvidia/cuda 的**依赖：
`flashinfer-python==0.6.12` 与 `humming-kernels[cu13]==0.1.4`。在 Ascend 上是死重量
（venv 7 GB），但**未验证能否安全卸载**，暂留。

两条配置结论：
- **PyPI 直连本机会 `ReadTimeout(files.pythonhosted.org)`**，用国内镜像（**镜像不是代理**，`*_proxy` 全程不设）；已写进 `.venv-023/pip.conf`：清华做 index、华为 Ascend 源做 extra-index。
- **numpy 定在 1.26.4**：`triton_ascend` 硬 pin `==1.26.4`；`ml-dtypes` / `opencv-python-headless` 声明的 `numpy>=2` 是保守打包，实测在 1.26.4 下**均能正常 import**。

### Phase B 收口：引擎管道实测（`npu_ops/python/probe_engine_paths_023.py`）

三条「import 验不出来、必须起真引擎」的路径，**一次全挂、一个根因、两个开关修好**。

**第一次全挂**（QwQ-32B / L=2 / TP=1 / bf16 / dummy）：

```
FAIL v0  AttributeError: 'LLMEngine' object has no attribute 'model_executor'
FAIL v1  AttributeError: 'SyncMPClient' object has no attribute 'engine_core'
FAIL collective_rpc  TypeError: Object of type <class 'function'> is not serializable
FAIL apply_model     （同上）
```

**根因只有一个：0.23.0 默认把 engine core 放进了独立进程**（`SyncMPClient` =
ZMQ + 后台进程），而 0.13.0 给 `LLM()` 的是 in-process 的 `InprocClient`
——我们那条私有链的 `engine_core.engine_core` 只在 `InprocClient` 上存在。
跨进程之后，传 callable 的 RPC 又撞上 0.23.0 收紧的序列化策略。

**加两个环境变量后全绿：**

| 开关 | 作用 |
|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | 强制 in-process engine core ⇒ 私有链恢复 |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | 允许 pickle 传 callable ⇒ 两个 RPC 恢复 |

| 项 | 结果 |
|---|---|
| `get_model()` | ✅ 走 **v0 路径**（`llm_engine.model_executor.driver_worker.model_runner.model`），返回 **`ACLGraphWrapper`** |
| `collective_rpc` | ✅ `['WorkerWrapperBase']` |
| `apply_model` 普查 | ✅ `[{'AscendUnquantizedLinearMethod': 8}]` ⇒ **默认装的正是我们主钩子要打的那个方法** |
| `LinearBase` 遍历 | ✅ 8 个（2 层 × 4 投影），形状 `qkv(7168,5120)` `o(5120,5120)` `gate_up(55296,5120)` `down(5120,27648)`，与我们的形状表逐字对上 |

**⚠ 两条必须记住的后果：**

1. **`get_model()` 现在返回的是 `ACLGraphWrapper` 而不是裸模型。**
   `named_modules()` 能穿过去（所以转换钩子没事），但任何假设「返回值就是模型」
   的代码（例如 `model.model.layers`）会断。
2. **`VLLM_ENABLE_V1_MULTIPROCESSING=0` 是非默认配置，会改变引擎行为**
   （in-process vs 独立进程的 engine core）。它恢复的是 0.13.0 的形态，
   对**同口径对比**反而是对的——但**任何用它测出来的数字都必须把这一条写进配置行**。

### D8（2026-09-04）私有链应当拆掉，改走公开 API

`get_model()` 那条四层私有属性链**已经被一个默认值的变化打断过一次**，
是整套接入里最脆的一环。而 `apply_model` / `collective_rpc` 是公开 API，
覆盖我们全部三个用途（转换、逐 rank 计数、`quant_method` 普查），
**且它们本来就在 worker 进程里执行——多 rank 下转换本来就只能在那里做**（P6 D8）。

⇒ 拆掉私有链后，`VLLM_ENABLE_V1_MULTIPROCESSING=0` 也就不再需要，
只剩序列化开关（或者干脆注册一个正经的 RPC 方法名，连那个也不用）。
**生产路径（`vllm.general_plugins` 入口点在每个 worker 里自我武装）本来就不吃这条链**，
受影响的只有 `convert_model` 的 TP=1 便捷路径和 bench 的取证查询。

### Phase B 最后一环：`npu_ops` 对 torch_npu 2.10 / CANN 9.1.0 重编，**数值逐位一致**

```
PYTHON=$PWD/.venv-023/bin/python BUILD_DIR=build-023 MAX_JOBS=24 bash npu_ops/build.sh
  torch=.venv-023/.../torch (2.10.0+cpu)   torch_npu=2.10.0.post4   cxx11_abi=1
  Built: libvllm_w4a8_npu_ops.so           BUILD_RC=0   errors=0
```

`torch.ops.npu.midgroup_quant_a` / `midgroup_w4a8_gemm` **均已注册**
（⚠ `dir(torch.ops.npu)` 返回空列表是 torch 的 `_OpNamespace` 不枚举算子，
**不是没注册**，用 `hasattr` 判断）。

**跨 CANN 数值回归（同 seed、同代码、三个 Qwen3.8 形状）：**

| 形状 | CANN 9.1.0 / torch 2.10 | CANN 8.5.0 / torch 2.8 |
|---|---|---|
| `o_proj` M=16 N=5120 K=6144 | SNR 18.4643 dB · max\|err\| 43.0914 · sum 28402.335938 | **完全相同** |
| `gate_up` M=16 N=34816 K=5120 | SNR 18.5073 dB · max\|err\| 39.6597 · sum 100368.414062 | **完全相同** |
| `down` M=16 N=5120 K=17408 | SNR 18.5040 dB · max\|err\| 66.1318 · sum 65876.648438 | **完全相同** |

⇒ **CANN 8.5 → 9.1.0 的迁移在数值上是零变化**，配合 B0 的 67 项门禁，
算子层的迁移可以判定为**完成**。

（torch_npu 2.10 在本机会打一条 info：`The current CANN and HDK(driver) versions
require processing for 32 padding size, with memory allocation` —— 记录，暂未见影响。）

### D9（2026-09-04）`.so` 的选择必须显式，不能猜

`w4a8_ops._lib_path()` 原本在 `build-venv` / `build` 之间靠
`"/.venv/" in sys.executable` 和 `torch.__version__.startswith("2.8")` **猜**，
根本不认识 `build-023` ⇒ 在 `.venv-023` 里会去加载 **CANN 8.5 的旧 `.so`**。

**ABI 不匹配那种反而安全（当场炸）；危险的是选中一份能加载但过时的**——
它安静地跑起来，而我们评的是另一个 kernel。与「`.inc` 改动不触发重编」
（P6.5 D4）同一类失败。

改为显式映射 `_BUILD_BY_TORCH = {"2.10": "build-023", "2.8": "build-venv"}`，
外加 `W4A8_OPS_LIB` 强制指定；**找不到就抛 `FileNotFoundError` 并列出试过的路径，
不再静默回退。**

## 单卡 Qwen3.8-27B 实测（2026-09-04，本期收工点）

**口径**：`models/Qwen3.8-27B-L8`（L=8 变体）· TP=1 · `load_format=dummy` ·
`FULL_DECODE_ONLY` · `gpu_memory_utilization=0.85` · 单张 910B4（60.96 GiB）·
`VLLM_ENABLE_V1_MULTIPROCESSING=0`（**非默认**）。

### 1. 混合注意力架构在 910B4 上跑通了 —— 这是本期最大的门槛

BF16 arm 起引擎并生成 token，`AscendUnquantizedLinearMethod` 覆盖 **154 个 Linear**，
**77.3 tok/s**（batch=1，step 12939.1 µs）。
⇒ **vllm-ascend 0.23.0 在 910B4 上实现了 Gated DeltaNet**（48/64 层都是它：
conv1d + chunked delta rule + fp32 递归状态）。**这条不通后面全免谈，现在通了。**

### 2. 内存账（同一张卡，只换 arm）

| 项 | BF16 | 我们的 W4A8 | 变化 |
|---|--:|--:|--:|
| 权重 | 11.39 GiB | **6.72 GiB** | −41% |
| 峰值激活 / NPU 图 | 2.85 / 0.60 GiB | 2.85 / 0.61 GiB | — |
| 可用 KV cache | 37.49 GiB | **42.17 GiB** | **+4.68 GiB** |
| KV 容量 | 126,720 tok | **142,520 tok** | +12.5% |
| 最大并发（160 tok/请求）| 792× | **890.75×** | +12.5% |

**省下的权重一比一变成了 KV**（−4.67 / +4.68 GiB）。这就是方案在单卡上的价值主张：
**不是让每一步更快，是让一张卡装得下更多上下文和并发。**

⚠ **L=8 的 1.69× 不代表 64 层的 3.0×**：embedding(2.54 GiB)/lm_head/vision 这些
**与层数无关的固定项**在 L=8 时占权重大头，稀释了压缩比。64 层的 50.9 → 17.0 GiB
仍是**推算**，只是与本次实测自洽。

### 3. W4A8 的吞吐**没有数字**，闸门正确拦下

```
[bench] converted=115 skipped=39
[bench] ERROR: skipped 39 linears -- they stayed BF16, so this is a MIXED arm, not w4a8
```

**39 个全部是设计内的跳过，没有覆盖率漏洞**（本期给闸门补上了逐条列名的能力才能这么说）：

| 层 | 个数 | 形状 | 原因 |
|---|--:|--:|---|
| `visual.blocks.*.mlp.linear_fc1` | 27 | (4304, 1152) | N=4304 不是 128 倍数；且 K=1152 在 `K*` 之下，压了不赚 |
| `linear_attn.in_proj_ba` | 6 | (96, 5120) | N=96 不对齐；**且它是递归衰减门，误差沿序列累乘，本就该留 BF16** |
| `linear_attn.conv1d` | 6 | (10240, 1, 4) | 三维张量，**根本不是 GEMM**，被 `dim()!=2` 正确拒绝 |

⇒ 解锁吞吐数字需要给 bench 一份**「预期跳过清单」**：允许这三类，同时在结果里
显式声明「vision 塔与 GDN 门控保持 BF16」。**放行的门槛必须比拦截的门槛更高**，
所以本期不顺手放行。

### D10（2026-09-04）计数不是证据，层名才是

`skipped=39` 这个整数**分不清两件相反的事**：跳过的是设计内那一批，还是覆盖率真有洞。
⇒ 给 `patch_unquantized_linear` 加了 `skipped_names`（层名 + 形状），
bench 在拒绝出数时逐条打印。**这是 P6 D8「日志的缺席不是证据」的自然延伸：
一个计数也不是证据。**

### 本期新增/改动的文件

```text
reports/QWEN38_W4A8_DEPLOY.html          Qwen3.8 量化部署总结报告
models/Qwen3.8-27B/                      config + tokenizer（ModelScope；HF 本体被墙）
models/Qwen3.8-27B-L8/                   L=8 变体（num_hidden_layers 与 layer_types 必须同时截断）
npu_ops/build-023/libvllm_w4a8_npu_ops.so   对 torch_npu 2.10 / CANN 9.1.0 重编
npu_ops/python/probe_engine_paths_023.py    引擎管道三条私有/公开路径的探针
int4_cube_lab/build-cann91/              CANN 9.1.0 的 lab 构建（8.5 的 build/ 未动）
.venv-023/                               vllm/vllm_ascend 0.23.0 全栈（.venv 未动）
/home/gongcheng/Ascend/cann-9.1.0/       CANN 9.1.0（/usr/local/Ascend 未动）
```

## 5. 决策与坑

### D1（2026-09-04）CV 流水页必须与代价模型同页

主管方要求把这条流水讲成贡献。**但 P7 D2 已经把它指认为固定开销的最大嫌疑人**
（每量化组一趟 `L0C→GM→UB` + 两次握手），P5 的结论也是「按冲刷次数收费，
不按字节」。只讲创新不讲边界，读到第 33 页「一条曲线解释全部三处失败」时会
**自相矛盾**。⇒ 同页给出「K=1280 只有 2 组摊不开 / K=27648 有 27 组就摊开」
这条分摊关系。**带着收益域讲的贡献比不带的强**。

### D2（2026-09-04）换模型无法绕开升级，但升级不在关键路径上

Qwen3.8-27B 是 `model_type: qwen3_5`（config 写着 `transformers_version:
5.8.0.dev0`），在 vllm 0.13.0 / transformers 4.57.6 里**架构根本不存在**——
不是跑不快，是加载不了。vllm-ascend 首次支持它是 **v0.23.0**。

但代价的落点被误判了：**贵的是 CANN 8.5 → 9.0.1 这个主版本跳**（打在 AscendC 的
`MmadParams` / `FixpipeParamsV220` / CrossCore flag / bisheng-ccec 上），
不是 python 接入层。⇒ 三段式：**先用不依赖 vLLM 的 lab 拿到算子级答案，
再决定要不要买升级。** 回退路径：若 Phase A 显示 Qwen3.8 形状上不赢，
P10 就地关闭，回 P7 打固定项，模型不换。

### D3（2026-09-04，用户判断）MoE 明确不做

Qwen3.5-397B-A17B 那条线：我们的算子是稠密 GEMM，MoE decode 需要
grouped/batched matmul + 路由。理论上「每专家 M 很小、K 很大」正是收益域，
**但 vllm-ascend 从 v0.19.1rc1 起已把 dispatch+FFN+combine 融成单 kernel**，
拆开换成我们的 per-expert GEMM 会先丢掉融合收益。⇒ 那是**新一期 kernel 工作，
不是移植**。本期不碰。

### D4（2026-09-04，**已被 Phase A 部分证伪，见修订**）Qwen3.8 的形状对我们更有利

> **修订（同日，Phase A 出数后）**：D4 的「有利 ②`o_proj` K 涨到 6144」在
> 加权口径下**被 FFN 变瘦盖过**——同 session 实测 Qwen3.8 比 QwQ **低 2–4%**
> （§4 结论 3）。⇒ **「形状更有利」这句话作废**，保留的是「**部署形态更有利**」：
> 17.0 GiB ⇒ TP=1 可用，而 TP=1 是 1.50×、TP=4 只有 1.19×。
> 下文原文保留，作为「开工时的判断哪里错了」的记录。

有利：① 17.0 GiB 单卡装得下 ⇒ **可以 TP=1**，而 TP 是我们唯一塌掉的维度
（P6.5：TP=4 只剩 1.16×，PP=4 是 2.0–2.2×），且 vllm-ascend 官方对 A2
就推荐 TP=1/2；② `o_proj` 的 K 从 5120 涨到 **6144**；③ KV 4× 更省。

不利，**必须主动说出口**：**48/64 层是 GDN**——conv1d、chunked delta rule、
fp32 状态更新，全是向量/标量工作，**我们一点都不加速**。QwQ 上的 2.0–2.2×
在这里**一定会被压低**，压多少**不能外推，只能测**（P6.5 D1 就是被逐层外推坑过）。
另外多模态 prefill 图片 token 多 ⇒ M 变大，而我们 **prefill 只是打平（0.96–0.99×）**。

### D5（2026-09-04）跳过 `in_proj_ba` 和 vision 是设计，不是妥协

两处都撞 `N % 128 == 0`（`midgroup_w4a8_gemm.cpp:78`），但**即使 N 轴 ragged
做完了也不该压它们**：`in_ba` 是递归衰减门，误差沿序列累乘；vision 的 K=1152
在 `K*` 以下。⇒ **不要把它们当成 P7 第 3 项（N 轴 ragged）的动机。**

### D6（2026-09-04）两个 sweep 并发跑同一张卡，看起来像数据而不像故障

第一次启动用 `nohup ... &`，harness 报了「completed」但**子进程还活着**；
以为它死了又起了第二个。两个实例**争同一张卡**：双方计时都被抬高，
而且都 `tee -a` 进同一个 CSV，行是**交错**的（tp=2 的行夹在 tp=1 中间）。
⇒ **它不会报错，只会产出一份看起来正常的表。** 症状是行序乱、比值整体偏软。

已在 `sweep_p10_qwen38.sh` 顶部加**互斥保护**（发现同名进程直接 `exit 3`）。
教训与 `multirank-probes-blame-your-own-sync-first` 同类：
**先怀疑自己的运行方式，再怀疑数据。**

### D7（2026-09-04）我们在容器里：driver 动不了，但 CANN 动得了

实测：`/.dockerenv` 存在、pid1 是 `docker-init`、hostname `daf135ff3ab3`。
**`/usr/local/Ascend/driver` 是从宿主机 `/dev/sdb2` bind-mount 进来的 `ro` 挂载**
（`/etc/ascend_install.info` 同理），容器内**没有 docker CLI**。
⇒ **driver / firmware 升级在这里做不到**，必须宿主机 + 重启，影响同机所有容器。

但**工具链侧完全是我们的**：`/usr/local/Ascend/` 本身 root 所有（建不了
`cann-9.x` 兄弟目录），可 `/home/gongcheng` 可写、**剩 4.3 TB**，
而 CANN 支持任意 `--install-path`；容器**出网正常**（pypi 200）。
⇒ **CANN 9.1.0 装进 `$HOME/Ascend` 不需要任何人批准**，这也顺带自动满足了
「不要原地覆盖 CANN」那条要求。

**「官方要求」不是硬门，我们手上就有反例**：当前 driver 的
`compatible_version` 只列到 **C23**，而它此刻正在跑 **CANN 8.5.0 = C25**。
⇒ 这张表是保守的、向后看的。**所以 B0 那个实验必须先做，不能拿文档当结论。**

设备侧：容器映射了 8 张卡（`/dev/davinci2..9` + `davinci_manager` +
`devmm_svm` + `hisi_hdc`）。

---

# Phase C — 完整 64 层 Qwen3.8-27B 单卡交付矩阵（2026-09-05）

**口径**：`models/Qwen3.8-27B` 真实 64 层几何 · TP=1 · 单张 910B4（60.96 GiB）·
`load_format=dummy` · `dtype=bfloat16` · `FULL_DECODE_ONLY` ·
`gpu_memory_utilization=0.95` · `max_num_seqs=32` · plen=128 / out_len=128 ·
`VLLM_ENABLE_V1_MULTIPROCESSING=0`（非默认）· vllm/vllm-ascend 0.23.0 ·
torch_npu 2.10.0.post4 · CANN 9.1.0 · `ASCEND_RT_VISIBLE_DEVICES=1`。
产物 `int4_cube_lab/results/p10_qwen38_engine.csv`。

## ★ 一句话

**64 层跑通了，权重 51.08 → 16.83 GiB（3.03×），KV 从 2,788 涨到 27,404 token（9.8×）；
但吞吐上 W4A8 只在 batch≤4 赢 W8A8，batch≥16 就输回去 —— 这正是 `K*` 成本模型
和 MSD 的预言，不是新问题。**

## 1. 内存：这才是本方案在 Qwen3.8 上的价值主张

| arm | 权重 | 可用 KV | KV 容量 | 最大并发(272 tok/请求) |
|---|--:|--:|--:|--:|
| BF16 | **51.08 GiB** | 3.90 GiB | **2,788 tok** | **10.2×** |
| 官方 W8A8（合成描述） | 28.16 GiB | 26.83 GiB | 19,244 tok | 70.8× |
| 我们的 W8A8 | 28.14 GiB | 26.84 GiB | 19,244 tok | 70.8× |
| 官方 W4A8（合成描述，vision 留 BF16）| 18.20 GiB | — | 25,704 tok | 94.5× |
| **我们的 W4A8** | **16.83 GiB** | **38.16 GiB** | **27,404 tok** | **100.8×** |

- **3.03× vs BF16 / 1.67× vs W8A8**，与 P10 §2.3 的推算（50.9 → 17.0 GiB）几乎逐位吻合，
  **推算就此转为实测**。
- **省下的 34.25 GiB 权重一比一变成了 KV**（3.90 → 38.16 GiB）。
- **BF16 在单卡上不是"慢"，是装不下**：`max_num_seqs=256` 时引擎直接拒绝启动 ——
  `max_num_seqs (256) exceeds available Mamba cache blocks (41)`。
  **48/64 层是 GDN，每条在跑的序列要占一块 Mamba state**，所以权重占用直接
  设死了并发上限。BF16 只剩 41 块，W4A8 有 100.8×。
  ⇒ **这是 Qwen3.8 特有的新约束，QwQ 上不存在**（那里只有 KV）。

## 2. 吞吐（tok/s，同一张卡、同一 session、同配置）

| arm | b=1 | b=4 | b=16 | b=32 |
|---|--:|--:|--:|--:|
| BF16 | 18.7 | 69.9 | **50.6** ⚠ | **38.8** ⚠ |
| 官方 W8A8 | 30.2 | 113.1 | **366.0** | **568.9** |
| 我们的 W8A8 | 27.8 | 102.6 | 339.1 | 537.5 |
| **我们的 W4A8** | **33.4** | **122.7** | 354.1 | 497.3 |
| 官方 W4A8 ‡ | 29.8 | 102.7 | 265.2 | 273.5 |

‡ **不同配置，只能定性读**：这条 arm 必须关掉 `npugraph_ex` 才起得来（见 §5'），
而另外四条都开着；且它的 vision 塔留在 BF16。它唯一站得住的结论是
**厂商 W4A8 是 weight-only（反量化回 bf16 再做 matmul，activation 的 int8
根本到不了 cube），所以在大 batch 掉得比谁都快**：b=32 只有 273.5 tok/s，
比同版本的**厂商自己的 W8A8**（568.9）慢 **2.08×**，也比我们的 W4A8（497.3）慢 1.82×。
**P4 D3 在 QwQ 上的判断在 Qwen3.8 上原样复现，换到 0.23.0 也没变**
（`vllm_ascend/quantization/methods/w4a8.py` 里那句 `# NOTE: activation \`x\` is not quantized` 还在）。

⚠ **BF16 在 b≥16 的两格不是速度，是容量**：2,788 token 的 KV 装不下
16×256，调度器开始抢占重算，step 从 57 ms 跳到 316 ms。**不要把 7.0×/12.8×
当成 GEMM 加速比引用**，那是"BF16 单卡塞不下"的另一种说法。

**可以引用的比值**（b=1/4，KV 都没打满）：

| 对比 | b=1 | b=4 |
|---|--:|--:|
| 我们的 W4A8 vs BF16 | **1.79×** | **1.76×** |
| 我们的 W4A8 vs 我们的 W8A8 | **1.20×** | **1.20×** |
| 我们的 W4A8 vs 官方 W8A8 | **1.11×** | **1.08×** |
| 我们的 W8A8 vs 官方 W8A8 | 0.92× | 0.91× |

**b=16/32 上 W4A8 输给 W8A8**（0.97× / 0.87× 对官方，1.04× / 0.93× 对我们自己）。
**这是预期内的**：MSD 把 int8 A 拆成两个 int4 平面，计算量翻倍，
所以在计算受限区（大 M）W4A8 的理论优势归零（见 `w4a8-win-region-small-m-large-k`）。
**W4A8 的收益域是小 M + 大 K，而单卡 Qwen3.8 的部署形态正好在那里。**

## 3. 覆盖率取证：四条 arm 的层集合逐字相同

```
bf16               {'AscendUnquantizedLinearMethod': 462}
官方 W8A8          {'AscendLinearMethod': 339, 'AscendUnquantizedLinearMethod': 123}
我们的 W8A8        {'MidGroupLinearMethod': 339, 'AscendUnquantizedLinearMethod': 123}
我们的 W4A8        {'MidGroupLinearMethod': 339, 'AscendUnquantizedLinearMethod': 123}
```

**339/123 完全一致** ⇒ 比值隔离的是 GEMM，不是层集合、不是 checkpoint。

## 4. 解锁吞吐数字的那把闸门（P10 §3 第 1 项）

P10 收工时 W4A8 吞吐是空白，因为闸门拒绝出数：`skipped=39`（L=8）/ `123`（L=64）。
本期没有把闸门放松成一个计数，而是让调用方**声明**它预期跳过哪些：

- `vllm_engine_patch.EXPECTED_SKIPS`：每条 = (层名模式, 层名模式, 为什么)。
- `bench_engine_decode.py --expect-skip qwen38`：
  **每一个跳过都必须被某条声明认领，一个认不出来就仍然失败**，
  并把声明连同理由打进日志。多 rank 下 `skipped_names` 随 RPC 一起回来。

L=64 的实际分类（全部命中，零 UNEXPECTED）：

| 数量 | 模式 | 为什么它即使形状放开也该留 BF16 |
|--:|---|---|
| 27 | `visual.*mlp.linear_fc1` | K=1152 在 `K*`(1300–1830) 之下，压了不赚 |
| 48 | `linear_attn*in_proj_ba` | GDN 递归衰减门，误差沿序列累乘 |
| 48 | `linear_attn*conv1d` | 3 维张量，根本不是 GEMM |

⇒ 结果的正确读法是 **"W4A8 EXCEPT 这三类"**，日志每次都会把这句话打出来。

## 5. 官方 arm：Qwen3.8 没有厂商量化 checkpoint，我们合成了描述文件

**ModelScope 实测（2026-09-05）**：`Qwen3.8-27B` 只有
`Qwen/Qwen3.8-27B`(BF16) 和 `unsloth/Qwen3.8-27B-GGUF`；
`<org>/Qwen3.8-27B-W8A8` 形状的 id **全部 404**。
⇒ **P10 Phase C 那句"A2 上有官方 W8A8"是错的，实际两个 `-native` arm 都没有权重。**

但 vllm-ascend **是打算支持它的**：`packed_modules_model_mapping` 里
`model_type "qwen3_5"` 有条目（qkv_proj / gate_up_proj / in_proj_qkvz / in_proj_ba）。
缺的只是标定权重 —— 而**本矩阵所有 arm 都是 `load_format=dummy`，没有任何 arm
声称精度**。所以合成一份 `quant_model_description.json` 就够让 vLLM 分配 int8/int4
参数并派发厂商的 kernel：`npu_ops/python/gen_vendor_quant_desc.py`。

**必须带着的限定**：这不是"厂商出货的那一版"。两件事是我们的：
① **层集合**（刻意对齐我们的 339，比值才隔离 GEMM）；② **没有标定**（所有 arm 都没有）。
真实厂商版会在 ① 上不同 —— QwQ 上他们把所有 `down_proj` 留在 FLOAT，
那是标定决策，合成文件复现不了。⇒ arm 名写作 `w8a8-native(synth-desc)`。

### 5'. 官方 W4A8 在 Qwen3.8 上两次撞墙（都不是我们的代码）

1. **不吃 3 维输入**：vision 塔的 `attn.qkv` 传进 3-D x，
   `AscendW4A8DynamicLinearMethod.apply` 不做 flatten ⇒
   `AclNN_Parameter_Error(EZ1001): x's dim should be in range [2, 2]. actual is [3]`。
   （厂商自己的 `W8A8_DYNAMIC` 有对应的 squeeze，所以 W8A8 那条没事。）
   ⇒ 只能把整个 vision 塔留 BF16，权重 18.20 GiB 比我们的 16.83 GiB 大。
2. **进不了 FULL_DECODE_ONLY 图**：`'CompilerConfig' object has no attribute
   'experimental_config'`（torch_npu `npugraph_ex` ← vllm_ascend
   `compiler_interface.patched_get_compiled_gm`）。**bf16 与 W8A8_DYNAMIC 同配置编得过。**
   绕法是 `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'`，
   加上它这条 arm 就起来了 —— **但那是一个另外四条 arm 没有的配置差异**，
   所以它的数字只能定性读，不能进同一张速度表逐格比。

⇒ **两条加起来的结论**：在 0.23.0 + torch_npu 2.10 上，
**厂商 W4A8 路径在 Qwen3.8-27B 这个多模态 + GDN 架构上是走不通的**
（3-D 输入直接报错、默认图后端编不过）。这不是"慢"，是"跑不了"。
我们的路径两条都不撞：`MidGroupW4A8Linear` 一开始就 flatten，
kernel 也不经过 npugraph_ex 的那条特殊路径。

⚠ **同一个坑我们自己也踩了一半**：`W8A8Linear.__call__` 原本直接把 x 喂给
`npu_dynamic_quant`，3-D x 会让 pertoken_scale 变成 2-D，`aclnnQuantMatmulV5`
报 `EZ0013 ... must be 1D` —— **64 层 W8A8 arm 第一次直接跑挂**。
QwQ 的每个 Linear 都是 2-D，所以这个 bug 藏了整整六期。已按
`MidGroupW4A8Linear` 的做法改成先 flatten 再 reshape。

## D11（2026-09-05）跳过清单必须是"声明"，不能是"计数"，也不能是"放行"

`skipped=39` 分不清设计内跳过和覆盖率漏洞（P10 D10 已指出）。
本期给出的解法不是把门槛降到计数，而是**要求调用方逐条声明**，
并保持"一个认不出来就整轮失败"。**放行的门槛比拦截高**这条没有松动：
声明表里每一条的理由都必须是"即使形状约束解除也该留 BF16"，
而不是"它现在过不了形状检查"。

## D12（2026-09-05）0.23.0 默认打开 async scheduling —— 旧的 1.25× 已经吃掉了

启动日志每条 arm 都有 `Asynchronous scheduling is enabled`。
P6.5 D12 把它记成"默认关、开了值 1.23–1.29×"，**在 0.23.0 上不再成立**。
更糟的是 bench 的 CSV 记的是**我们传的 flag**，于是本期每一行都会写
`async_sched=0` 而引擎实际是开的。已改为读引擎解析后的
`vllm_config.scheduler_config.async_scheduling`。
**与 P6 D8 同类：不要把"我们请求了什么"当成"引擎做了什么"。**

## D13（2026-09-05）Qwen3.8 的单卡瓶颈是 Mamba state，不是 KV

BF16 在 `max_num_seqs=256` 下**根本起不来**：
`exceeds available Mamba cache blocks (41)`。48/64 层是 GDN，
**每条在跑的序列要一块与长度无关的 state**，所以权重占用直接换算成并发上限。
⇒ 报告里"省内存"的落点应该是**并发**（10.2× → 100.8×），
而不是只讲上下文长度。**这是 QwQ 时代没有的约束。**

## 本期新增/改动

```text
npu_ops/python/vllm_engine_patch.py       EXPECTED_SKIPS + classify_skips；W8A8Linear 支持 N 维输入
npu_ops/python/bench_engine_decode.py     --expect-skip / --max-num-seqs / --additional-config；
                                          skipped_names 随 RPC 回传；CSV 记引擎解析后的 async_sched
npu_ops/python/gen_vendor_quant_desc.py   合成 ModelSlim quant_model_description.json（新）
models/Qwen3.8-27B-L16/                   L=16 变体（num_hidden_layers 与 layer_types 同时截断）
models/Qwen3.8-27B-W8A8-Synth/            官方 W8A8_DYNAMIC arm（合成描述）
models/Qwen3.8-27B-W4A8-Synth/            官方 W4A8_DYNAMIC arm（合成描述，vision 留 BF16）
int4_cube_lab/results/p10_qwen38_engine.csv   Phase C 交付矩阵
```
