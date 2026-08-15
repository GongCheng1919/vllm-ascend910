# P3 — mid-group W4A8 kernel（基准 `W4A8-mg1024-asym`）

**日期** 2026-08-13 · **硬件** Ascend 910B4 · CANN 8.5.0 · `ASCEND_RT_VISIBLE_DEVICES=1`
**代码** `vllm/int4_cube_lab/` · **执行记录** `MGCKPT/P3_KERNEL.md`
对应 `MID_GROUP_ROADMAP.md` §7

> 本报告按产物契约分开陈述**执行成功**与**性能结论**。
> **前置声明：P3 是 at-risk 施工。** P2 精度门禁**没有通过**——PPL 过了（+3.52%），
> 但路线图 §6 自己写明「以任务评测为准」，而唯一测过的任务指标 GLUE 是
> **asym vs W8A8 平均 −4.5 个百分点**（cola −11.7 / qqp −13.0）。P2 与 P3 并行是
> 2026-08-13 的明示决策（`MGCKPT/P2_ACCURACY.md` D8），**本报告的性能数字不构成
> 方案可用的证明**。

---

## 1. 交付了什么

| 项 | 状态 |
|---|---|
| mid-group W4A8 GEMM，非对称权重，GK=1024 编译期常量，TILE_M ∈ {16,64,128} | ✅ |
| 激活量化 kernel（bf16 → 两个 int4 平面 + per-group scale + per-group 行和） | ✅ |
| CPU 参考（按定义，非 kernel 恒等式）+ 逐 shape 正确性 | ✅ |
| race 压测 | ✅ |
| 与 `fakequant_lab` 的算法接缝检查 | ✅ |
| QwQ-32B 四形状 × M ∈ {1,4,16,64,128,512,1024} cold 扫描 | ✅ |

---

## 2. 数值契约

激活对称 int8、权重非对称 int4（每组零点），A/W 共用 K 方向的组边界：

```
y[m,n] = Σ_g as[g,m]·ws[g,n]·( 16·S_hi[g] + S_lo[g] + 8·w_ksum[g,n] − wz[g,n]·a_ksum[g,m] )
```

与 `fakequant_lab/quant.py` 逐条对齐：int4 码值是有符号 nibble `[-8,7]`，
零点是码值单位的整数，scale 在**量化之前**就 round 到 bf16。

**两个容易写错、且不会报错的地方：**

1. **`w_ksum` 不减零点。** 它来自 MSD 拆分的 `+8` 偏置，与非对称无关。
   零点走独立的 `wz·a_ksum` 项。两者形状相同、来源无关，混用只掉 SNR 不报错。
2. **`a_ksum` 必须是 int32，不能是 bf16。** GK=1024 的行和达 ~1.3e5，bf16 的
   8 位尾数会舍到 ~0.2%；零点项与主项同量级，误差直接落到输出。而 fakequant 是
   fp32 精确算的——**存 bf16 会让 kernel 与它声称实现的算法系统性偏离**。
   对照：`wz` 留 bf16 是安全的（`[-8,7]` 的整数，bf16 精确）。
   **判据是"这个量能否被 bf16 精确表示"，不是"它是不是 scale 类的量"。**

---

## 3. 执行成功（正确性）

### 3.1 逐 shape 比对（真实 per-group scale + 真实零点，门槛 SNR ≥ 40 dB）

| QwQ-32B Linear | M=16 | M=64 |
|---|---:|---:|
| qkv (N=7168, K=5120) | 92.54 dB | 91.66 dB |
| o (N=5120, K=5120) | 139.56 dB | 93.92 dB |
| gate_up (N=55296, K=5120) | 92.40 dB | 94.52 dB |
| down (N=5120, K=27648) | 110.45 dB | 104.20 dB |

CPU 参考按**定义**计算 `Σ_k A·(W − wz)`，**不用 kernel 的 rank-1 恒等式**——
用了的话符号写反会两边一起错，检查是循环的。

### 3.2 race 压测（tile 数 > blockDim）

| build | shape | 结果 |
|---|---|---|
| m16_g1024_asym | 16×2048×5120 | 30/30 PASS |
| m64_g1024_asym | 64×2048×5120 | 30/30 PASS |
| m128_g1024_asym | 512×1024×5120 | 30/30 PASS |

### 3.3 与算法的接缝（`scripts/seam_fakequant.py`）

其余检查都是"kernel vs 同目录下、共享同一套约定的 C++ 参考"，**测不出算法与 kernel
之间的分歧**。接缝检查由 `fakequant_lab` 驱动：它用自己的 `quant.quantize` 量化，
`FakeQuantLinear._kernel_gemm` 给出期望输出，同一份整数码与 bf16 scale 经
`--load-dir` 喂进 kernel。

| tileM | 逐位相同 | SNR |
|---:|---|---:|
| 16 | **8192/8192** | 336.29 dB |
| 64 | 32767/32768 | 174.79 dB |
| 128 | 65535/65536 | 177.79 dB |

### 3.4 负对照（**这一节决定上面三节值不值钱**）

| 故意引入的错误 | 结果 |
|---|---:|
| `a_ksum` 符号翻转 | **−3.26 dB FAIL** |
| 同一份 asym 数据喂给 sym build（编译期丢掉零点项）| **9.78 dB** |

没有这两个数，"PASS 116 dB" 只能证明两边一致，不能证明两边正确。
**推论：每加一个修正项，配一个故意写错的负对照。**

### 3.5 激活量化 kernel

四个输出（`a_hi` / `a_lo` / `a_scale` / `a_ksum`）对 `quant.py` 的 CPU 模型
**逐字节相同**，M ∈ {1,16,64,1024} × K ∈ {5120,27648} 全 PASS。
量化是整数产物，用逐字节而不是 SNR。

量化 kernel 用 nitro `midgroup_rowq.cpp` 的 job 结构（16 行 × 一组，8 行 sub-tile），
每条向量指令覆盖 8192 个元素；prefill 档 326 GB/s（仍有 ~3 倍余量，见 D9）。

打包不需要手写：`Cast<int4b_t>(half)`（`vconv_f162s4r`）一条指令完成 nibble 打包。
MSD 拆分在浮点域做，全程无位运算：

```
hi = floor(q/16)     == q >> 4          （算术右移即向下取整）
lo = q − 16·hi − 8   == (q & 15) − 8    （host 的 `^8` 的数值形式）
```

---

## 4. 性能结论

### 4.1 每层合计（QwQ-32B 四种 Linear 之和，µs，cold，中位数 of 3）

`results/midgroup_w4a8_qwen.csv` + `results/quant_a_cost.csv`。
`quant` = 每层的激活量化：3×(K=5120) + 1×(K=27648)。

| M | W8A8 | W4A8 pc | **mg1024-asym** | quant | asym+quant | 仅 GEMM | **含 quant** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 446.9 | 257.2 | **282.2** | 28.4 | 310.6 | 1.58× | **1.53×** |
| 4 | 443.6 | 257.7 | **283.5** | 28.9 | 312.4 | 1.56× | **1.51×** |
| 16 | 444.8 | 258.6 | **281.9** | 37.3 | 319.2 | 1.58× | **1.51×** |
| 64 | 513.3 | 327.5 | **387.9** | 51.1 | 439.0 | 1.32× | 1.29× |
| 128 | 642.1 | 495.5 | **545.1** | 73.9 | 619.0 | 1.18× | 1.16× |
| 512 | 2075.7 | 1348.0 | **1844.2** | 218.2 | 2062.4 | 1.13× | 1.11× |
| 1024 | 4042.1 | 2598.1 | **3711.3** | 417.3 | 4128.6 | 1.09× | 1.08× |

> 最后一列**给 W8A8 也记了同一份量化开销**（它的 GEMM 同样吃 int8 激活）。
> 不给它记账会把 W4A8 的优势夸大。

### 4.2 Exit Criteria 核对

路线图 §7 的判据是「**decode 档延迟低于同结构 W8A8 kernel**」：

- **M ∈ {1,4,16}：1.51–1.53×（含量化），满足。**
- M=64 降到 1.26×，M=128 到 1.14×——仍然优于 W8A8，但优势快速收窄。
- **M ≥ 512（prefill）：仅 GEMM 1.09–1.13×，含量化 1.07–1.10×。**

**逐投影看，prefill 档 `qkv` / `o` 实际上比 W8A8 慢**（M=1024 分别 0.76× / 0.79×）：
它们 N 小、K 小，mid-group 的每组冲刷摊不开，`asym/pc` 到 1.45–1.47×。
合计仍 >1 是靠 `gate_up` / `down` 撑着（1.11× / 1.19×）。
**若 P4 要在 prefill 上也用 mid-group，qkv/o 应该走 per-channel W4A8 或 W8A8。**

### 4.3 mid-group 相对 per-channel 的代价

`asym/pc` 在 decode 档是 **1.05–1.12×**，M=512/1024 涨到 **1.34–1.47×**。
这与 P0 的归因一致：代价是**每组固定的 AIV 指令发射 + 冲刷**，与 M 无关；
M 小时它占比小（分母也小但 cube 更闲），M 大时 AIV 成为瓶颈。

### 4.4 非对称本身几乎不要钱

同协议的 sym/asym A/B（`results/midgroup_cost_v5_asym.csv`）：decode 档
**asym/sym = 1.002–1.006**，最贵的是 M=64 的 1.031（BATCH_G 32→21 所致，
不是那条额外的向量指令）。**非对称不是性能上的取舍点。**


---

## 5. 决策与坑（完整版见 `MGCKPT/P3_KERNEL.md` §6）

- **D5 参考实现必须按定义算**，不能借 kernel 的恒等式，否则检查是循环的。
- **D6 `a_ksum` 用 int32**，见 §2。
- **D7 纯 AIV kernel 里 `GetSubBlockIdx()` 恒为 0**：GEMM 是 mix kernel，用
  `blockIdx*2 + subBlockIdx` 取 AIV 编号；量化 kernel 是 AIV-only build，同一写法
  只产生**偶数**编号，一半 work item 从没跑过、输出保持缓冲区旧值，**不报错**。
  「一半对一半错」是核编号出错的指纹，不要先怀疑数值。
  host 侧 `blockDim` 对 AIV-only kernel 计的是**向量核**（40）不是 AI 核（20）。
- **D8 量化 kernel 在 decode 档是被融合掉，不是被优化**：6.4 µs 处理 10 KB，
  几乎全是下发开销。
- **D9 别猜瓶颈，做消融**：量化 kernel V1 是 0.94 µs/work item 且与数据量无关。
  我先猜标量往返（实测只值 35 µs），再猜散写（只值 40 µs），真因是**每 1024 个
  元素付一次 ~19 条向量指令的固定发射开销**——与 P0 §2.5 同源。照搬 nitro
  `midgroup_rowq.cpp` 的 16 行 job 结构（每条指令覆盖 8192 元素）后 prefill 快 2.5×
  （658→260 µs，129→326 GB/s）。语义上没照抄 nitro：它用未舍入的 max 做乘法却存
  舍入后的 scale，会打破与 `quant.py` 的逐位接缝。

---

## 6. 产物

| 路径 | 内容 |
|---|---|
| `int4_cube_lab/kernels/midgroup_w4a8_gemm.inc` | GEMM（`ASYM_CFG` 编译期开关）|
| `int4_cube_lab/kernels/midgroup_quant_a.inc` | 激活量化 + MSD 拆分 |
| `int4_cube_lab/host/quant_harness.h` | 量化 kernel 的 CPU 模型与计时 |
| `int4_cube_lab/scripts/seam_fakequant.py` | 与 fakequant_lab 的接缝检查 |
| `int4_cube_lab/results/midgroup_w4a8_qwen.csv` | 最终性能扫描 |
| `int4_cube_lab/results/midgroup_cost_v5_asym.csv` | asym vs sym 代价回归 |
| `int4_cube_lab/results/quant_a_cost.csv` | 量化 kernel 延迟 |
