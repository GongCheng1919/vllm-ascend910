# P3 — kernel 实现（基准：W4A8-mg1024-**asym**）

**状态：已完成（Exit Criteria 全部满足；仍是 at-risk 交付，见 D2）** · 最后更新 2026-08-13
对应 `MID_GROUP_ROADMAP.md` §7 · 代码 `vllm/int4_cube_lab/` · 环境 `ASCEND_RT_VISIBLE_DEVICES=1`

---

## 1. 当前状态

P0 的探针 kernel（`kernels/midgroup_w4a8_gemm.inc`）已经是完整的 mid-group 流水——
分组冲刷、双 workspace slot + 两对 flag、group-major scale、BATCH_G 批量 staging
全部在位且压测通过。P3 起步时只差三件事：喂真实 per-group scale、加非对称零点项、
补上激活侧的量化 kernel。**前两件已完成并验证（§4），只剩量化 kernel 和最终性能扫描。**

**基准冻结为 `W4A8-mg1024-asym`**（用户决定 2026-08-13，见 D1）：GK=1024 为编译期常量，
非对称为默认数值语义。sym 版保留为 A/B 对照 build，不作为交付目标。

### 一句话给接手的人

> 数值契约见 §2。**非对称的性价比问题已经结案（§4.3）：decode 档 +0.2~0.6%，
> 对 W8A8 仍 1.54–1.56×。** 现在起 asym 是无条件的默认，不必再为它做取舍。
>
> **零点数学也已验证（§4.4）**：真实 per-group scale 下 91.7–139.6 dB，race 30/30，
> 且与 `fakequant_lab` 的接缝近乎逐位相同（8192/8192 @ tileM=16）。两个负对照证明
> 检查有牙齿（D5）。
>
> **激活量化 kernel 也完成了（§4.6）**：逐字节正确，`Cast<int4b_t>` 让打包只要一条指令。
> **计入之后 decode 仍有 1.50×**（给 W8A8 也记了同样的量化开销）。
>
> **剩下最终性能扫描。** 另注意 quant kernel 在 prefill 档占 22% 且离带宽还远——
> 它的定位是被融进 rmsnorm，不是独立优化（D7）。

---

## 2. 数值契约（与 `fakequant_lab` 严格一致）

`fakequant_lab/quant.py` 的定义：`x ≈ scale·(q − zero)`，int4 码值仍是**有符号 nibble**
`[-8, 7]`（`quant.py:109`），零点 `zero` 是**码值单位的整数**，以 bf16 存储。
**所以 `W_q` 的存储格式、`PackInt4` 布局、cube 的 s4×s4 通路一律不变。**

激活侧**仍是对称** int8（`fake_linear.py:92` 只把 `sym=False` 给了 W）。

MSD 拆分：`A_q = 16·h + l + 8`，`h, l ∈ [-8,7]`，对全 int8 精确。合并后：

```
y[m,n] = Σ_g  as[g,m] · ws[g,n] · ( 16·S_hi[g] + S_lo[g] + 8·w_ksum[g,n] − wz[g,n]·a_ksum[g,m] )

  S_hi[g] = Σ_{k∈g} h[m,k]·W_q[n,k]        (cube, int32)
  S_lo[g] = Σ_{k∈g} l[m,k]·W_q[n,k]        (cube, int32)
  w_ksum[g,n] = Σ_{k∈g} W_q[n,k]           权重侧预处理，**不变**（这是 MSD 的 +8 偏置）
  a_ksum[g,m] = Σ_{k∈g} A_q[m,k]           **新增**，激活量化 kernel 顺带吐出
  wz[g,n]                                  **新增**，权重零点，group-major bf16
```

> `w_ksum` 用的是原始码值 `W_q`，**不减零点**——它来自 MSD 的 `+8` 偏置，与非对称无关。
> 这两个修正项容易混，写反了 SNR 会掉到 20 dB 量级而不是报错。

张量清单（相对 P0 探针的增量）：

| 张量 | 形状 | 类型 | 来源 |
|---|---|---|---|
| `w_zero` | `[G, N]` group-major | bf16 | 权重侧离线预处理 |
| `a_ksum_neg` | `[G, M]` group-major | **int32** | 激活量化 kernel，**直接吐负值** |

`a_ksum` 吐负值是为了让 §3.1 的融合指令直接可用，省一条 `Muls`；
用 int32 而非 bf16 是**必须的**，理由见 D6。

---

## 3. 相对 P0 探针的改动

### 3.1 AIV：非对称只加一条指令

天真做法要先材料化 outer-product tile `aksum[m]·wz[n]` 再相减，+2 条指令。
但 `AccumulateGroupChunk`（`midgroup_w4a8_gemm.inc:185`）里现成的 `MulAddDst`
能同时吃**两种不同广播模式**的操作数——`BinaryRepeatParams` 对 src0 / src1 各有独立的
blk/rep stride：

```cpp
// dst[m,n] += src0_brcb[m] * src1_row[n]
//   src0: Brcb 块（blkStride 0, repStride 1）  <- (-a_ksum)
//   src1: 64-float 行重读（blkStride 1, repStride 0）  <- wz
BinaryRepeatParams zpParam(1, 0, 1, 16, 1, 0);
MulAddDst(temp[o], negAksumBrcb[ro], wzF32[s], 64, CHUNK_M, zpParam);
```

插在现有 `Add(ksF32)` 之后、`Mul(wsF32)` 之前。**每组每 chunk-半 +1 条向量指令**
（6 → 7 条 tile 等效，+17%），不新增中间 tile。

代价的量级（v4 实测，四投影 M=1 合计）：per-channel 260.5 µs → g1024 285.3 µs，
**mid-group 的全部代价只有 24.8 µs**。非对称的增量落在这 24.8 里面，不是落在 260 的主体上，
所以预期 **1–3%**，不威胁 1.57× 的 W8A8 优势。**但这是推算，§5 第一项就是去实测。**

### 3.2 真正的风险：UB 预算压低 BATCH_G

`kScaleBytesPerG`（`midgroup_w4a8_gemm.inc:130`）要新增：

```
+ TILE_N * sizeof(bfloat16_t)   // w_zero bf16 staging
+ TILE_M * sizeof(int32_t)      // a_ksum_neg int32 staging（见 D6）
+ TILE_N * sizeof(float)        // w_zero fp32
+ TILE_M * sizeof(float)        // a_ksum_neg fp32
+ HALF_M * 8 * sizeof(float)    // a_ksum_neg 的 Brcb 块
```

约 **+60%**，`BATCH_G = (UB − hot − reserve) / kScaleBytesPerG` 相应下降。
而 scale 前导批量化本身值 **17–22%**（P0 §2.5 的探针上界），**这是比 +1 条指令大一个量级的
杠杆**。若 BATCH_G 掉得太狠，优先考虑：

1. `a_ksum_neg` 与 `a_scale` 合并成一个 `[G, 2M]` 张量，省一次 DataCopy 的下发开销；
2. `wz` 与 `ws` 在权重侧预乘成 `wzs = ws·wz`，换掉一次 fp32 cast（但会改 §2 的结合律，
   需要重新对拍 SNR）；
3. 退回每组加载（放弃批量化）只能是最后手段。

### 3.3 host / 参考实现：拆掉 P0 的"退化 scale"捷径 ✅ 已完成（§4.4）

P0 D2 为了让代价数字可信，把 per-channel scale **复制**到每个 group 并复用
`ReferencePerchannelGemm`。P3 已还原为真实 per-group（`--degenerate` 保留旧路径）：

- 生成真实的 per-group `as[G,M]` / `ws[G,N]` / `wz[G,N]`（W 按 `[-8,7]`、A 按全 int8 范围）；
- 新写 `ReferenceMidGroupGemm`（含零点项），CPU 侧按 §2 的式子逐组 fp32 累加；
- **对拍口径**：CPU 参考与 `fakequant_lab` 的 `_kernel_gemm`（`fake_linear.py:155`）
  在同一份输入上应当一致——这是 P3 与 P2 之间唯一的接缝，务必先拉通。

### 3.4 激活侧量化 kernel（新建，路线图 §7 已记为缺口）✅ 已完成（§4.6）

吃 bf16 激活，一趟吐四样：`a_hi` / `a_lo`（int4 平面）、`as[G,M]`（bf16）、
`a_ksum_neg[G,M]`（bf16）。逐组求 max 的 reduce 已经在做，顺带求 sum 几乎免费。

**这一步的开销必须单独计量并计入端到端账**——否则 §5 的性能数字是虚的。

---

## 4. 已完成

### 4.1 非对称 kernel（2026-08-13）

`ASYM_CFG` 编译期开关，签名**无条件**带上 `w_zero` / `a_ksum`（sym build 不读，
host 只有一条路径，A/B 只是翻一个宏）。新增 3 个 build：
`midgroup_w4a8_gemm_m{16,64,128}_g1024_asym`，12 个 build 全部编译通过。

**§3.1 的单指令折叠成立**：`BinaryRepeatParams(1, 0, 1, 16, 1, 0)` 让 `MulAddDst`
的 src0 走 Brcb 块、src1 走行重读，零点项不需要材料化 tile。

**正确性（asym 探针：`w_zero`=0 + 真实 `a_ksum`）**：512×512×2048 三档全 PASS，
**SNR 116.50 dB，与 P0 §2.3 同形状的 sym 数字逐位相同** —— 零点项精确贡献 0，
其余流水未被扰动。这只验证了"没改坏"，**零点数学本身的正确性仍待 §5 第二项**。

### 4.2 BATCH_G 实测（§3.2 标为"真正的风险"，在 decode 档没有兑现）


| TILE_M | sym | asym | QwQ 实际影响 |
|---:|---:|---:|---|
| 16（decode） | 32 | **32** | 无变化（UB 还剩很多） |
| 64 | 32 | 21 | `down`(G=27) 1 批 → 2 批 |
| 128（prefill） | 8 | 5 | `down` 4 批 → 6 批；G=5 的三个投影无变化 |

TILE_M=16 的 hot buffer 只占 18 KB，非对称多要的 1120 B/组 吃不动它。
**代价（若有）集中在 prefill 的 `down`，不在 decode。**

### 4.3 非对称的代价（实测，`results/midgroup_cost_v5_asym.csv`）

QwQ-32B 四种 Linear，cold，中位数 of 3，与 v4 同协议同形状。
**对照：sym 列复现了 v4**（qkv M=1 31.9/31.9、gate_up 149.7/150.2、down 77.0/77.2、
o 26.7/26.8），所以 asym/sym 这一列是干净的。

| M | tileM | w8a8 | W4A8 pc | mg1024 sym | mg1024 **asym** | asym/sym | asym/pc | **asym vs w8a8** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16 | 444.3 | 258.4 | 286.0 | **287.8** | **1.006** | 1.113 | **1.54×** |
| 16 | 16 | 446.3 | 258.8 | 286.3 | **286.8** | **1.002** | 1.108 | **1.56×** |
| 64 | 64 | 516.5 | 328.9 | 381.4 | **393.1** | **1.031** | 1.195 | **1.31×** |
| 512 | 128 | 2083.1 | 1352.8 | 1792.9 | **1816.8** | 1.013 | 1.343 | 1.15× |
| 1024 | 128 | 4066.1 | 2603.1 | 3611.9 | **3663.8** | 1.014 | 1.408 | 1.11× |

（四投影延迟合计，µs／每层每 token 批次。逐投影明细见 CSV。）

**结论：非对称在 decode 档只值 0.2–0.6%，实质免费。** 逐投影的 asym/sym 落在
0.998–1.022，跨过了 0——即在测量噪声内。§3.1 推算的 1–3% 是上界，实际更低。

**最贵的一档是 M=64（+3.1%），不是 prefill。** 这正是 BATCH_G 从 32 掉到 21 的那一档
（§4.2），两件事对得上：代价来自 staging 批次变多，不是来自那条额外的向量指令。
TILE_M=128 的 BATCH_G 掉得更狠（8→5）却只有 +1.3%，因为那里 AIV 的元素工作量已经
足够大，前导摊薄了。

**W8A8 的优势保住了**：decode 档 1.54–1.56×（v4 的 sym 是 1.57×）。

### 4.4 零点数学的正确性 + 与 fakequant 的接缝（2026-08-13）

**`a_ksum` 从 bf16 改成 int32**（数值契约变更，§2 已同步）。GK=1024 时行和可达 ~1.3e5，
bf16 只有 8 位尾数会把它舍到 ~0.2%；而零点项与主项同量级，且 `fakequant_lab` 是 fp32
精确算的——**留 bf16 会让 kernel 与它要实现的算法系统性地对不上**。`w_zero` 保持 bf16
（码值整数 [-8,7]，bf16 精确）。

**harness 改造**：真实 per-group scale + 真实零点成为默认；`--degenerate` 保留 P0 探针路径
（v4/v5 的代价数字仍可复现）；新增 `--load-dir` / `--dump`。

**新的 CPU 参考 `ReferenceMidGroupGemm`**：**按定义算 `Σ_k A·(W − wz)`，不用 kernel 的
rank-1 恒等式**——否则符号写反会两边一起错，检查是循环的（D5）。

正确性（真实 per-group scale + asym，门槛 40 dB）：

| 形状 | M=16 | M=64 |
|---|---:|---:|
| qkv (N=7168, K=5120) | 92.54 dB | 91.66 dB |
| o (N=5120, K=5120) | 139.56 dB | 93.92 dB |
| gate_up (N=55296, K=5120) | 92.40 dB | 94.52 dB |
| down (N=5120, K=27648) | 110.45 dB | 104.20 dB |

race 压测：`m16` 16×2048×5120、`m64` 64×2048×5120、`m128` 512×1024×5120，**各 30/30 PASS**。

**接缝（`scripts/seam_fakequant.py`）**：由 `fakequant_lab` 驱动——它用自己的
`quant.quantize` 量化，`FakeQuantLinear._kernel_gemm` 给出期望输出，同一份整数码与 bf16
scale 经 `--load-dir` 喂进 kernel：

| tileM | 逐位相同 | SNR |
|---:|---|---:|
| 16 | **8192/8192** | 336.29 dB |
| 64 | 32767/32768 | 174.79 dB |
| 128 | 65535/65536 | 177.79 dB |

**两个负对照证明检查有牙齿**（D5）：把 `a_ksum` 符号写反 → −3.26 dB FAIL；
把同一份 asym 数据喂给 sym build → 9.78 dB。

### 4.6 激活量化 kernel（2026-08-13）

`kernels/midgroup_quant_a.inc` + `midgroup_quant_a_g1024`，纯 AIV。
一趟吐四样：`a_hi` / `a_lo`（packed int4）、`a_scale[G,M]` bf16、`a_ksum[G,M]` int32（已取负）。

**打包不需要手写。** `Cast<int4b_t>(half)`（`vconv_f162s4r`）在 910B4 上是支持的，
一条指令完成 nibble 打包。配合浮点域的 MSD 拆分：

```
hi = floor(q / 16)      == q >> 4        （算术右移就是向下取整）
lo = q - 16*hi - 8      == (q & 15) - 8  （host 的 `^8` 写成数值形式）
```

两者都落在 [-8,7]，half 精确表示。**全程不需要位运算。**

**正确性：四个输出对 `quant.py` 的 CPU 模型逐字节相同**（M ∈ {1,4,7,9,16,64,128,512,1024}
× K ∈ {5120,27648} 全 PASS，含非 16 整除的部分 job）。量化是整数产物，没有"接近"一说。

#### V1 的 8 倍水分与修法（用户 2026-08-13 指出）

V1 每个 (row, group) 一个 work item，实测 **0.94 µs/item，与它碰的 3 KB 数据完全无关**：
M=1024/K=27648 有 27648 个 item → 650 µs 搬 85 MB = **129 GB/s**，而本 lab 访存受限的
kernel 能到 ~1 TB/s（W8A8 `down` M=1: 141.6 MB / 124.3 µs = 1139 GB/s）。

两参数模型 `t = 5.06 µs + 0.939 µs × ceil(items/40)` 拟合全部 14 个测点误差 ±2%。
消融定位：**去掉所有写回只省 40 µs，去掉标量往返只省 35 µs**——剩下的是
**~19 条向量指令 × 固定发射开销，每 1024 个元素付一次**。
这就是 P0 §2.5 在 GEMM 上撞到的同一堵 AIV 指令发射墙。

**修法照搬 nitro 的 `midgroup_rowq.cpp` 的 job 结构**（用户指路）：job = 16 行 × 一个组，
向量工作按 8 行的 sub-tile 走，**每条指令覆盖 8192 而不是 1024 个元素**；
per-row absmax 用 chunked `Max` 折叠 + 一次 `WholeReduceMax`。
分组主序还让每个 job 的 16 个 scale / ksum 各自变成**一次连续 DataCopyPad**，
替掉了 V1 每行一次 2 字节 + 一次 4 字节的散写。

**没有照抄的是数值语义**：nitro 用 `inv = 127/rowMax`（未舍入的 max）做乘法，却存
`bf16(rowMax/127)`——量化用的和存下的不是同一个数。`quant.py` 除的是**已舍入**的 scale，
P2 的精度数字就是这么算的，所以这里必须除。**照抄会让逐位接缝失效。**

| M | K | V1 | V2 | |
|--:|--:|--:|--:|--:|
| 1 | 5120 | 7.4 | 7.2 | 1.03× |
| 16 | 27648 | 15.3 | 10.1 | 1.51× |
| 64 | 27648 | 45.7 | 22.5 | 2.04× |
| 128 | 27648 | 85.1 | 38.8 | 2.19× |
| 512 | 27648 | 323.2 | 132.7 | 2.43× |
| 1024 | 27648 | **658.4** | **260.2** | **2.53×** |

M=1024/K=27648 从 129 → **326 GB/s**。**仍有约 3 倍余量**：剩下的是 ~22 遍向量运算
本身的吞吐（每 sub-tile 8192 元素、22 条指令），不再是发射开销。若最终融进 rmsnorm，
其中的读入与部分类型转换会被吸收掉。

**代价（`results/quant_a_cost.csv`，每层 3×K=5120 + 1×K=27648）**：

| M | GEMM asym | quant | quant 占比 | W8A8 GEMM | 含 quant 的加速比 |
|--:|--:|--:|--:|--:|--:|
| 1 | 282.2 | 28.4 | **9.1%** | 446.9 | **1.53×** |
| 4 | 283.5 | 28.9 | 9.3% | 443.6 | **1.51×** |
| 16 | 281.9 | 37.3 | 11.7% | 444.8 | **1.51×** |
| 64 | 387.9 | 51.1 | 11.6% | 513.3 | 1.29× |
| 128 | 545.1 | 73.9 | 11.9% | 642.1 | 1.16× |
| 512 | 1844.2 | 218.2 | 10.6% | 2075.7 | 1.11× |
| 1024 | 3711.3 | 417.3 | 10.1% | 4042.1 | 1.08× |

> 最后一列**给 W8A8 也记了一次同样的量化开销**——W8A8 的 GEMM 同样吃 int8 激活，
> 不给它记账是在偏袒 W4A8。V2 之后 prefill 档 quant 占比从 22% 降到 ~10%。

### 4.5 P0 遗产（直接继承，不重做）

- 分组冲刷流水 + 双 workspace slot + 两对 flag（P0 §2.1 / D5）
- group-major scale 布局 `[G,N]` / `[G,M]`（P0 D4），`fakequant_lab` 已按同布局产出
- BATCH_G 批量 scale staging（P0 §3 的待办已落地，v4 数据即此版本）
- 9 个 build 的几何映射与 static_assert（P0 §2.2）
- race 压测框架（P0 §2.4）
- AIV 的 WAR 屏障下沉到每组（P0「坑」一节）

---

## 5. 待办

- [x] **先测代价，再追精度**（2026-08-13 完成，§4.1–4.3）——asym decode 档 +0.2~0.6%，
      `asym/pc` = 1.11，对 W8A8 仍 **1.54–1.56×**。→ `results/midgroup_cost_v5_asym.csv`
- [x] 拆掉退化 scale，写 `ReferenceMidGroupGemm`（含零点项），与 `fakequant_lab`
      的 `_kernel_gemm` 对拍（2026-08-13，§4.4）——三档 tileM 近乎逐位相同。
- [x] 正确性：QwQ-32B 四种 Linear 形状逐 shape 比对，**91.7–139.6 dB**（门槛 40）。
- [x] race 压测：三个形状各 30/30 PASS。
- [x] 激活量化 kernel（§3.4），开销单独计量（2026-08-13，§4.6）——逐字节正确，
      decode 档占 9–11.5%，加速比 1.54× → **1.50×**。
- [x] **量化 kernel 的 prefill 优化**（2026-08-13，用户指出 1 ms 站不住）——
      照搬 nitro `midgroup_rowq.cpp` 的 16 行 job 结构，**prefill 提速 2.4–2.5×**
      （M=1024/K=27648 658→260 µs，129→326 GB/s），quant 占比 22%→~10%。
      注意：我最初归因于标量往返，**消融证明那只值 35 µs**，真因是向量指令发射（D9）。
- [ ] **（选做）量化 kernel 还剩 ~3 倍余量**：326 GB/s vs 参照 ~1 TB/s。现在是
      ~22 遍向量运算本身的吞吐，不是发射开销。**若融进 rmsnorm，这条大部分作废。**
- [x] 性能扫描：四种 Linear × `M ∈ {1,4,16,64,128,512,1024}`，cold（2026-08-13）。
      → `results/midgroup_w4a8_qwen.csv`，**decode 1.51–1.53×（含量化）满足判据**。
- [x] 裁剪 build 矩阵：交付面只有 GK=1024 × `TILE_M ∈ {16,64,128}` 的 asym；
      sym 与 GK=256/512 保留为对照，不进 P4。
- [x] 写 `reports/midgroup/P3_kernel.md`。
- [ ] **交给 P4 的一条**：prefill 档 `qkv`/`o` 用 mid-group 反而比 W8A8 慢
      （M=1024 为 0.76×/0.79×），应按投影分派——见 §4.7。

---

## 6. 决策与坑

### D1 · 以 `W4A8-mg1024-asym` 为基准，而不是先做 sym 再加 asym（用户决定 2026-08-13）

理由：sym 版的性能数字**不可发布**——主线算法是非对称的（P2 §2.4：sym +9.03% vs
asym +3.56%），先测 sym 再发现 asym 更贵，等于白测一轮。**基准必须与要交付的数值语义一致。**

sym build 保留，但只作为 A/B 对照（用来把"非对称的代价"单独摘出来），不是交付目标。

### D2 · P2 未过门禁就开工 P3，是**明示的并行决策**，不是忘了门禁

路线图 §6 原文：「**门禁在 P2。** P2 不过就不做 P3」。这里是一次有理由的偏离，记录在案：

**理由**：P2 剩下的是算法探索（GSM8K / 中文 / 定门槛），在 NPU 上迭代慢；P3 是 kernel 工程，
两者无依赖。而 P0 D2 的副作用让 P3 的边际成本远低于路线图写作时的估计
（探针已经是完整流水）。继续串行是在浪费墙钟时间。

**但 P2 的当前状态不是「做了一半」，是「当前判据下没过」**：路线图 §6 自己写了
「若三项指标冲突，以**任务评测**为准」，而唯一测过的任务指标 GLUE 是
**asym vs w8a8 平均 −4.5 个百分点**（cola −11.7 / qqp −13.0）。**P3 是 at-risk 施工。**

**回退路径（若 P2 最终判 no-go）**：回退到 **W8A8-mg（−0.10%）**，它复用 P3 的
**同一套分组冲刷流水**和**同一个 per-token mid-group 量化 kernel**，只有 MSD 拆分
和零点项作废。**这是并行成立的真正前提**——不是"反正都要做"，而是最坏情况下
P3 的主体结构仍然有归宿。

### D3 · `w_ksum` 不减零点

见 §2 的注。`w_ksum` 来自 MSD 的 `+8` 偏置，用原始码值；零点走独立的
`wz·a_ksum` 项。两者都是"每组每通道一个标量"，形状相同、来源无关，**写混了不会报错，
只会掉 SNR**。CPU 参考里把这两项分开命名。

### D5 · 参考实现必须按定义算，不能借 kernel 的恒等式

`ReferenceMidGroupGemm` 里零点是**逐个权重减掉**的（`Σ_k A·(W − wz)`），而 kernel 用的是
rank-1 恒等式（`−wz · Σ_k A`）。第一版参考图省事写成了后者——那样**符号写反会两边一起错**，
检查完全是循环的。

**推论：每加一个修正项，就要配一个"故意写错"的负对照。** 本轮做了两个：
`a_ksum` 符号翻转 → **−3.26 dB**；同一份 asym 数据喂 sym build → **9.78 dB**。
没有这两个数，"PASS 116 dB"什么也不能证明。

### D6 · `a_ksum` 必须是 int32，不能是 bf16

GK=1024 的行和到 ~1.3e5，bf16 的 8 位尾数把它舍到 ~0.2%；零点项与主项同量级，
误差直接落到输出上。而 `fakequant_lab` 用 fp32 精确算——**存 bf16 会让 kernel 与它
声称实现的算法系统性偏离，且不会有任何报错**。改 int32 后接缝近乎逐位相同。

对照：`w_zero` 留 bf16 是安全的，它是 `[-8,7]` 的整数码值（`quant.py:111` 的
`zero = qmin - round(mn/scale)`），bf16 精确表示。**判据是"这个量能不能被 bf16 精确
表示"，不是"它是不是 scale 类的量"。**

### D7 · AIV-only kernel 里 `GetSubBlockIdx()` 恒为 0

GEMM 是 mix kernel，AIV 侧用 `GetBlockIdx()*2 + GetSubBlockIdx()` 得到 AIV 编号。
量化 kernel 是**纯 AIV** build，`GetSubBlockIdx()` 恒返回 0，同一个写法只产生**偶数**
编号——**一半的 work item 从来没跑过，输出保持缓冲区里的旧值**，不报错、不越界。

第一次跑出来的现象是 index 0/2 完全正确、1/3 全零。**"一半对一半错"这个形态本身
就是核编号出错的指纹**，不要先去怀疑数值。纯 AIV kernel 用
`GetBlockIdx()` / `GetBlockNum()`，且 host 侧的 `blockDim` 计的是**向量核**（40），
不是 AI 核（20）——同一个参数名，两种单位。

### D9 · 别猜瓶颈，做消融——我在这上面连错两次

量化 kernel V1 是 0.94 µs/work item，**与数据量完全无关**。我先后猜了两次：

1. **猜「两次标量往返排空向量流水」** → 改成全向量域，只快 6%（658→618 µs）。
2. **猜「散写的 2 字节 / 4 字节 DataCopyPad」** → 消融掉**所有**写回只省 40 µs。

真因是**每 1024 个元素付一次 ~19 条向量指令的固定发射开销**——和 P0 §2.5 在 GEMM 上
撞到的是同一堵墙，我却没有第一时间往那想。**修法也和 P0 一样：让每条指令覆盖更多数据。**

**教训**：这类"延迟与数据量无关"的现象，先做消融把候选项一个个减掉，
再动手改。两次猜测各花了一轮编译+实测，一次消融就定位了。

### D8 · 量化 kernel 的定位是被融合掉，不是被优化

decode 档它 6.4 µs 处理 10 KB（qkv 的激活）——**这 6.4 µs 几乎全是下发开销**，
按带宽算数据搬运只值几十纳秒。所以：

- **不要单独优化它的 decode 路径**，优化不掉下发；正解是融进前面的 rmsnorm/激活
  kernel，这也是 vLLM 在 GPU 上的做法。
- **prefill 档是另一回事**：M=1024/K=27648 实测 ~131 GB/s，V1 的每 (row, group)
  两次标量往返把向量流水串起来了。若最终不融合，这里有实打实的余量。

**判断"贵不贵"的分母同 D4**：quant 在 decode 只把加速比从 1.54× 推到 1.50×，
因为 **W8A8 也要付一次同样的量化**。只看 W4A8 单边的绝对增量会得出错误的紧迫感。

### D4 · 非对称的代价要看 24.8 µs 这个分母

判断"asym 贵不贵"不能拿总延迟做分母。mid-group 相对 per-channel 的全部代价在 decode 档
只有 24.8 µs / 285.3 µs。asym 的增量落在**分子的分子**里。同理，若某项优化声称
"降低 5% 总延迟"，要先确认它动的是这 24.8 还是那 260.5。

**已结案（§4.3）**：asym 实测 +1.8 µs / 287.8 µs（decode，M=1），即 mid-group 那
24.8 µs 里再加 7%——**用总延迟做分母是 0.6%**。这个分母的选择正是为什么当初的
"1–3%"推算没有惊动任何决策：它一开始就落在正确的量级上。

---

## 7. 产物

| 路径 | 内容 | 状态 |
|---|---|---|
| `int4_cube_lab/kernels/midgroup_w4a8_gemm.inc` | 主 kernel（待加 asym） | P0 遗产 |
| `int4_cube_lab/results/midgroup_cost_v5_asym.csv` | asym 代价回归 | **已产出** |
| `int4_cube_lab/scripts/seam_fakequant.py` | 与 fakequant_lab 的接缝检查 | **已产出** |
| `int4_cube_lab/kernels/midgroup_quant_a.inc` | 激活量化 + MSD 拆分 | **已产出** |
| `int4_cube_lab/results/quant_a_cost.csv` | 量化 kernel 延迟 | **已产出** |
| `int4_cube_lab/results/midgroup_w4a8_qwen.csv` | 最终性能扫描 | 待产出 |
| `reports/midgroup/P3_kernel.md` | 正确性 + 压测 + 性能 | 待产出 |
