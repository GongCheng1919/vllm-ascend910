# CHECKPOINT — 三个纯 GEMM kernel（bf16/int8/int4）吞吐对比

日期: 2026-08-09 · 环境: 910B4 (16 AIC) + CANN 8.5.0 + torch_npu 2.7.1
接手人: Claude Code（用户指定）· 前序工作: Reasonix 会话（含大量踩坑，见下）

## 任务目标（一句话）

测 910B4 上三个**纯** GEMM kernel（BF16×BF16、INT8×INT8、INT4×INT4，无量化/反量化）
在不同计算规模（方形 2048/4096/8192 + QwQ 形状 K=5120/N=6912）下的吞吐（TFLOPS），
对比硬件指令密度（预期 s8s8=2×bf16、s4s4=4×bf16），并量化 vLLM W4A8 中 MSD
（3×INT4×INT4 分解）的代价。与 vLLM 完全无关，是独立的 kernel 基准。

## 当前状态速览

| kernel | 实现 | 状态 | 方形吞吐 (2048/4096/8192) |
|---|---|---|---|
| gemm_int8 | 手写流水（用户 mid_group_gemm_fwd_lab 结构） | ⚠️ **之前记录的 ✅ PASS 需要复核**——2026-08-09 下午发现小 shape (512×512×1024, 16 tile) 下 correctness 不稳定，见下方"新发现" | 169 / 356 / **378** TFLOPS（官方 npu_quant_matmul 对照 268/418/400，吞吐数字本身未受影响，只是 correctness 覆盖面存疑） |
| gemm_bf16 | 手写流水（改自 int8） | ❌ aicore exception | 用官方 torch.mm 数据: 193/246/228 |
| gemm_int4 | 手写流水（int8 字节视图 + mad_s4） | ✅ **int4 专属 layout bug 已修复**（见下）；多 tile 时命中与 int8 相同的共享竞争 bug | 待跑（先修共享竞争 bug） |

关键结论已达成：int8 手写流水追平官方（94%），证明用户结构正确；tile 流 MatmulImpl
版只有 30-40% 性能（MatmulImpl 黑盒小 tiling 的锅）。

## 2026-08-09 下午更新（Claude Code 接手后）

### int4 layout bug——已修复，3 处，根因是 int4 打包(2元素/字节)的单位换算

`kernels/gemm_int4.cpp` 照抄 `gemm_int8.cpp` 的公式时，`PHYSICAL_K`/`INNER_K` 在 int4
版里是**元素数**（未变，256/1024），但底层 GM/L1 缓冲区是**字节视图**（int4 打包，
2 元素/字节），几处公式忘了除以 2：

1. **`kKBlocksInner`/`kInnerStrideL1Bytes`**：`INNER_K / kK0Int8` 少除一次 2
   （`kK0Int8=32` 是字节块常量，`INNER_K` 是元素数），导致 LoadData 的
   `repeatTimes` 和 K-slice 偏移量全部多算一倍。
2. **`LoadAicTileL1ToL0` 里的 `LoadData` 调用类型是 `int8_t`**——AscendC 对
   `int4b_t` 走独立硬件指令 `load_cbuf_to_ca_s4`/`load_cbuf_to_cb_s4`（见
   `kernel_operator_mm_impl.h` dav_c220），普通 `int8_t` 类型只会走
   `load_cbuf_to_ca`/`load_cbuf_to_cb`（int8 通用指令），产出的 L0 布局不对。
   修法：在 `LoadData` 调用处对 aL0/aL1（已经是字节偏移过的）做
   `.ReinterpretCast<int4b_t>()`，让编译器派发到 `_s4` 指令；GM→L1 的 Nd2Nz
   阶段两者共用同一条 `copy_gm_to_cbuf_multi_nd2nz_b8` 指令（AscendC 对
   int4b_t 直接 reinterpret 成 int8_t 转发，不需要改），**不需要**改。
3. **`aTileBase`/`bTileBase`/`nextGKOffset`/`nextAStart`/`nextBStart`**（tile
   之间的行偏移）都用 `mOff * K` 这种"元素数"算法，同样忘了 `/2`——单 tile
   测试（M=N=128）测不出来（tile 偏移恒为0），多 tile 才会暴露，是"结果全错"
   在大规模下更严重、小规模又诡异 PASS 的真正原因。

验证：单 tile（128×128×1024，全零 A + 真实数据）稳定 PASS 3/3；多 tile
（512×512×1024）后，失败特征从"几乎全错"变成与 int8 完全一致的信号
（见下），证明 int4 专属 bug 已清干净。

### 新发现：int8/int4 共享的多 tile 竞争 bug（未解决，阻塞多 tile 正确性验证）

`gemm_int8.cpp`（以及照抄它结构的 `gemm_int4.cpp`）在 tile 数接近/等于
`blockDim`（16）时，correctness 概率性失败：
- 症状：整块 **128×128 tile**（不是零散元素）结果错误，错误 tile 数量和位置
  每次运行不同；tile 数越多（4→8→16）失败概率越高（4 tile 时 3 次全 PASS，
  8 tile 时 3 次里 2 次 FAIL，16 tile 时基本必现）。
- **已排除**的 4 个假设（逐一实测证伪，见下，不要重复验证）：
  1. AIV 写 GM 后 L2/HBM 未刷新（host DMA 读到脏缓存）——加
     `DataCacheCleanAndInvalid<..., ENTIRE_DATA_CACHE, CACHELINE_OUT>` 无效。
  2. FFTS+ 同步 flag 在进程/launch 起始时残留脏状态——加一次不计时的
     warmup launch 无效（甚至更稳定地失败）。
  3. AIV 内部 `WaitFlag(V0)+write0` / `WaitFlag(V1)+write1` 交错顺序与原始
     模板（先两次 Wait 再两次 write）不同——改成模板顺序无效。
  4. `GetBlockIdx()/GetTaskRation()` 算出的逻辑 blockIdx 在 AIC/AIV 间错配
     （21个物理核间"串台"）——加 printf 实测：16 AIC (taskRation=1) 与 32
     AIV (taskRation=2, 每 2 个 rawBlockIdx 配对成 1 个 logicalBlockIdx)
     的配对**完全正确**，排除。
- **关键对照实验**：`/home/gongcheng/PureInt8LLMPretraining/NPU-OP/
  mid_group_gemm_fwd_lab/kernels/perchannel_int8_gemm.cpp`（checkpoint 引用
  的"已验证 PASS"原始模板，带 per-channel scale）在**完全相同的 512×512×1024
  / 16 tile 配置**下跑 3 次全部稳定 PASS。AIC 侧代码逐行 diff 后与我们的
  `gemm_int8.cpp` **完全一致**；AIV 侧唯一结构性差异是原始模板在
  `CrossCoreWaitFlag(kAicToAivFlag)` 之前有一段真实的 scale 预处理工作
  （DataCopy+Cast+Brcb），我们的纯 GEMM 版本没有——但假设3已经测试过"让
  AIV 顺序更像模板"无效，所以不是纯�時序/指令排列的问题。
  → **意味着这不是我们移植时引入的 bug，是移植去掉 scale 逻辑后才暴露的
  某种潜藏时序敏感性**（很可能是 checkpoint pitfall #9 提到的 MTE3 不能
  直接读 MTE2 写的 UB 缓冲那类硬件时序坑的近亲，但具体机制未定位）。
- **未尝试的方向**（留给下一个接手者）：
  - 用 `DumpTensor` 在 AIC 的 Fixpipe 之后、AIV 的 DataCopy 之前分别转储
    workspace 内容，逐 tile 比对时间戳/内容，而不是靠 host 端最终结果反推。
  - 尝试给 AIV 在 `CrossCoreWaitFlag(kAicToAivFlag)` 之前插入等量的"空转"
    （比如无意义的 Duplicate 填充），而不是真实 scale 工作，看是否单纯是
    "AIV 到达 wait 点的时机"而非"scale 数据本身"在起作用。
  - 检查 `KERNEL_TASK_TYPE_MIX_AIC_1_2` 在极短 kernel（本例纯搬运，无实际
    计算）下是否有已知的 FFTS+ 调度极限（例如某些块在硬件真正完成调度前
    就报告"完成"）——用更长/更慢的 pattern（比如加一个 Duplicate 循环模拟
    真实计算量）测试是否会让 race 消失，可以验证"kernel 太短"这个方向。

### int4 更大 tile (128×256) 的可行性——已探路，未落地

用户建议：int4 打包密度是 int8 的 2x，同样字节预算下 N 方向可以翻倍到 256
（算过预算完全吻合：L0B 256×128(int4元素)/2=32KB/depth×2 ping-pong=64KB 正好
用满 L0B；L0C 128×256×4B(int32累加器)=128KB 正好用满 L0C 全部预算；L1 总计
384KB 在 512KB 预算内）。已实现基础设施改动：`gemm_driver.h` 的 `RunGemm`/
`CheckGemmShape` 加了 `tileM,tileN` 参数（默认 128/128，向后兼容，bf16/int8
不受影响），把 `wsCount`/`dimUsed` 的 tile 尺寸从写死的 128 参数化——**这部分
改动已保留**，为将来做 256 宽 tile 提供了 host 侧支撑。

但把 `kernels/gemm_int4.cpp` 的 `TILE_N` 从 128 改成 256（其余常量都是从
TILE_N 派生，本以为只需改一行）编译通过，跑起来单 tile 就稳定 100% 错
（32749/32768，无 race 特征，纯 deterministic bug）。**判断是 `Mmad`/`mad_s4`
硬件单次调用的 N 方向宽度上限是 128**（PE 阵列/L0C bank 宽度的常见限制），
不是内存预算问题。要做 256 宽 N 需要把内循环拆成两次 128 宽的 Mmad+Fixpipe
调用（每个 K-slice 输出两个 N-half tile 而不是一次 256 宽输出），是一次
真正的 kernel 重构，不是改常量。**已回退**（`TILE_N` 改回 128，
`main_gemm_int4.cpp` 也改回默认 128/128 调用），gemm_int4.cpp 仍处于
"single-tile PASS，多 tile 卡共享竞争 bug" 的状态。

下一个接手者如果要做 256 宽 tile：先确认 `Mmad`/`MmadParams.n` 的硬件上限
（查 `kernel_operator_mm_impl.h` 的 `mad_s4` intrinsic 或问官方文档），如果
确认是 128，则两次 Mmad 分别指向 L0C 的不同 N-half（可能需要 L0C 从单
depth-1 buffer 改成按 N-half 各开一块，或者用同一块但两次 Fixpipe 各输出
一半），同时 AIC 的 Nd2Nz/LoadData 对 B 侧也要相应拆成两个 N-half 的加载。

### 2026-08-09 傍晚补充：race 比最初判断的严重得多——会真挂起，不只是算错

继续排查过程中发现三个重要新证据，推翻了之前"只是概率性算错"的判断：

1. **同一个 shape 的失败率会在会话过程中漂移**：128×512×1024（4 tile）会话最
   开始测是 3/3 稳定 PASS，几十次 NPU 调用之后同一条命令变成 1/4 PASS、
   3/4 FAIL——npu-smi 温度全程 39-43°C 无异常，不是过热。说明"之前测出来
   稳定 PASS"很可能只是小样本的运气，不能作为"这个 shape 没问题"的证据；
   之前证伪的 5 个假设（cache flush/warmup/reorder/blockIdx 映射/wsGm DCCI）
   每个都只测了 5 次，样本量在这个失败率下明显不够，**这些"证伪"结论的置
   信度不如之前汇报时说的那么高**。
2. **加了逐 block workspace 落盘对比后**（新诊断：读回 `workspace`，用每个
   block 最后一次分到的 tile 去对比 CPU 参考值，能区分是 AIC 写错还是 AIV
   读错/时序错——见 `gemm_driver.h` 的 `[ws-check] summary` 输出，这部分诊断
   代码**已保留**在 driver 里，对现有行为无副作用），跑了几次都是
   `0/N blocks have wrong workspace` 但 `[check] c FAIL`——即 **workspace
   本身是对的，AIC 侧计算/Fixpipe/寻址没问题，錯在 AIV 读 workspace 之后到
   写 outGm 之前的某个环节**（缩小了范围：不是 Mmad/Fixpipe/tile 寻址的
   bug，是 AIV 侧或者 AIC↔AIV 交接的问题）。
3. **同一个 128×512×1024 shape 会真的完全挂起**（不是算错，是 kernel 从不
   返回）：用新诊断代码测的时候，有一次跑了 6+ 分钟 CPU time 纹丝不动
   （正常这个 shape 应该 <1 秒完成），`kill -9` 后进程进入不可中断的 D 状态
   又过了几分钟才彻底清除（和本文档更早记录的"环境事故"是同一种现象，但
   **这次确认不是我操作失误导致，是同一个 bug 的另一种表现形式**）。一个
   会偶尔"读到脏数据继续跑"、偶尔"永远等不到该等的信号"的同步原语用法，
   是非常典型的 **CrossCoreSetFlag/CrossCoreWaitFlag 信号丢失或乱序**的
   症状——不是简单的一致性/缓存问题（DCCI 强制刷新试过，无效）。

**当前结论**：这不是"改 int4 引入的小 bug"，是 `mid_group_gemm_fwd_lab` 这
一整套 MIX_AIC_1_2 + `CrossCoreSetFlag`/`CrossCoreWaitFlag` 写法本身在高并
发（多 block 同时用同一对 flag ID）下有概率丢信号/乱序的问题，可能是这套
写法从来没有被高样本量地压测过（之前的"验证 PASS"很可能也是小样本幸运）。
继续在当前会话里盲试局部代码修补（我已经排除了 5 种）性价比很低。

**建议的下一步方向**（比继续裸测更有效）：
- 找 Huawei/CANN 官方文档确认 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 在
  MIX_AIC_1_2、多 block 并发、同一对 flag ID 被所有 block 复用的场景下是否
  有官方认可的用法边界（比如是否要求每个 block 用不同的 flag ID、或者有
  最大并发 block 数限制）。
- 或者换一种不依赖 `CrossCoreSetFlag`/`WaitFlag` 复用同一 flag ID 的同步
  策略（比如 `SyncAll()` 全核屏障，或每个 block 分配独立 flag ID 而不是
  全部共用 `kAivToAicFlag=3`/`kAicToAivFlag=5` 两个 ID）。
- 大样本量回归（每个 shape 跑 20-30 次而不是 3-5 次）来获得真实、可信的
  失败率基线，而不是被小样本误导。

### 环境事故记录（无后续影响，仅供参考）

调试过程中一次不当的 debug printf + `GetValue()` 裸读 L0/L1 tensor（未走
正常同步）导致 AICore 挂起 5 分钟，`kill -9` 该进程后一度怀疑设备状态被
污染。后续用固定 seed 多次重跑排查，确认那只是一次性 transient stall，
且 npu-smi 全程显示设备健康；不是本节描述的竞争 bug的成因（该竞争 bug
在与本次事故完全无关的后续测试里稳定复现）。**教训：不要在 kernel 里对
DataCopy/LoadData 刚写完的 LocalTensor 做未同步的 `GetValue()` 裸读**，
需要用 DumpTensor 或确保正确的 SetFlag/WaitFlag 链路后再读。

## 文件说明（gemm_precision_lab/）

- `kernels/gemm_int8.cpp` — ✅ **可用**。手写流水：L1/L0 depth-2 ping-pong +
  4-slice 软件流水（PHYSICAL_K=1024, INNER_K=256）+ 单 L0C 跨 K 累加 +
  每 tile 一次 Fixpipe + CrossCore 同步 + tile 流 + 跨 tile 预取。
  MIX_AIC_1_2（1 AIC + 2 AIV）。AIV: ws int32 → `Adds`(V 中转) → 输出 int32。
- `kernels/gemm_bf16.cpp` — ❌ WIP。同结构，L0C 用 float（bf16 mmad 输出 fp32），
  PHYSICAL_K=512/INNER_K=128（L1/L0 字节预算减半）。aicore exception，原因见坑 #8。
- `kernels/gemm_int4.cpp` — ❌ WIP。GM 打包字节流（每字节 2 int4，偶 k 低 nibble），
  Nd2Nz 按 int8 字节视图（K/2 字节），Mmad 用 `ReinterpretCast<int4b_t>` + k=INNER_K。
  结果全错，原因见坑 #9。
- `host/gemm_driver.h` — 通用 host（args/check/profile/ACL）。当前为**手写结构**：
  M/N 128 倍数、K 512 倍数、launch(blockDim, stream, a, b, c, workspace, M, N, K)，
  workspace = blockDim×128×128 int32。int4 打包字节数 = 元素数/2（aBytes 已处理）。
  含 DEBUG `[ws-check]`（单 tile 时对比 workspace 与参考，定位 AIC vs AIV）。
- `host/gemm_common.h` — PackInt4（偶 k 低 nibble）、CPU 参考（RefBf16Gemm/RefIntGemm/
  RefInt4Gemm）、GenerateInt8 范围 -8..7（与用户 lab 一致）。
- `host/main_gemm_{bf16,int8,int4}.cpp` — launch lambda（签名见 driver）。
- `scripts/build.sh / run.sh / profile.sh` — 构建/运行/msprof 包装。

复现:
```bash
source scripts/common.sh
cd gemm_precision_lab
bash scripts/build.sh                 # 全部三个 test
bash scripts/run.sh gemm_int8_test --rows 8192 --cols 8192 --k 8192 --profile --warmup 2 --repeat 10
bash scripts/run.sh gemm_int8_test --rows 512 --cols 512 --k 1024 --check   # 正确性
```

## 用户参考（权威结构来源）

`/home/gongcheng/PureInt8LLMPretraining/NPU-OP/mid_group_gemm_fwd_lab/`
- `kernels/perchannel_int8_gemm.cpp` — int8 GEMM + per-channel scale，**本工作手写流水的模板**（432 行，完整 AIC/AIV 流水）
- `kernels/perchannel_int8_gemm_fp16.cpp` — fp16 变体（注意：也是 int32 L0C，与 bf16 不同）
- host/scripts/ 全套 + 该 lab 在 910B4 上验证 PASS

## 踩坑记录（接手者必读，避免重蹈覆辙）

1. **`matmul::Matmul` 别名在 2201 上是 MatmulClient（KFC）**，`REGIST_MATMUL_OBJ`
   注册后 AIC server 空转死锁（AICore 100%）。必须用 `matmul::MatmulImpl`。
   （cann-op-make skill 的 "Wrong patterns" 也写了这条——**先读 skill**：
   `read_skill(name="cann-op-make")`，它是 910B4+CANN8.5 的完整 playbook。）
2. **2201 强制 `MatmulApiStaticTiling` MM_CFG**（`GetMatmulApiTiling<...>(CFG_NORM)`，
   全 -1 动态），普通 CFG_NORM 编译不过（CopyTiling static_assert）。
3. **`KERNEL_TYPE_MIX_AIC_1_2` 下 AIV 也会执行 kernel 主体**——纯 cube 代码必须
   用 `KERNEL_TYPE_AIC_ONLY`（MatmulImpl 场景）或按 `ASCEND_IS_AIC/AIV` 分叉（手写场景）。
4. **`MultiCoreMatmulTiling` 自动多核分块与 MatmulImpl 在 2201 不匹配**：
   core1 不写输出段（静默错）。绕开方案：单核 tiling + 手动切分。
   官方 tiling 参数（baseK=64, dA1=4, dB1=8）与单核 tiling 完全一致，差距在
   N 方向切分（官方 sN=1024/核）而非 tiling 参数。
5. **MatmulImpl 的 C 行 stride = tiling.N**：N 方向切分的 tile 流与标准 [M,N]
   C 布局不兼容 → N 方向必须全宽（tile = tileM 行 × 全 N）。
6. **segM 必须整除 M**（否则最后一核越界写 C，check 检测不到，越界在分配区外）。
7. **910B4 实际 16 AIC**（官方 tiling 只认 24/16，20 核 launch 曾致 M=4096 崩坏；
   后证实崩坏真因是 #8 的 CPU 参考 + 核数无差别）。
8. **driver 的 CPU 参考（genRef）必须只在 --check 时生成**——M=4096 的 double
   三重循环要几分钟，profile 模式会被拖死（曾误判为 kernel 崩坏）。
9. **910B4: MTE3 不能直接读 MTE2 写的 UB 缓冲**——输出从第 8 行起随机错。
   必须 V 管道中转（`Adds(ubOut, ubInt32, 0, ...)` 或 Cast）+ `V_MTE3` 事件
   （SetFlag/WaitFlag）。用户原版的 AIV Cast 缓冲隔离是必需的。
10. **int4 的 Nd2Nz 不能按 int8 字节视图**（K/2 字节）——mad_s4 期望 int4 规则
    的 NZ 布局（C0=64 元素/块，块内排列与 int8 视图不同），当前全错。
    官方参考: `grouped_matmul_swiglu_quant_v2_a8w4_msd_{mid,pre}.h` 里 int4
    搬运全用 `DataCopyPad` 手工处理，无 Nd2Nz int4 直接参考。
11. **bf16 手写 Mmad**: `MmadCal` 支持 Tuple<float, bf16, bf16>（L0C 必须 float），
    但 mad 内建报 `__ca__ short*` 类型期望（可能是 bfloat16_t 的 L0A 类型不匹配），
    aicore exception。需要研究 bf16 的 L0A 类型/LoadData 布局（cann-op-make skill
    有 "bfloat16_t works on 910B4 AIV" 提示但无 GEMM 参考）。
12. **Cast<int,int> 不存在**——int32 中转用 `Adds(...,0)`。
13. tile 流（MatmulImpl 版）在 M=16/32/64/128 时核数用不满（1/2/4/8 核），
    小 M 吞吐天然受限——这是 decode 场景的真实瓶颈，不是 bug。

## 下一步计划（Claude 接手）

1. **int4 手写流水**（最高价值，无官方纯路径竞争，int8 已证明结构可达 378T，
   int4 预期 ~600-700T）：研究 int4 NZ 布局——参考
   `grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h` 的 DataCopyPad 手工搬法，
   或查 `Ascend910B4.ini`/CANN int4 布局文档，让 Nd2Nz/LoadData 按 int4 规则
   （K0=64 元素）工作。
2. **bf16 手写 Mmad**：解决 mad 的 bf16 L0A 类型（可能需 LocalTensor<half>
   存储 bf16 数据 + 特定 LoadData，或查官方 bf16 GEMM kernel）。
3. 三个 kernel 全量数据回归：方形 2048/4096/8192 + QwQ 形状（K=5120, N=6912,
   M=16..4096），更新 `results/20260809_gemm3_custom_kernel/REPORT.md`。
4. 与官方路径对照表（torch.mm / npu_quant_matmul / npu_weight_quant_batchmatmul）。

## 关键参考资料

- 本会话完整报告: `results/20260809_gemm3_custom_kernel/REPORT.md`
- W4A8/MSD 分析: `reports/W4A8_PERGROUP_ANALYSIS.md`（s4s4 硬件支持证据）
- 用户 lab: `/home/gongcheng/PureInt8LLMPretraining/NPU-OP/mid_group_gemm_fwd_lab/`
- skill: `read_skill(name="cann-op-make")`（910B4+CANN8.5 完整 playbook，
  FAST path + validated patterns + 错误决策树）
- 官方 int4 搬运参考: `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/
  tbe/impl/ops_transformer/ascendc/grouped_matmul_swiglu_quant_v2/`
