# P8 — 通信 kernel（**支线**，不是主线）

**状态：未开工（2026-08-19 立项）** · 上游 `P6.5_TP.md` D16–D19
**定位：用户 2026-08-19 明确定为支线任务。主线是 P7。**
环境固定：`export ASCEND_RT_VISIBLE_DEVICES=1`（device 0 已挂死；多卡从 1 号起排）

---

## ★ 一句话给接手的人

**算子已经写完、验完、定完价了（P6.5 D16–D18）。P8 只剩「接进 vLLM 并在引擎上量一次」。**

自研 all-reduce：peer HBM 直读 + ACL IPC 映射，**零 HCCL**，barrier 在 kernel 内
（所以能进捕获图），多核靠跨 rank 对称切分 ⇒ rank 内零同步，**bf16 原生**
（= GEMM 的输出格式，累加仍在 f32）。定价 **解码档 16.9 µs vs HCCL 242–268 µs = 15×**
（TP=2 是 22×）。门禁：6 项 × 2 dtype × TP=2/4 全过，含 skew、200 次无 host barrier、
**24 张图同时存活**、NPUGraph 捕获重放正确。

**为什么是支线**：P6.5 D19 判定它在路线图 DoD 之外——同一个 TP 下三个 arm 付的是
同一笔通信，所以消掉通信改变的是 TP 的绝对延迟，**不改变交付要的三方相对关系**。

**开工前必须先看 P7 的第 1 项**（profile split）。它决定 P8 值不值得做：
- 若通信占 P6.5 那 31 ms/step 的一大块 ⇒ P8 的天花板很高，值得从支线提上来；
- 若通信只占一小部分 ⇒ **P8 不值得开**，问题在 P7。

---

## 1. 继承的结论（不要重新验证）

| 结论 | 来源 |
|---|---|
| all-reduce **纯延迟受限**，10 KB 与 640 KB 同价 ⇒ **消掉它，不要切分它** | P6.5 D11 |
| **没有跨设备 UVA**：各 rank 的 `data_ptr` 数值相同，裸指针会静默读本地 | P6.5 D16；必须 `aclrtIpcMem*` 且断言地址不同 |
| AscendC **运行期下标的指针数组只读到 element 0**，无警告 ⇒ 必须展开 | P6.5 D16 |
| **NPUGraph 的输入张量必须保活**，重放读的是冻结地址 | P6.5 D17 |
| MC2（厂商融合算子）**单跑绿、进 vLLM 必死**，两种图模式都挂 ⇒ 不能拿它定价 | P6.5 D14/D15 |
| 被 kill 的 vLLM worker 会**毒掉 HCCL**，之后所有 run 死在 error 7 | memory |
| 多 rank 探针出问题时，**先怀疑自己的同步**，不要先怪平台 | P6.5 D16 前后三次 |
| **真融合最多再省 ~10 µs/call**，而 drop-in 已经省 ~98 µs ⇒ 融合排最后 | P6.5 D18 |

---

## 2. 待办（按依赖排序）

### 0. 先读 P7 第 1 项的结果，再决定要不要开工。

### 1. drop-in 接进 vLLM 行并行路径

- 在 `MidGroupLinearMethod` 的行并行分支里，把 `dist.all_reduce` 换成
  `torch.ops.npu.peer_allreduce`；
- **staging 与 IPC 映射在 worker 初始化时做一次**（常驻 buffer，之后所有 shape
  都是它的切片——**IPC key 是有限资源**，P6.5 D16）；
- 地址交换要走 vLLM 的 `collective_rpc`，**不能**靠测试里那种用 `dist.all_reduce`
  搬字节的土办法。

### 2. 门禁（引擎内，不是探针内）

- TP=2 与 TP=4 **各验一次**输出与 `dist.all_reduce` 的一致性（rank 数要进归约树）；
- `FULL_DECODE_ONLY` 能否捕获几十张图（探针里过了 24 张，引擎里要再确认一次）；
- ⚠ **`async_scheduling` 必须开**，否则 host 空闲会淹掉这笔收益（P6.5 D12）。

### 3. 引擎级定价

对 P6.5 D20 那张表的同一批测点重跑，报 drop-in 前后的 step 差。
天花板：D13 的 3632 µs 通信（L=16 口径），按 D18 的账 drop-in 能吃掉 ~86%。
⚠ **L=64 的通信量没有实测过**，D20 里那个「按层数外推约 14 ms」是推测。

### 4. 真融合（GEMM epilogue 内归约）—— 可选中的可选

设计已经清楚（输出 tile 划分本就跨 rank 对称 ⇒ 每 tile 一个旗标，rank 内零同步），
唯一的新问题是 epilogue 挂 AIC 侧还是 AIV 侧。**不要在第 3 步出数之前做。**

---

## 3. 决策与坑

（本期开工后填写。P6.5 D14–D18 的坑已在上表继承，不要重复踩。）

---

## 4. 产物

已存在（P6.5 交付，可直接用）：

| 文件 | 内容 |
|---|---|
| `npu_ops/kernel/peer_allreduce_f32.cpp` | 算子本体：in-kernel barrier、双 slot 按 seq 奇偶、多核对称切分；一个模板两个入口 `peer_allreduce_{f32,bf16}` |
| `npu_ops/host/peer_allreduce.cpp` | `npu.peer_allreduce` / `npu.peer_staging_elems` |
| `npu_ops/host/peer_ipc.cpp` | `npu.ipc_{get_bare_tgid,export,import,close}` |
| `npu_ops/python/test_peer_barrier.py` | 6 项门禁 × TP=2/4 |
| `npu_ops/python/bench_peer_allreduce.py` | 定价（我们走图内、HCCL 走 eager）|
| `npu_ops/python/bench_fused_layer.py` | 层级三臂（gemm / +ours / +hccl）|
