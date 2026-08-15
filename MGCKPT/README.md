# MGCKPT — mid-group W4A8 任务检查点

`MID_GROUP_ROADMAP.md` 的执行记录。**一期一文件**，随工作推进持续更新，
不要把多期内容混进一个文件。

| 文件 | 阶段 | 状态 |
|---|---|---|
| [P0_COST.md](P0_COST.md) | P0 可行域与代价（kernel 侧探针） | **已完成** — 结论：GK=1024 唯一现实选项 |
| [P1_HARNESS.md](P1_HARNESS.md) | P1 fake-quant harness | **已完成** — 结论：RTN 出局，GK 不是杠杆，P2 直接上 GPTQ |
| [P2_ACCURACY.md](P2_ACCURACY.md) | P2 精度门禁 | **待完成（暂缓）** — PPL 过（+3.52%），GLUE 未过（−4.5 点），门槛未定 |
| [P3_KERNEL.md](P3_KERNEL.md) | P3 kernel 实现 | **已完成（at-risk）** — decode 1.51–1.53× vs W8A8（含量化）；prefill 的 qkv/o 需按投影分派 |
| [P4_E2E.md](P4_E2E.md) | P4 端到端接入 vLLM | **CKPT-B：第 3 步已跑通（§0'）** — kernel 在真引擎 ACL 图里 replay，**1.34× / 层**；手搭基线冻结在 CKPT-A（§0）。下一步：补齐做进算子，占 33% 解码设备时间（D10）|
| [P5_KERNEL_OPT.md](P5_KERNEL_OPT.md) | P5 GEMM 优化（消融定向）| **进行中（2026-08-14 重开，靶子换成大 M）** — 小 M 那期的「纯 GM→L1 受限、只剩 25 µs」只对 M≤128 成立。大 M 消融（§3''）定位到：分组税 100% 在 Fixpipe（`nofix` 就等于 per-channel）。**调度层已走完：四次改动全部 ≤5%**（TILE_N=64 纸上出局；组内交替 +3%；两趟式真跨组双缓冲 中性；L2 分带 −2~4.6%，逐位过、建议设默认）。**砍字节也不行**：窄化下发（int32→fp16，SNR 72–74 dB）全形状慢 1.5~6.2%（D10）——代价跟的是冲刷**次数/行数**而非字节体积。**本期到此为止**，大 M 与 W8A8 持平即可，重心回小 M。收尾：分带设默认 + 同步 npu_ops |

> **2026-08-13：P2 与 P3 改为并行。** 路线图 §6 原本把 P2 定为硬 go/no-go，
> 现由用户决定偏离——理由、风险与回退路径记在 `P3_KERNEL.md` D2，**不要当成 P2 已通过**。

## 约定

- 每个 CKPT 文件固定四段：**当前状态 / 已完成 / 待办 / 决策与坑**。
- 「决策与坑」记录**为什么这么做**和**踩过什么**，这是重开会话时最值钱的部分。
- 数字一律落盘到 `int4_cube_lab/results/` 或 `reports/midgroup/`，CKPT 里只放结论和路径。
- 环境固定：`export ASCEND_RT_VISIBLE_DEVICES=1`（device 0 已挂死）。
