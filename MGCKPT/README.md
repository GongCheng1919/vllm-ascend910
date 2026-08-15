# MGCKPT — mid-group W4A8 任务检查点

`MID_GROUP_ROADMAP.md` 的执行记录。**一期一文件**，随工作推进持续更新，
不要把多期内容混进一个文件。

| 文件 | 阶段 | 状态 |
|---|---|---|
| [P0_COST.md](P0_COST.md) | P0 可行域与代价（kernel 侧探针） | **已完成** — 结论：GK=1024 唯一现实选项 |
| [P1_HARNESS.md](P1_HARNESS.md) | P1 fake-quant harness | **已完成** — 结论：RTN 出局，GK 不是杠杆，P2 直接上 GPTQ |
| [P2_ACCURACY.md](P2_ACCURACY.md) | P2 精度门禁 | **待完成（暂缓）** — PPL 过（+3.52%），GLUE 未过（−4.5 点），门槛未定 |
| [P3_KERNEL.md](P3_KERNEL.md) | P3 kernel 实现 | **已完成（at-risk）** — decode 1.51–1.53× vs W8A8（含量化）；prefill 的 qkv/o 需按投影分派 |
| [P4_E2E.md](P4_E2E.md) | P4 端到端接入 vLLM | **已冻结于 CKPT-E（§0''''）** — 引擎口径每层斜率，`FULL_DECODE_ONLY`。batch=1 我们 W4A8 **408 µs/层**（vs 原生 BF16 2.63×、原生 W8A8 1.94×、我们自己的 W8A8 1.19×）；batch≥96 输给自己的 W8A8。报告 `reports/midgroup/P4_e2e.md`。**第 1 步（权重导出器）、第 4 步未做** |
| [P5_KERNEL_OPT.md](P5_KERNEL_OPT.md) | P5 GEMM 优化（消融定向）| **已关闭（2026-08-15）** — 大 M 的分组冲刷税是 MSD+分组量化的**结构性代价**，六条路只有 L2 分带有净收益（−2~4.6%，逐位一致，**待设默认**）。成本模型：按**冲刷次数/行数**收费，不按字节体积（调度改动全 ≤5%，窄化字节 +1.5~6.2%）。默认保持 int32 累加/下发。见结项总结 |

> **2026-08-13：P2 与 P3 改为并行。** 路线图 §6 原本把 P2 定为硬 go/no-go，
> 现由用户决定偏离——理由、风险与回退路径记在 `P3_KERNEL.md` D2，**不要当成 P2 已通过**。

## 约定

- 每个 CKPT 文件固定四段：**当前状态 / 已完成 / 待办 / 决策与坑**。
- 「决策与坑」记录**为什么这么做**和**踩过什么**，这是重开会话时最值钱的部分。
- 数字一律落盘到 `int4_cube_lab/results/` 或 `reports/midgroup/`，CKPT 里只放结论和路径。
- 环境固定：`export ASCEND_RT_VISIBLE_DEVICES=1`（device 0 已挂死）。
