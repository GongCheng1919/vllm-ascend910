#!/usr/bin/env python3
"""P6 PP 报告图表生成 —— 高并发 & 长上下文吞吐对比（含全部 5 条 baseline arm）。

数据源：int4_cube_lab/results/p6_pp_concurrency.csv（高并发）
        int4_cube_lab/results/p6_pp_longctx.csv（长上下文，decode + prefill 分离）
输出：reports/midgroup/figs/p6_pp_*.png（英文标签，无 CJK 字体依赖）
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "int4_cube_lab", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# ---- arm 显示名：顺序即图例顺序（bf16 -> native -> 我们） --------------------
ARM_ORDER = ["bf16", "w8a8-native", "w4a8-native", "w8a8", "w4a8"]
ARM_LABEL = {
    "bf16": "BF16 (baseline)",
    "w8a8-native": "Native W8A8*",
    "w4a8-native": "Native W4A8 (wt-only)**",
    "w8a8": "Ours W8A8 (per-ch)",
    "w4a8": "Ours W4A8 (mid-group)",
}
ARM_STYLE = {
    "bf16":        dict(color="#767676", marker="o", ls="--", lw=1.6, zorder=2),
    "w8a8-native": dict(color="#2196F3", marker="s", ls=":",  lw=1.6, zorder=3),
    "w4a8-native": dict(color="#00BCD4", marker="^", ls=":",  lw=1.6, zorder=3),
    "w8a8":        dict(color="#FF9800", marker="v", ls="-",  lw=2.0, zorder=4),
    "w4a8":        dict(color="#E53935", marker="o", ls="-",  lw=2.4, zorder=5),
}
# 高亮我们的两条（加粗/描边）
HILITE = {"w8a8", "w4a8"}


def read_csv(name):
    """读回 dict: (arm, pp, batch, prompt_len) -> row"""
    rows = {}
    with open(os.path.join(ROOT, name), newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["arm"], int(r["pp"]), int(r["batch"]), int(r["prompt_len"]))] = r
    return rows


def plot_lines(ax, data, keys, xvals, xlabel, get_metric, get_x=None, logx=False, logy=False):
    """keys: list of (arm, extra_filters...) -> 每 arm 一组线"""
    for arm in ARM_ORDER:
        if arm not in {k[0] for k in keys}:
            continue
        xs, ys = [], []
        for key in keys:
            if key[0] != arm:
                continue
            row = data.get(key)
            if row is None:
                continue
            x = get_x(key) if get_x else None
            xs.append(x if x is not None else key[2])
            ys.append(get_metric(row))
        if not ys:
            continue
        xs, ys = zip(*sorted(zip(xs, ys)))
        st = ARM_STYLE[arm]
        label = ARM_LABEL[arm] + ("  [ours]" if arm in HILITE else "")
        ax.plot(xs, ys, label=label, **st)
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4, zorder=0)


def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=11, pad=10, weight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")


# =========================================================================
# 图 1：高并发 —— 吞吐 tok/s vs batch（PP=4 / L=64 真实模型，含全部基线）
# p6_pp_concurrency.csv: arm x pp{1,2,4} x batch{1,16,64,128,256}
# =========================================================================
conc = read_csv("p6_pp_concurrency.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
PP = 4
batches = [1, 16, 64, 128, 256]
keys = [(a, PP, b, 128) for a in ARM_ORDER for b in batches]

# 左图：绝对吞吐 tok/s
plot_lines(ax1, conc, keys, batches, "batch (concurrency)", lambda r: float(r["throughput_tok_s"]), logx=True)
style_ax(ax1, f"Throughput vs concurrency — PP={PP}, 64 layers (real model)", "throughput (tok/s)")

# 标注交叉点 batch=256：我们 W4A8 输给我们 W8A8
t_w8 = float(conc[("w8a8", PP, 256, 128)]["throughput_tok_s"])
t_w4 = float(conc[("w4a8", PP, 256, 128)]["throughput_tok_s"])
ax1.annotate(f"cross at b=256:\nW4A8 {t_w4:.0f} vs W8A8 {t_w8:.0f} tok/s\n(W4A8 = {t_w4/t_w8:.2f}×)",
             xy=(256, t_w4), xytext=(70, t_w4 * 0.80),
             fontsize=8.5, color="#E53935",
             arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.2))

# 右图：相对 BF16 的加速比（同 batch 内）
rat_keys = [(a, PP, b, 128) for a in ["bf16", "w8a8", "w4a8"] for b in batches]
bf16_base = {b: float(conc[("bf16", PP, b, 128)]["throughput_tok_s"]) for b in batches}
for arm in ["w8a8", "w4a8"]:
    xs = batches
    ys = [float(conc[(arm, PP, b, 128)]["throughput_tok_s"]) / bf16_base[b] for b in batches]
    ax2.plot(xs, ys, label=ARM_LABEL[arm], **ARM_STYLE[arm])
ax2.axhline(1.0, color="#767676", ls="--", lw=1.0)
ax2.text(1.15, 1.02, "1.0 = BF16 parity", fontsize=7.5, color="#767676")
ax2.set_xscale("log")
ax2.set_xlabel("batch (concurrency)")
ax2.set_ylabel("speedup vs BF16 (tok/s ratio)")
ax2.grid(True, which="both", ls=":", alpha=0.4)
ax2.legend(fontsize=8, loc="best")
ax2.set_title(f"Speedup vs BF16 — PP={PP}, 64 layers\n(decays with concurrency; W4A8 crosses W8A8 at b≈256)",
              fontsize=11, pad=10, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6_pp_highconcurrency.png"), dpi=150)
plt.close(fig)

# =========================================================================
# 图 2 & 3：长上下文 —— 解码吞吐 tok/s 与 prefill 延迟 ms（PP=4/L64）
# p6_pp_longctx.csv: arm x plen{128,1024,4096,8192} x batch{1,8}
# =========================================================================
lc = read_csv("p6_pp_longctx.csv")
plens = [128, 1024, 4096, 8192]

# ---- 图 2：解码吞吐（batch=1 实线，batch=8 虚线）----
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for arm in ARM_ORDER:
    for b, ls in [(1, "-"), (8, "--")]:
        st = dict(ARM_STYLE[arm])
        st["ls"] = ls
        xs, ys = [], []
        for pl in plens:
            row = lc.get((arm, 4, b, pl))
            if row:
                xs.append(pl)
                ys.append(float(row["throughput_tok_s"]))
        if not ys:
            continue
        label = ARM_LABEL[arm] + ("" if arm not in HILITE else "  [ours]") + f"  b={b}"
        ax.plot(xs, ys, label=label, **st)
ax.axvline(1024, color="#B0BEC5", ls=":", lw=1.0)
ax.text(1050, ax.get_ylim()[1] * 0.96, "KV-heavy decode\n(GEMM rows = batch)", fontsize=8, color="#607D8B")
ax.set_xscale("log")
ax.set_xlabel("context length (prompt_len)")
ax.set_ylabel("decode throughput (tok/s)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=7.5, framealpha=0.9, loc="upper right", ncol=2)
ax.set_title("Long context — decode throughput (PP=4, 64 layers, prefix-cached)\n"
             "Ours W4A8 degrades least: 128→8192 = +9.6% (vs BF16 +8.2%, our W8A8 +17.0%)",
             fontsize=11, pad=10, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6_pp_longctx_decode.png"), dpi=150)
plt.close(fig)

# ---- 图 3：prefill 延迟 ms（越低越好）----
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for arm in ARM_ORDER:
    for b, ls in [(1, "-"), (8, "--")]:
        st = dict(ARM_STYLE[arm])
        st["ls"] = ls
        xs, ys = [], []
        for pl in plens:
            row = lc.get((arm, 4, b, pl))
            if row and row.get("prefill_ms"):
                xs.append(pl)
                ys.append(float(row["prefill_ms"]))
        if not ys:
            continue
        label = ARM_LABEL[arm] + ("" if arm not in HILITE else "  [ours]") + f"  b={b}"
        ax.plot(xs, ys, label=label, **st)
ax.annotate("ours W4A8 prefill = only 0.64–0.69× of ours W8A8\n(W4A8 is a decode-only scheme)",
            xy=(8192, 8069.81), xytext=(1400, 5200),
            fontsize=8.5, color="#E53935",
            arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.2))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("prompt length (context)")
ax.set_ylabel("prefill latency (ms, fresh prompt, max_tokens=1)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=7.5, framealpha=0.9, loc="upper left", ncol=2)
ax.set_title("Long context — prefill latency (PP=4, 64 layers)\n"
             "big-M flush tax hits mid-group: W4A8 loses to our W8A8 on every point",
             fontsize=11, pad=10, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6_pp_longctx_prefill.png"), dpi=150)
plt.close(fig)

print("figs written:")
for fn in sorted(os.listdir(OUT)):
    print(" ", os.path.join(OUT, fn))