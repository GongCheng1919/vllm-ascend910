#!/usr/bin/env python3
"""Turn `p65_tp_e2e.csv` into the P6.5 TP=4 delivery tables.

Three tables, never blended:

  concurrency : plen=1024, out=256 -- the roadmap's own decode-dense point
  long context: plen 128..8192, batch 1/8 -- decode with a growing KV
  prefill     : the `prefill_ms` column of BOTH axes, reported on its own

WHY PREFILL IS SPLIT OUT.  Decode and prefill move in OPPOSITE directions for
mid-group W4A8: decode is small-M (our win), prefill is large-M where the group
flush tax is structural (P5) and PP already measured 0.64-0.69x.  A blended
"end-to-end" number hides both halves, so this script refuses to produce one.

RATIOS.  Two denominators, because they answer different questions:
  vs bf16 : "is quantisation worth it at all"
  vs w8a8 : "is W4A8 worth it over W8A8" -- OURS vs OURS, same weights, same
            hook, same 64 linears, so the ratio isolates the GEMM.
The `-native` arms are the vendor's answer to a THIRD question and are printed
but never used as a denominator (vllm-ascend leaves all down_proj in FLOAT --
P6 D12 -- so it is not the same model).

Engine reproducibility is +-15% (P6 D11), single runs.  Anything under ~5% is
printed with a `~` and is NOT a claim.

usage: python reports/midgroup/make_p65_tp_tables.py [csv] [--md out.md]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

ARM_ORDER = ["bf16", "w8a8", "w4a8", "w8a8-native", "w4a8-native"]
NOISE = 0.05          # P6 D11: +-15% run-to-run; under 5% is not a claim


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("step_us") or r["step_us"] == "FAILED":
                continue
            r["batch"] = int(r["batch"])
            r["prompt_len"] = int(r["prompt_len"])
            r["out_len"] = int(r["out_len"])
            r["step_us"] = float(r["step_us"])
            r["throughput_tok_s"] = float(r["throughput_tok_s"])
            r["prefill_ms"] = float(r["prefill_ms"]) if r.get("prefill_ms") else None
            r["async_sched"] = int(r.get("async_sched") or 0)
            rows.append(r)
    return rows


def ratio(num, den):
    """num/den as 'a.bbx', flagged when inside the noise band."""
    if not den or not num:
        return "-"
    v = den / num                      # step_us: smaller is faster => den/num
    mark = "~" if abs(v - 1.0) < NOISE else ""
    return f"{mark}{v:.2f}x"


def table(rows, key, key_name, value, unit, title, note=""):
    """One table: arms as columns, `key` as rows."""
    by = defaultdict(dict)
    for r in rows:
        v = r[value]
        if v is not None:
            by[r[key]][r["arm"]] = v
    arms = [a for a in ARM_ORDER if any(a in d for d in by.values())]
    out = [f"\n### {title}", ""]
    if note:
        out += [note, ""]
    hdr = f"| {key_name} | " + " | ".join(arms) + " | w4a8 vs bf16 | w4a8 vs w8a8 |"
    out += [hdr, "|" + "---|" * (len(arms) + 3)]
    for k in sorted(by):
        d = by[k]
        cells = [f"{d[a]:.1f}" if a in d else "-" for a in arms]
        r_bf = ratio(d.get("w4a8"), d.get("bf16")) if value != "throughput_tok_s" \
            else ratio(d.get("bf16"), d.get("w4a8"))
        r_w8 = ratio(d.get("w4a8"), d.get("w8a8")) if value != "throughput_tok_s" \
            else ratio(d.get("w8a8"), d.get("w4a8"))
        out.append(f"| {k} | " + " | ".join(cells) + f" | {r_bf} | {r_w8} |")
    out += ["", f"_{unit}.  `~` = inside the +-5% noise floor, not a claim._"]
    return out


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "int4_cube_lab/results/p65_tp_e2e.csv"
    rows = load(path)
    if not rows:
        print(f"no usable rows in {path}")
        return 1

    tp = rows[0]["tp"]
    L = rows[0]["layers"]
    out = [f"# P6.5 · TP={tp} delivery matrix (L={L}, real QwQ-32B geometry)", "",
           f"source: `{path}`  ·  five arms  ·  `FULL_DECODE_ONLY`  ·  single runs",
           "", "**The HCCL all-reduce is an arm-independent tax at TP>1** (3632 us/step,",
           "`Overlapped=0`, P6.5 D9), so every ratio here is DILUTED relative to the",
           "same ratio at TP=1 (2.28x vs bf16, D12).  That is a property of TP, not a",
           "measurement error; D17/D18 priced removing ~86% of it and that work is",
           "shelved, not in this path (D19)."]

    # async_sched=1 is the delivery caliber; the off rows are a control group.
    del_rows = [r for r in rows if r["async_sched"] == 1]
    off_rows = [r for r in rows if r["async_sched"] == 0]

    conc = [r for r in del_rows if r["out_len"] == 256]
    ctx = [r for r in del_rows if r["out_len"] == 64]

    if conc:
        out += table(conc, "batch", "batch", "step_us", "us per decode step",
                     "Concurrency — decode step (plen=1024, out=256)")
        out += table(conc, "batch", "batch", "throughput_tok_s", "tok/s (higher is better)",
                     "Concurrency — decode throughput")
        out += table(conc, "batch", "batch", "prefill_ms", "ms for one full prefill of batch x 1024 tokens",
                     "Concurrency — PREFILL (separate pass, unseen prompts)",
                     "Large-M regime: mid-group pays the P5 flush tax here and is "
                     "expected to LOSE, as it did under PP (0.64-0.69x, P6 D13).")
    if ctx:
        for b in sorted({r["batch"] for r in ctx}):
            sub = [r for r in ctx if r["batch"] == b]
            out += table(sub, "prompt_len", "prompt_len", "step_us", "us per decode step",
                         f"Long context — decode step (batch={b}, out=64)",
                         "M stays = batch while attention grows with the KV, so the "
                         "GEMM win gets diluted: the ratio should sag toward 1.0.")
            out += table(sub, "prompt_len", "prompt_len", "prefill_ms", "ms",
                         f"Long context — PREFILL (batch={b})")

    if off_rows:
        out += ["", "### async_scheduling control (same conc axis, flag OFF)", "",
                "D12 measured 1.23-1.29x for this flag at L=16, where device idle was "
                "44% of the step.  At L=64 idle should fall to ~17%, so the flag is "
                "predicted to be worth LESS here -- this is that test.", "",
                "| arm | batch | step_us OFF | step_us ON | async worth |",
                "|---|---|---|---|---|"]
        on = {(r["arm"], r["batch"]): r["step_us"] for r in conc}
        for r in sorted(off_rows, key=lambda x: (x["arm"], x["batch"])):
            k = (r["arm"], r["batch"])
            if k in on:
                out.append(f"| {r['arm']} | {r['batch']} | {r['step_us']:.1f} | "
                           f"{on[k]:.1f} | {ratio(on[k], r['step_us'])} |")

    text = "\n".join(out)
    print(text)
    if "--md" in sys.argv:
        dst = sys.argv[sys.argv.index("--md") + 1]
        with open(dst, "w") as f:
            f.write(text + "\n")
        print(f"\n[wrote] {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
