"""Downstream task evaluation of a fake-quant config via lm_eval.

Runs one config at a time against a resident model (see `resident.py`), so both
loglikelihood tasks (GLUE, C-Eval, CMMLU, MMLU) and generative ones (GSM8K) go
through the same path.

Generative tasks are the informative ones for QwQ-32B: it is a reasoning model,
and the concern is error accumulating over a long chain of thought, which
loglikelihood scoring on short continuations cannot show.

Run:
    ASCEND_RT_VISIBLE_DEVICES=2,3 python -m fakequant_lab.run_tasks \\
        --config w4a8-mg1024-asym --tasks gsm8k --limit 200 \\
        --out fakequant_lab/results/tasks_w4a8_asym.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

from .fake_linear import Config, build_configs
from .quant import QuantSpec
from .resident import load_fakequant_model

TASK_SETS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"],
    "reasoning": ["gsm8k"],
    "chinese": ["ceval-valid", "cmmlu"],
}


def resolve_config(name: str, gk: int, gptq_dir: str) -> Config:
    """Config by name, or one derived from a GPTQ directory's own config.json."""
    if gptq_dir:
        meta = json.loads((Path(gptq_dir) / "config.json").read_text())
        return Config(name, QuantSpec(meta["bits"], meta["gk"], "W", sym=meta["sym"]),
                      QuantSpec(8, meta["gk"], "A"))
    for c in build_configs(gk, include_asym=True):
        if c.name == name:
            return c
    raise KeyError(f"unknown config {name!r}; have "
                   f"{[c.name for c in build_configs(gk, include_asym=True)]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B")
    ap.add_argument("--config", required=True, help="e.g. bf16, w8a8, w4a8-mg1024-asym")
    ap.add_argument("--gk", type=int, default=1024)
    ap.add_argument("--gptq-dir", default="", help="calibrated weights from run_gptq.py")
    ap.add_argument("--devices", default="", help="default: every visible NPU")
    ap.add_argument("--tasks", default="gsm8k",
                    help="comma-separated lm_eval task names, or a set: "
                         + ", ".join(TASK_SETS))
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="examples per task")
    ap.add_argument("--batch-size", default="1")
    ap.add_argument("--max-gen-toks", type=int, default=1024,
                    help="QwQ emits long chains of thought; too low truncates them")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    n = torch.npu.device_count() if hasattr(torch, "npu") else 0
    devices = ([d.strip() for d in args.devices.split(",")] if args.devices
               else [f"npu:{i}" for i in range(n)])
    assert devices, "no devices"

    tasks: list = []
    for t in args.tasks.split(","):
        tasks.extend(TASK_SETS.get(t.strip(), [t.strip()]))

    cfg = resolve_config(args.config, args.gk, args.gptq_dir)
    print(f"config={cfg.name}  devices={devices}  tasks={tasks}", flush=True)

    t0 = time.perf_counter()
    model = load_fakequant_model(args.model, cfg, devices, args.gptq_dir or None)
    print(f"model ready in {time.perf_counter() - t0:.0f}s", flush=True)

    lm = HFLM(pretrained=model, tokenizer=args.model, batch_size=args.batch_size,
              max_gen_toks=args.max_gen_toks)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=args.limit,
                                  num_fewshot=args.num_fewshot, bootstrap_iters=0)

    print(f"\n{'task':>16s}  {'metric':>22s}  {'value':>9s}")
    flat = {}
    for task, metrics in res["results"].items():
        for k, v in metrics.items():
            if k == "alias" or not isinstance(v, (int, float)):
                continue
            flat[f"{task}/{k}"] = v
            print(f"{task:>16s}  {k:>22s}  {v:9.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        config=cfg.name, model=args.model, gptq_dir=args.gptq_dir, tasks=tasks,
        limit=args.limit, num_fewshot=args.num_fewshot,
        max_gen_toks=args.max_gen_toks, seconds=time.perf_counter() - t0,
        results=flat), indent=2))
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
