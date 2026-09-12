"""P10 Phase B: does our engine plumbing still reach the model on vLLM 0.23.0?

Everything else in the symbol audit is a plain import (P10 §Phase B).  These three
cannot be checked that way -- they only exist once an engine is up:

  1. `get_model()`  -- a PRIVATE attribute chain into the V1 engine core.  This is
     the highest-risk item in the whole migration: it is not API, it is us reaching
     through four layers of vLLM internals.
  2. `collective_rpc` -- how the per-rank `converted=` evidence is collected.
     A parent-process counter is ALWAYS 0 under multi-rank (P6 D8), so this is the
     only honest way to prove an arm did what its name says.
  3. `apply_model` -- the per-layer `quant_method` census, the check that caught
     the vendor W8A8 leaving all 64 down_proj in FLOAT (P6 D12).

Deliberately BF16 + dummy weights + 2 layers: this probe is about the PLUMBING,
not about numbers.  Our own kernels are not loaded (the .so is still built against
the CANN 8.5 / torch_npu 2.8 venv), and they are not needed to answer the question.

usage:  source $HOME/Ascend/cann-9.1.0/set_env.sh
        ASCEND_RT_VISIBLE_DEVICES=1 .venv-023/bin/python npu_ops/python/probe_engine_paths_023.py
"""
from __future__ import annotations

import os
import sys
import traceback

MODEL = os.environ.get("PROBE_MODEL", "models/QwQ-32B")
LAYERS = int(os.environ.get("PROBE_LAYERS", "2"))


def ok(label, val=""):
    print(f"  OK    {label}" + (f"  -> {val}" if val else ""))


def fail(label, e):
    print(f"  FAIL  {label}\n          {type(e).__name__}: {str(e)[:200]}")


def main() -> int:
    from vllm import LLM

    print(f"=== 起引擎 {MODEL} layers={LAYERS} tp=1 bf16 dummy ===")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        load_format="dummy",
        hf_overrides={"num_hidden_layers": LAYERS},
        max_model_len=256,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("=== 1. get_model() 的私有路径 ===")
    model = None
    for name, fn in (
        ("v0  driver_worker.model_runner.model",
         lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model),
        ("v1  engine_core.engine_core.model_executor...",
         lambda: llm.llm_engine.engine_core.engine_core.model_executor
                 .driver_worker.model_runner.model),
    ):
        try:
            model = fn()
            ok(name, type(model).__name__)
            break
        except Exception as e:
            fail(name, e)

    print("=== 2. collective_rpc ===")
    try:
        r = llm.collective_rpc(lambda w: type(w).__name__)
        ok("collective_rpc(lambda w: ...)", r)
    except Exception as e:
        fail("collective_rpc", e)

    print("=== 3. apply_model 的 quant_method 普查 ===")
    try:
        from vllm.model_executor.layers.linear import LinearBase

        def census(m):
            out = {}
            for n, mod in m.named_modules():
                if isinstance(mod, LinearBase):
                    k = type(getattr(mod, "quant_method", None)).__name__
                    out[k] = out.get(k, 0) + 1
            return out

        r = llm.apply_model(census)
        ok("apply_model(census)", r)
    except Exception as e:
        fail("apply_model", e)

    if model is not None:
        print("=== 4. 直接遍历 LinearBase（转换钩子要走的那条路）===")
        try:
            from vllm.model_executor.layers.linear import LinearBase
            lin = [(n, tuple(mod.weight.shape))
                   for n, mod in model.named_modules()
                   if isinstance(mod, LinearBase) and hasattr(mod, "weight")]
            ok(f"{len(lin)} 个 LinearBase", ", ".join(f"{n}{s}" for n, s in lin[:4]))
        except Exception as e:
            fail("named_modules 遍历", e)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
