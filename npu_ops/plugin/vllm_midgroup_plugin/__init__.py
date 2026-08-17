"""vLLM general plugin that installs the mid-group W4A8 linear patch.

Why this exists: `bench_engine_decode.py` used to monkeypatch
`AscendUnquantizedLinearMethod` in the parent process, which is enough for
TP=PP=1 because vLLM v1 with VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the model
in-process.  With TP>1 or PP>1 the executor SPAWNS worker processes; they import
vLLM fresh and never see the parent's patch, so the run silently degrades to
BF16 (observed 2026-08-15: PP4 came up fine and reported converted=0).

`vllm.general_plugins` entry points are loaded by every vLLM process, worker
processes included, before the model is built -- which is exactly the hook the
patch needs.

Activated only when MIDGROUP_ARM is set, so installing this package cannot
change the behaviour of an unrelated vLLM run.
"""
import os
import sys


def register() -> None:
    arm = os.environ.get("MIDGROUP_ARM", "").strip()
    if not arm:
        return
    ops_path = os.environ.get("MIDGROUP_OPS_PATH", "").strip()
    if ops_path and ops_path not in sys.path:
        sys.path.insert(0, ops_path)
    try:
        import vllm_engine_patch
    except Exception as e:                       # noqa: BLE001
        print(f"[midgroup-plugin] FAILED to import vllm_engine_patch: {e!r}",
              flush=True)
        raise
    vllm_engine_patch.patch_unquantized_linear(arm=arm)
    print(f"[midgroup-plugin] pid={os.getpid()} armed arm={arm}", flush=True)
