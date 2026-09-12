"""Generate with the exported mid-group W4A8 checkpoint, REAL weights.

Every performance number in this project so far was measured with
`load_format=dummy`, i.e. random weights: correct shapes, meaningless outputs.
This is the first end-to-end run where the weights are the real QwQ-32B ones
pushed through our quantiser, so it is the first evidence that the whole chain
-- exporter -> checkpoint -> vLLM loader -> AscendC kernel -- produces a working
model and not just a fast one.

It is NOT an accuracy evaluation.  P2's task gate (GLUE -4.5 points) is still
open; coherent text here does not close it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/QwQ-32B-W4A8-MG")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    args = ap.parse_args()

    import midgroup_quant
    midgroup_quant.register()
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(model=args.model, quantization="midgroup_w4a8",
              tensor_parallel_size=1, gpu_memory_utilization=args.gpu_util,
              max_model_len=2048, trust_remote_code=True,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})
    print(f"[gen] engine up in {time.perf_counter()-t0:.1f}s", flush=True)

    try:
        for r, qm in enumerate(llm.apply_model(
                lambda m: {type(getattr(l, "quant_method", None)).__name__:
                           1 for _, l in m.named_modules()
                           if hasattr(l, "quant_method")})):
            print(f"[gen] rank {r} quant methods present: {sorted(qm)}")
    except Exception as e:                       # noqa: BLE001
        print(f"[gen] could not census quant methods: {e!r}")

    prompts = [
        "简要解释什么是张量并行(tensor parallelism)，以及它和流水并行的区别。",
        "The capital of France is",
        "Write a Python function that returns the n-th Fibonacci number.",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)
    for o in outs:
        print("\n" + "=" * 78)
        print("PROMPT :", o.prompt[:110].replace("\n", " "))
        print("OUTPUT :", o.outputs[0].text.strip()[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
