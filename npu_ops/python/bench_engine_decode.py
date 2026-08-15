#!/usr/bin/env python3
"""End-to-end decode latency in the REAL vLLM engine, across five arms.

P4 step 3.  Every number before this came from a hand-built layer; this runs the
kernels under the actual engine (scheduler, paged attention, ACL graph).

    bf16          stock BF16                                  models/QwQ-32B
    w8a8-native   vllm-ascend's own W8A8 path                 models/QwQ-32B-W8A8
    w4a8-native   vllm-ascend's own W4A8 path                 models/QwQ-32B-W4A8-Random
    w8a8          OURS: per-channel INT8 + per-token dynamic  models/QwQ-32B
    w4a8          OURS: mid-group W4A8 (P3 kernels)           models/QwQ-32B

The two `-native` arms load the VENDOR checkpoints through vllm-ascend's
quantisation path (`quantization="ascend"`, driven by `quant_model_description.json`
-- neither config.json carries a `quantization_config`).  They answer "is the
shipping product faster", which is a DIFFERENT question from the one the
`w8a8`/`w4a8` pair answers.  That pair is converted from the same BF16 weights at
the same hook over the same 64 linears, so its ratio isolates the GEMM.  Read
each pair against its own question and do not mix them.

`w4a8-native` in particular is NOT a real W4A8: it is weight-only, dequantising
to bf16 before the matmul, so the activation's int8 never reaches the cube
(P4 D3).  Expect it to lose to BF16, and do not read that as "W4A8 is slow".

Defaults are a SMOKE configuration: QwQ-32B geometry cut to a few layers with
`--layers`, and `load_format=dummy` so no 65 GB of weights is read from disk.
**Per-layer SLOPE is what transfers to the full model, absolute time is not** --
take two layer counts and difference them, which also removes the ~4.1 ms/step
fixed cost (lm_head alone is 1.56 GB of bf16 per step and is NOT converted by
any arm).  End-to-end ratios are therefore lower than slope ratios: at 16 layers
ours measured 1.19x while the slope said 1.32x.  Random weights -- output text
is garbage by construction, and P2's accuracy gate has not passed anyway (D1).

usage:
  ASCEND_RT_VISIBLE_DEVICES=1 .venv/bin/python npu_ops/python/bench_engine_decode.py \
      --arm w4a8 --layers 8 --batch 1 --cudagraph-mode FULL_DECODE_ONLY
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# arm -> (default model, vLLM `quantization=`, patch arm passed to
# vllm_engine_patch).  None means "leave it alone".
ARMS = {
    "bf16":        ("models/QwQ-32B",             None,     None),
    "w8a8-native": ("models/QwQ-32B-W8A8",        "ascend", None),
    "w4a8-native": ("models/QwQ-32B-W4A8-Random", "ascend", None),
    "w8a8":        ("models/QwQ-32B",             None,     "w8a8"),
    "w4a8":        ("models/QwQ-32B",             None,     "w4a8"),
}

# vLLM v1 runs EngineCore in a child process by default; the in-process engine is
# what lets this script's monkeypatch reach the model.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default="w4a8",
                    help="see the module docstring; the two `-native` arms load "
                         "vendor checkpoints, the w8a8/w4a8 pair converts from "
                         "the same BF16 weights so its ratio isolates the GEMM")
    ap.add_argument("--model", default=None,
                    help="override the arm's default checkpoint")
    ap.add_argument("--layers", type=int, default=8,
                    help="override num_hidden_layers; 0 = keep the real 64")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--eager", action="store_true",
                    help="disable ACL graph (host-bound; see P4 D7)")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", default="")
    ap.add_argument("--profile-dir", default="",
                    help="capture a torch_npu profile of ONE timed generate")
    ap.add_argument("--cudagraph-mode", default="",
                    help="PIECEWISE (vllm-ascend default) | FULL_DECODE_ONLY | "
                         "FULL | FULL_AND_PIECEWISE.  PIECEWISE splits the graph "
                         "at every attention, leaving python glue between the "
                         "pieces -- ~390 us/layer of device idle at batch=1 "
                         "(P4_E2E.md §0''.5).  This switch measures what "
                         "collapsing those boundaries is worth.")
    args = ap.parse_args()
    if args.profile_dir:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profile_dir

    default_model, quantization, patch_arm = ARMS[args.arm]
    model = args.model or default_model

    if patch_arm:
        import vllm_engine_patch
        vllm_engine_patch.patch_unquantized_linear(arm=patch_arm)

    from vllm import LLM, SamplingParams

    overrides = {}
    if args.layers:
        overrides["num_hidden_layers"] = args.layers

    kw = {}
    if args.cudagraph_mode:
        kw["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    if quantization:
        # Neither vendor config.json has a `quantization_config`; the method is
        # declared in quant_model_description.json and only reached by asking
        # for vllm-ascend's quant config explicitly.
        kw["quantization"] = quantization

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tp,
        load_format="dummy",
        hf_overrides=overrides or None,
        max_model_len=args.prompt_len + args.out_len + 16,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.eager,
        dtype="bfloat16",
        trust_remote_code=True,
        **kw,
    )
    t_load = time.perf_counter() - t0
    print(f"[bench] engine up in {t_load:.1f}s  arm={args.arm} model={model} "
          f"layers={args.layers or 64} tp={args.tp} "
          f"graph={'off' if args.eager else 'on'}"
          f"{' mode=' + args.cudagraph_mode if args.cudagraph_mode else ''}")

    if patch_arm:
        st = vllm_engine_patch.stats()
        print(f"[bench] converted={st.get('converted')} skipped={st.get('skipped')} "
              f"quantise_time={st.get('t', 0):.0f}s")
        for n in st.get("names", []):
            print(f"[bench]   e.g. {n}")
        if not st.get("converted"):
            print("[bench] ERROR: nothing converted -- the arm is a BF16 run")
            return 2

    rows = []
    n_layers = args.layers or 64
    for b in args.batch:
        # Distinct prompts so the scheduler cannot dedupe or prefix-cache them.
        prompts = [f"{i} " + "word " * (args.prompt_len - 2) for i in range(b)]
        sp = SamplingParams(temperature=0.0, max_tokens=args.out_len,
                            ignore_eos=True)
        llm.generate(prompts, sp)                      # warm up + capture
        if args.profile_dir:
            llm.start_profile()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        if args.profile_dir:
            llm.stop_profile()

        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        per_step = dt / args.out_len * 1e6              # us per decode step
        per_layer = per_step / n_layers
        print(f"[bench] batch={b:4d}  total={dt * 1e3:8.1f} ms  "
              f"gen={gen} tok  step={per_step:8.1f} us  "
              f"per-layer={per_layer:7.1f} us  "
              f"throughput={gen / dt:7.1f} tok/s")
        rows.append((b, dt, gen, per_step, per_layer))

    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a") as f:
            if new:
                # `cudagraph_mode` is NOT optional bookkeeping: the same kernels
                # measure 1.41x under the vllm-ascend default PIECEWISE and
                # 2.78x under FULL_DECODE_ONLY.  A row without it is unreadable.
                f.write("arm,layers,tp,graph,cudagraph_mode,batch,total_s,"
                        "gen_tok,step_us,per_layer_us\n")
            mode = args.cudagraph_mode or "PIECEWISE(default)"
            for b, dt, gen, ps, pl in rows:
                f.write(f"{args.arm},{n_layers},{args.tp},"
                        f"{'eager' if args.eager else 'aclgraph'},{mode},{b},"
                        f"{dt:.4f},{gen},{ps:.2f},{pl:.3f}\n")
        print(f"[bench] appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
