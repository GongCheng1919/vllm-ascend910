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
    # Loads an ALREADY-INT4 checkpoint (export_w4a8_checkpoint.py) through our own
    # QuantizationConfig, so vLLM allocates int4-shaped parameters during model
    # construction.  Same kernels and bitwise-identical weights as "w4a8"
    # (test_w4a8_ckpt_seam.py), but peak memory is the int4 footprint instead of
    # the BF16 one -- which is what makes single-card 64 layers possible at all.
    "w4a8-ckpt":   ("models/QwQ-32B-W4A8-MG", "midgroup_w4a8", None),
}

# vLLM v1 runs EngineCore in a child process by default; the in-process engine is
# what lets this script's monkeypatch reach the model.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")
# tp*pp>1 forks workers (VLLM_WORKER_MULTIPROC_METHOD defaults to "fork"), and
# `MultiprocExecutor` calls `torch.set_num_threads(1)` in the child *only when
# OMP_NUM_THREADS is unset*.  This parent has already initialised OpenMP, so
# that post-fork call aborts the worker with "Invalid thread pool!" and the
# engine never comes up.  Setting it here takes that branch out of play.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _rank_conversion_stats(worker) -> dict:
    """Run inside each worker process to report what IT converted.

    Shipped to the workers by `collective_rpc`, so it must import its own
    dependencies -- the worker got `vllm_engine_patch` onto `sys.path` via the
    `vllm.general_plugins` entry point in `npu_ops/plugin`.
    """
    import os
    try:
        import vllm_engine_patch
        st = vllm_engine_patch.stats()
    except Exception as e:                            # noqa: BLE001
        return {"pid": os.getpid(), "arm": f"IMPORT FAILED: {e!r}",
                "converted": 0, "skipped": 0, "skipped_names": []}
    # The NAMES travel, not just the count: the gate that reads this has to
    # classify every skip against the declared policy, and a count cannot be
    # classified (P10 D10).
    return {"pid": os.getpid(), "arm": st.get("arm", "<unpatched>"),
            "converted": st.get("converted", 0), "skipped": st.get("skipped", 0),
            "skipped_names": list(st.get("skipped_names", []))}


def _quant_methods(model) -> dict:
    """Count each Linear's quant_method class.  Runs per rank via apply_model.

    This is the ONE evidence line that works for all five arms.  The
    `converted=` counters only exist for the two arms that go through our
    patch; the `-native` arms are quantised by vllm-ascend from the vendor
    checkpoint, so without this a silently-BF16 native arm would look exactly
    like a working one -- the same failure mode as P6 D8, just one layer over.
    """
    from collections import Counter
    from vllm.model_executor.layers.linear import LinearBase
    c: Counter = Counter()
    for _, m in model.named_modules():
        if isinstance(m, LinearBase):
            c[type(getattr(m, "quant_method", None)).__name__] += 1
    return dict(c)


def _audit_skips(names, policy: str, arm: str, who: str) -> bool:
    """Classify every skipped layer against the DECLARED expected-skip set.

    Returns True only if the set of skips is exactly what the caller said it
    would be.  The count is never enough on its own: `skipped=39` on Qwen3.8 is
    the by-design set, and `skipped=39` with one vision layer swapped for a
    `down_proj` is a coverage hole that would report a partly-BF16 model under
    the arm's name.  Same integer, opposite conclusions (P10 D10).
    """
    import vllm_engine_patch
    try:
        tally, unexpected = vllm_engine_patch.classify_skips(names, policy)
    except KeyError as e:                                  # noqa: BLE001
        print(f"[bench] ERROR: {e}")
        return False
    if unexpected:
        print(f"[bench] ERROR: {who} skipped {len(unexpected)} linears that the "
              f"expected-skip set `{policy}` does not cover -- they stayed BF16, "
              f"so this is a MIXED arm, not {arm}")
        for nm in unexpected:
            print(f"[bench]   UNEXPECTED skip: {nm}")
        return False
    print(f"[bench] {who}: all {len(names)} skips are inside the declared set "
          f"`{policy}`.  This run is `{arm} EXCEPT the following, which stay "
          f"BF16` -- carry that clause with the number:")
    for (a, b), got in tally.items():
        reason = next(r for x, y, r in
                      vllm_engine_patch.EXPECTED_SKIPS[policy]
                      if (x, y) == (a, b))
        ex = f"  e.g. {got[0]}" if got else ""
        print(f"[bench]   {len(got):3d}x  *{a}*{b}*{ex}")
        print(f"[bench]         why: {reason}")
    return True


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
    ap.add_argument("--tp", type=int, default=1,
                    help="Tensor parallel size.  WARNING at GK=1024: TP shards "
                         "the K of row-parallel layers, and o_proj K=5120=5*GK "
                         "/ down_proj K=27648=27*GK only stay group-aligned for "
                         "TP dividing 5 and 27 -- i.e. TP=1 only.  Any other TP "
                         "makes _supported() reject those two layers, which the "
                         "patch SILENTLY falls back to BF16 (34% of the weights). "
                         "The script refuses that below rather than report it as "
                         "a W4A8 number.")
    ap.add_argument("--pp", type=int, default=1,
                    help="Pipeline parallel size.  PP splits by LAYER, so every "
                         "rank keeps a full K and GK=1024 stays aligned -- this "
                         "is the parallelism that works unchanged.")
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[128],
                    help="one or more prompt lengths, swept inside ONE engine "
                         "(long-context sweeps otherwise pay a fresh engine "
                         "start per point).  max_model_len is sized from the "
                         "largest.")
    ap.add_argument("--prefill-probe", action="store_true",
                    help="also time prefill.  The normal `step_us` warms up on "
                         "the SAME prompts it then times, and prefix caching is "
                         "on, so the timed pass reuses the prefix KV and pays "
                         "almost no prefill -- deliberately, since that isolates "
                         "decode with a long KV.  This flag adds a separate "
                         "max_tokens=1 pass over UNSEEN prompts, which is where "
                         "the prefill (large-M GEMM) cost actually shows up.")
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--eager", action="store_true",
                    help="disable ACL graph (host-bound; see P4 D7)")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="cap concurrent sequences (0 = vLLM's default 256).  "
                         "Load-bearing on Qwen3.8: 48 of its 64 layers are Gated "
                         "DeltaNet and each running sequence needs one Mamba "
                         "state block, so the weight footprint sets a HARD "
                         "concurrency ceiling.  At 64 layers BF16 leaves room "
                         "for only 41 blocks and the engine refuses to start "
                         "with `max_num_seqs (256) exceeds available Mamba cache "
                         "blocks (41)`.  Lowering this is how that arm produces "
                         "a number at all -- and the ceiling itself is a result, "
                         "so record it next to the throughput.")
    ap.add_argument("--allow-misaligned-tp", action="store_true",
                    help="Run a group-misaligned TP anyway, knowingly measuring "
                         "a mixed W4A8/BF16 model.")
    ap.add_argument("--expect-skip", default="none",
                    help="name of a declared expected-skip set in "
                         "vllm_engine_patch.EXPECTED_SKIPS (`none` | `qwen38`). "
                         "A skip leaves that linear in BF16, so by default ANY "
                         "skip fails the run.  Qwen3.8 skips a fixed set BY "
                         "DESIGN (vision `mlp.linear_fc1`, the GDN `in_proj_ba` "
                         "gate, the GDN `conv1d`; P10 D5), and without this flag "
                         "that arm can never produce a number.  This does NOT "
                         "relax the gate to a count: every skipped layer must "
                         "match a DECLARED pattern, one unclassified skip still "
                         "fails, and the declaration is printed so the result "
                         "reads `W4A8 except <these>` rather than `W4A8`.")
    ap.add_argument("--real-weights", action="store_true",
                    help="load the checkpoint for real instead of load_format=dummy. "
                         "Every prior measurement used dummy; keep it off for "
                         "comparability unless you are checking generation.")
    ap.add_argument("--additional-config", default="",
                    help="JSON passed straight to vLLM's `additional_config=`, "
                         "which is where vllm-ascend keeps its own knobs.  Needed "
                         "for `{\"ascend_compilation_config\": "
                         "{\"enable_npugraph_ex\": false}}`: on 0.23.0 the "
                         "vendor W4A8 path cannot be compiled by torch_npu's "
                         "npugraph_ex backend (`'CompilerConfig' object has no "
                         "attribute 'experimental_config'`), while bf16 and "
                         "W8A8_DYNAMIC compile fine.")
    ap.add_argument("--out", default="")
    ap.add_argument("--profile-dir", default="",
                    help="capture a torch_npu profile of ONE timed generate")
    ap.add_argument("--async-scheduling", action="store_true",
                    help="overlap the scheduler / input-prep host work with the "
                         "PREVIOUS step's forward.  P6.5 measured 4.05 ms of the "
                         "10.8 ms step as pure device idle at L=16/batch=1, 91% of "
                         "it in five discrete host stalls BEFORE the model graph "
                         "(the biggest 1.9 ms, in front of an input-buffer Fill), "
                         "so this is the matching fix.  Not supported with PP.")
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

    # The group-alignment gate that used to live here is GONE, because the
    # constraint it enforced is gone: the op takes any K (npu_ops/kernel/mg_kgeom.h),
    # so `K_rank = K/TP` no longer has to be a multiple of GK and o_proj/down_proj
    # are converted at every TP.  It is NOT replaced by nothing -- the thing it was
    # really protecting against (an arm that quietly converts only part of the
    # model) is now caught for ALL FIVE arms, after the engine is up, by the
    # per-rank `converted=/skipped=` query and the `quant_methods` census below.
    # That is strictly stronger: it checks what actually happened instead of
    # predicting it from config.json, and it covers the two native arms too, which
    # this gate never could (P6 D8, D12).
    if args.allow_misaligned_tp:
        print("[bench] note: --allow-misaligned-tp is a no-op now; K alignment is "
              "no longer a constraint.")

    multi_rank = args.tp * args.pp > 1
    if patch_arm:
        # TP/PP > 1 spawns worker PROCESSES; a parent-process monkeypatch never
        # reaches them (PP4 came up clean with converted=0 on 2026-08-15).  The
        # vllm.general_plugins entry point installed by npu_ops/plugin runs in
        # every vLLM process, so hand the arm over through the environment and
        # let each worker arm itself.
        os.environ["MIDGROUP_ARM"] = patch_arm
        os.environ["MIDGROUP_OPS_PATH"] = os.path.dirname(os.path.abspath(__file__))
        import vllm_engine_patch
        vllm_engine_patch.patch_unquantized_linear(arm=patch_arm)

    from vllm import LLM, SamplingParams

    overrides = {}
    if args.layers:
        overrides["num_hidden_layers"] = args.layers

    kw = {}
    if args.additional_config:
        import json as _json
        kw["additional_config"] = _json.loads(args.additional_config)
    if args.max_num_seqs:
        kw["max_num_seqs"] = args.max_num_seqs
    if args.async_scheduling:
        if args.pp > 1:
            print("[bench] REFUSING: --async-scheduling is not supported with PP")
            return 2
        kw["async_scheduling"] = True
    if args.cudagraph_mode:
        kw["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    if quantization == "midgroup_w4a8":
        # Ours, not vllm-ascend's: registers the config class that config.json's
        # `quantization_config.quant_method` names.  Must happen before LLM().
        import midgroup_quant
        midgroup_quant.register()
    if quantization:
        # Neither vendor config.json has a `quantization_config`; the method is
        # declared in quant_model_description.json and only reached by asking
        # for vllm-ascend's quant config explicitly.
        kw["quantization"] = quantization

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        load_format=("auto" if args.real_weights else "dummy"),
        hf_overrides=overrides or None,
        max_model_len=max(args.prompt_len) + args.out_len + 16,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.eager,
        dtype="bfloat16",
        trust_remote_code=True,
        **kw,
    )
    t_load = time.perf_counter() - t0
    # What the ENGINE actually resolved, not what we asked for.  vllm 0.23.0
    # turns async scheduling ON BY DEFAULT ("Asynchronous scheduling is enabled"
    # in every arm's startup log), whereas P6.5 measured it as an opt-in worth
    # 1.23-1.29x.  Writing our own flag into the CSV would label every row of
    # this phase `async_sched=0` while the engine ran with it on -- the same
    # class of error as trusting a log line for conversion coverage (P6 D8).
    try:
        _async = bool(llm.llm_engine.vllm_config.scheduler_config.async_scheduling)
    except Exception:                                     # noqa: BLE001
        _async = bool(args.async_scheduling)
    print(f"[bench] engine up in {t_load:.1f}s  arm={args.arm} model={model} "
          f"layers={args.layers or 64} tp={args.tp} pp={args.pp} "
          f"graph={'off' if args.eager else 'on'}"
          f"{' mode=' + args.cudagraph_mode if args.cudagraph_mode else ''}"
          f"{' async-sched' if args.async_scheduling else ''}")

    if patch_arm:
        st = vllm_engine_patch.stats()
        print(f"[bench] converted={st.get('converted')} skipped={st.get('skipped')} "
              f"quantise_time={st.get('t', 0):.0f}s")
        for n in st.get("names", []):
            print(f"[bench]   e.g. {n}")
        if multi_rank:
            # The model lives in the worker processes, so the parent's counters
            # stay at zero and say nothing.  Do NOT fall back to reading the
            # workers' log lines: `[midgroup] N converted` only prints every 64
            # conversions, and a short `--layers` run never reaches 64, so its
            # absence is not evidence.  That is precisely how PP4 was first
            # measured as a BF16 run (2026-08-15).  Ask every rank directly.
            per_rank = llm.collective_rpc(_rank_conversion_stats)
            for r, s in enumerate(per_rank):
                print(f"[bench] rank {r}: pid={s['pid']} arm={s['arm']} "
                      f"converted={s['converted']} skipped={s['skipped']}")
            dead = [r for r, s in enumerate(per_rank) if not s["converted"]]
            if dead:
                print(f"[bench] ERROR: ranks {dead} converted nothing -- those "
                      "ranks' layers ran in BF16, so this is not a "
                      f"{patch_arm} number")
                return 2
            # `skipped` is now load-bearing for BOTH arms.  A skip leaves that
            # linear in BF16, which is a PARTIAL arm reported under a whole arm's
            # name -- the D0 failure that made every TP>1 W8A8 baseline a 66/34
            # mixture.  Since K is unconstrained there is no legitimate reason to
            # skip a QwQ-32B projection, so any skip is a bug, not a fallback.
            for r, s in enumerate(per_rank):
                if s["skipped"] and not _audit_skips(
                        s["skipped_names"], args.expect_skip, patch_arm,
                        f"rank {r}"):
                    return 2
        elif not st.get("converted"):
            print("[bench] ERROR: nothing converted -- the arm is a BF16 run")
            return 2
        elif st.get("skipped"):
            if not _audit_skips(st.get("skipped_names", []), args.expect_skip,
                                patch_arm, "rank 0"):
                return 2

    # Works for every arm, including the two `-native` ones that our patch never
    # touches.  An arm whose linears all report UnquantizedLinearMethod is a
    # BF16 run wearing the arm's name.
    try:
        for r, qm in enumerate(llm.apply_model(_quant_methods)):
            print(f"[bench] rank {r} quant_methods: {qm}")
    except Exception as e:                                # noqa: BLE001
        print(f"[bench] WARNING: could not read quant_methods: {e!r}")

    rows = []
    n_layers = args.layers or 64
    for plen in args.prompt_len:
      for b in args.batch:
        # Distinct prompts so the scheduler cannot dedupe them.  They ARE
        # prefix-cached between the warmup and the timed pass, on purpose: it
        # makes `step_us` a decode measurement even at plen=8192.  Prefill is
        # measured separately by --prefill-probe.
        prompts = [f"{plen}_{i} " + "word " * (plen - 2) for i in range(b)]
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

        prefill_ms = ""
        if args.prefill_probe:
            # UNSEEN prompts (different leading token => different prefix hash),
            # one output token: this pass pays the prefill the timed pass above
            # deliberately skips.
            fresh = [f"probe{plen}_{b}_{i} " + "word " * (plen - 2)
                     for i in range(b)]
            sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            t1 = time.perf_counter()
            llm.generate(fresh, sp1)
            prefill_ms = f"{(time.perf_counter() - t1) * 1e3:.2f}"

        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        per_step = dt / args.out_len * 1e6              # us per decode step
        per_layer = per_step / n_layers
        print(f"[bench] plen={plen:6d} batch={b:4d}  total={dt * 1e3:8.1f} ms  "
              f"gen={gen} tok  step={per_step:8.1f} us  "
              f"per-layer={per_layer:7.1f} us  "
              f"throughput={gen / dt:7.1f} tok/s"
              + (f"  prefill={prefill_ms} ms" if prefill_ms else ""))
        rows.append((plen, b, dt, gen, per_step, per_layer, prefill_ms))

    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a") as f:
            if new:
                # `cudagraph_mode` is NOT optional bookkeeping: the same kernels
                # measure 1.41x under the vllm-ascend default PIECEWISE and
                # 2.78x under FULL_DECODE_ONLY.  A row without it is unreadable.
                # `async_sched` is last so readers of the older CSVs keep
                # working; it is not optional bookkeeping either -- it is worth
                # 1.23-1.29x on its own (P6.5 D12), so a row without it cannot
                # be compared against one with it.
                f.write("arm,layers,tp,pp,graph,cudagraph_mode,batch,"
                        "prompt_len,out_len,total_s,gen_tok,step_us,"
                        "per_layer_us,throughput_tok_s,prefill_ms,"
                        "async_sched,expect_skip\n")
            mode = args.cudagraph_mode or "PIECEWISE(default)"
            for plen, b, dt, gen, ps, pl, pf in rows:
                f.write(f"{args.arm},{n_layers},{args.tp},{args.pp},"
                        f"{'eager' if args.eager else 'aclgraph'},{mode},{b},"
                        f"{plen},{args.out_len},"
                        f"{dt:.4f},{gen},{ps:.2f},{pl:.3f},{gen / dt:.2f},{pf},"
                        f"{int(_async)},{args.expect_skip}\n")
        print(f"[bench] appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
