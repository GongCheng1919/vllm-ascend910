"""Price a FUSED matmul+all-reduce against the split one, before writing any kernel.

WHY THIS EXISTS.  P6.5 D9 measured 3632 us/step of NON-OVERLAPPED communication at
TP=4, and D11 showed the collective is latency-bound (10 KB and 640 KB cost the
same), so the plan is a fused mid-group GEMM + all-reduce that ELIMINATES the
separate collective call rather than chunking it.  Writing that kernel is days of
AscendC work, and its whole value rests on one unmeasured number: **how much does
fusing actually save at decode sizes on this machine?**

CANN already ships `npu_mm_all_reduce_base`, a fused quantised-matmul + all-reduce.
Its W8A8 mode (`dequant_scale` + `pertoken_scale`) matches our W8A8 arm exactly, so
it prices the fusion for free -- no kernel needed.  It CANNOT express our mid-group
W4A8 (MSD two-plane split + per-group asymmetric zero points), which is why we
would still have to write our own; but the saving it shows is the saving ours could
hope for.

    arm A  npu_quant_matmul  +  dist.all_reduce      <- today's structure
    arm B  npu_mm_all_reduce_base                    <- fused

WHAT THIS SCRIPT CAN AND CANNOT SETTLE (D15).  Eager timings are host-bound here
(see the `npu-eager-bench-is-host-bound` note): in eager, arm A pays two dispatches
and arm B pays one, so the "saving" printed below is mostly python and **must not
be quoted as the value of fusing**.  It answers one narrower question -- does the
fused path run at all and does it return the right numbers.

The in-graph question (does fusing remove the 3632 us/step of non-overlapped
collective?) is NOT settled here: NPUGraph capture of `dist.all_reduce` hangs on
this box, so `--graph` is off by default.  Settle it at the ENGINE instead, where
vLLM's FULL_DECODE_ONLY capture already works:

    bench_engine_decode.py --arm bf16 --tp 4 --async-scheduling   # baseline
    VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1 ... same command        # collective gone

vllm-ascend's `MatmulAllreduceRowParallelOp` (ops/linear_op.py:414) is exactly the
"eliminate the collective" structure we would build, and the step-time delta is the
ceiling our own fused op could reach.  It only works on the BF16 arm -- its
`apply_impl` bypasses `quant_method` and feeds `layer.weight.t()` straight to the
kernel, which our patch has deleted -- but the collective it removes is the same
one on every arm.

D14 called MC2 "blocked on this machine" (`HcclAllocComResourceByTiling ret=4`).
That was wrong: the failure is a ONE-SHOT first-call comm-resource allocation that
succeeds on the second attempt and never recurs (0/50 in steady state).  `_mc2_warmup`
below absorbs it.

usage (4 ranks):  ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 python npu_ops/python/bench_fused_allreduce.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# NPUGraph capture of a plain `dist.all_reduce` HANGS on this box: all four ranks
# print "Waiting for pending NCCL work to finish before starting graph capture"
# and never come back (20+ min, killed).  So the graph column is OFF by default --
# and it is not worth fixing here, because vLLM already captures collectives
# correctly under FULL_DECODE_ONLY, which makes the ENGINE the better instrument
# for the in-graph question anyway (see the note at the top of the module).
GRAPH_MODE = "--graph" in sys.argv

# QwQ-32B row-parallel shapes, K already sharded by TP.
#   (name, K_rank, N)
SHAPES = [("o_proj  TP=4", 1280, 5120), ("down_proj TP=4", 6912, 5120)]
MS = [1, 16, 64]
WARMUP, ITERS = 20, 100
GRAPH_REPS = 8          # ops per captured graph, so replay overhead is amortised


def _mc2_warmup(hcom: str, dev: str, world: int) -> bool:
    """Absorb the one-shot MC2 comm-resource failure (D15).

    The first `npu_mm_all_reduce_base` on a fresh communicator fails on EVERY rank
    with `HcclAllocComResourceByTiling ret = 4`; the second succeeds and the steady
    state is clean (0/50).  Retried in lockstep so a rank that happens to succeed
    first does not run ahead of the others.
    """
    import torch_npu

    x = torch.randint(-8, 8, (1, 256), dtype=torch.int8, device=dev)
    w = torch.randint(-8, 8, (256, 256), dtype=torch.int8, device=dev)
    ws = torch.ones(256, device=dev, dtype=torch.float32)
    pts = torch.ones(1, device=dev, dtype=torch.float32)
    out = _lockstep(lambda: torch_npu.npu_mm_all_reduce_base(
        x, w, hcom, dequant_scale=ws, pertoken_scale=pts), dev, world, tries=5)
    return out is not None


def _lockstep(fn, dev: str, world: int, tries: int = 4):
    """Retry an MC2 call until it sticks, WITHOUT any other collective in between.

    Three protocols were tried here; only this one works, and the two failures are
    worth keeping because they look like different bugs and are the same one:

      1. per-rank `try/except: continue`  -> the failing ranks skip to the next
         shape while their peers enter the timing loop.  Deadlock.
      2. attempt, then `all_reduce` a success flag, then retry together -> every
         shape reports "unavailable", and even the warm-up fails all 5 tries.
         Inserting a plain collective BETWEEN two MC2 attempts breaks the retry:
         MC2's `HcclAllocComResourceByTiling` does not survive it.
      3. (this one) attempt, sync, attempt again -- nothing else on the comm.
         Attempt 1 fails on every rank, attempt 2 succeeds on every rank, so the
         ranks stay in step without needing to be told.

    The "sporadic, per-rank" look the failure had in the first probes was an
    artefact of exactly the barriers those probes used to observe it.
    """
    for _ in range(tries):
        try:
            out = fn()
            torch.npu.synchronize()
            return out
        except Exception:                                     # noqa: BLE001
            continue
    return None


def timed(fn, sync) -> float:
    for _ in range(WARMUP):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync()
    return (time.perf_counter() - t0) / ITERS * 1e6


def timed_graph(fn, iters: int = 50) -> float:
    """Per-op us with GRAPH_REPS copies captured into one NPUGraph and replayed.

    This is the engine's structure (FULL_DECODE_ONLY captures the whole decode
    step), so it is the only column that can be compared against the 113 us/layer
    of in-graph collective that D9/D11 measured.
    """
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(5):
            fn()
    torch.npu.current_stream().wait_stream(s)
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(GRAPH_REPS):
            fn()
    torch.npu.synchronize()
    for _ in range(3):
        g.replay()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters / GRAPH_REPS * 1e6


def worker(rank: int, world: int) -> None:
    import torch_npu
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    pg = dist.distributed_c10d._get_default_group()
    try:
        hcom = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    except AttributeError:
        hcom = pg.get_hccl_comm_name(rank)
    dev = f"npu:{rank}"
    sync = torch.npu.synchronize

    mc2_ok = _mc2_warmup(hcom, dev, world)
    if rank == 0:
        print(f"world={world} hcom={hcom} mc2_warmup={'ok' if mc2_ok else 'FAILED'}", flush=True)
        print(f"  {'shape':>16} {'M':>4} | {'eager split':>11} {'eager fused':>11} "
              f"| {'GRAPH split':>11} {'GRAPH fused':>11} {'saved':>7} {'ratio':>6}", flush=True)

    for name, K, N in SHAPES:
        w = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
        ws = (torch.rand(N, device=dev) * 0.002 + 0.0005).to(torch.float32)
        for M in MS:
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            qx, pts = torch_npu.npu_dynamic_quant(x)

            # fp16 on both arms: npu_mm_all_reduce_base returns fp16 for the int8
            # path, and comparing across dtypes would compare two different ops.
            def split():
                y = torch_npu.npu_quant_matmul(qx, w, ws, pertoken_scale=pts,
                                               output_dtype=torch.float16)
                dist.all_reduce(y)
                return y

            def fused():
                return torch_npu.npu_mm_all_reduce_base(
                    qx, w, hcom, dequant_scale=ws, pertoken_scale=pts)

            # correctness first: a fused op that returns the wrong thing is not a
            # measurement, and the two paths must agree before their times mean
            # anything (P6 D8: prove the path did the work).
            #
            # Two things this loop must get right, both learned the hard way:
            #   * MC2's ret=4 can still fire on the first call for a NEW tiling
            #     even after the warm-up, so retry before giving up (D15);
            #   * the give-up decision must be COLLECTIVE.  A per-rank `continue`
            #     on a per-rank exception desyncs the ranks -- the failing ones
            #     move to the next M while the others enter the timing loop, and
            #     the next collective deadlocks.  That is what hung this bench.
            b = _lockstep(fused, dev, world)
            if b is None:
                if rank == 0:
                    print(f"  {name:>16} {M:>4}   fused path unavailable after "
                          f"retries -- skipped", flush=True)
                continue
            a = split()
            err = (a.float() - b.float()).abs().max().item()
            mag = a.float().abs().max().item()
            ok = err <= 2e-2 * max(mag, 1e-6)
            if not ok and rank == 0:
                print(f"  {name:>16} {M:>4}   MISMATCH max_err={err:.3e} "
                      f"(|y|max={mag:.3e}) -- times below are NOT comparable", flush=True)

            t_split = timed(split, sync)
            t_fused = timed(fused, sync)
            if not GRAPH_MODE:
                gtxt = "  (graph off; pass --graph, and see the note above)"
            else:
                try:
                    g_split = timed_graph(split)
                    g_fused = timed_graph(fused)
                    gtxt = (f"{g_split:>11.2f} {g_fused:>11.2f} "
                            f"{g_split - g_fused:>7.2f} {g_fused / g_split:>6.3f}")
                except Exception as e:                        # noqa: BLE001
                    gtxt = f"  graph capture failed: {type(e).__name__}"
            if rank == 0:
                print(f"  {name:>16} {M:>4} | {t_split:>11.2f} {t_fused:>11.2f} "
                      f"| {gtxt}{'' if ok else '  (MISMATCH)'}", flush=True)
    dist.destroy_process_group()


def main() -> int:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    world = len(vis.split(",")) if vis else 1
    if world < 2:
        print("need >=2 devices in ASCEND_RT_VISIBLE_DEVICES")
        return 1
    mp.start_processes(worker, args=(world,), nprocs=world, join=True,
                       start_method="spawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
