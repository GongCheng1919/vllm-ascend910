"""Price the all-reduce that row-parallel TP pays, on its own.

WHY THIS EXISTS.  The engine profile (P6.5) measured 3632 us/step of NON-OVERLAPPED
communication at TP=4 -- 32 collectives of a [1, 5120] bf16 tensor, i.e. ~113 us for
10 KB.  That is an order of magnitude above what intra-node HCCS should cost, so
before anyone builds a fused GEMM+all-reduce to hide it, two things have to be
separated:

  * is it LATENCY (a fixed per-call cost) or BANDWIDTH?  Sweeping the payload
    answers that: a flat curve is latency, a linear one is bandwidth.  Only the
    latency case makes "fewer, bigger collectives" a lever, and only the bandwidth
    case makes overlap the whole answer.
  * is it the DEVICE SET?  A ring that straddles a topology group pays a slower
    path.  Device 0 is wedged on this box, so TP=4 runs have been using logic
    devices 1,2,3,4 -- which is exactly the kind of off-by-one that could cross a
    group boundary.  Compare against 2,3,4,5.

usage (4 ranks):
    ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 python npu_ops/python/bench_allreduce.py
    ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 python npu_ops/python/bench_allreduce.py
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# QwQ-32B decode: the row-parallel all-reduce is [batch, hidden] bf16.
# 5120 elements = 10 KB is the batch=1 case the profile measured; batch=64 is
# 640 KB.  The last two sizes are past the latency knee, where BANDWIDTH -- and
# therefore the device set -- starts to matter: this box has two groups of four
# boards sharing two bandwidth domains, so a BALANCED cross-group ring (2+2) can
# use both while a lopsided one (3+1) or a within-group one cannot.
SIZES = [5120, 5120 * 4, 5120 * 16, 5120 * 64, 5120 * 256, 5120 * 1024]
WARMUP, ITERS = 50, 300


def worker(rank: int, world: int, out) -> None:
    import torch_npu  # noqa: F401
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)

    res = []
    for n in SIZES:
        x = torch.ones(n, dtype=torch.bfloat16, device=f"npu:{rank}")
        for _ in range(WARMUP):
            dist.all_reduce(x)
        torch.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(x)
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) / ITERS * 1e6
        kb = n * 2 / 1024
        res.append((kb, us))
        if rank == 0:
            # 2*(world-1)/world * bytes is the ring all-reduce bus volume
            alg = n * 2 * 2 * (world - 1) / world / (us * 1e-6) / 1e9
            print(f"  {kb:>9.1f} KB  {us:>9.2f} us/call  {alg:>8.2f} GB/s (ring bus)",
                  flush=True)
    if rank == 0:
        out.put(res)
    dist.destroy_process_group()


def main() -> int:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    world = len(vis.split(",")) if vis else 1
    if world < 2:
        print("need at least 2 devices in ASCEND_RT_VISIBLE_DEVICES")
        return 1
    print(f"all_reduce over {world} ranks, ASCEND_RT_VISIBLE_DEVICES={vis}")
    print(f"  {'payload':>9}      {'latency':>9}       {'bus bw':>8}")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(worker, args=(world, q), nprocs=world, join=True,
                       start_method="spawn")
    res = q.get()
    lat = res[0][1]
    big = res[-1][1]
    print(f"\n  10 KB costs {lat:.1f} us; {res[-1][0]:.0f} KB costs {big:.1f} us "
          f"({big/lat:.1f}x for {res[-1][0]/res[0][0]:.0f}x the bytes)")
    print("  => " + ("LATENCY-bound at decode sizes: fewer/bigger collectives is a "
                     "real lever" if big / lat < 4 else
                     "BANDWIDTH-bound: only overlap helps"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
