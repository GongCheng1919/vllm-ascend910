"""Is FUSING the all-reduce into the GEMM worth the kernel surgery? (P6.5 D18)

D17 delivered a standalone all-reduce that costs 16.9 us where HCCL costs ~250.
The fused op in the P6.5 plan goes one step further and does the cross-rank sum in
the GEMM's epilogue.  Before touching a 63 KB kernel that carries 62 gates, this
measures what fusion could still win -- because the two candidate wins are very
different sizes:

    drop-in replacement   : HCCL (250 us) -> ours (17 us)      <- already in hand
    fusion on top of that : one launch + one staging copy of y  <- unknown, measured here

METHOD.  Three arms, each a full row-parallel decode layer (quant_a -> mid-group
W4A8 GEMM -> reduction), all timed the same way -- captured into a graph, replayed,
wall/replays -- so the arms are comparable to each other:

    1. gemm         : no reduction at all.  The floor.
    2. gemm + ours  : the drop-in.  (2) - (1) is the MARGINAL cost of our
                      all-reduce in situ, which is the entire budget fusion can
                      attack.
    3. gemm + hccl  : status quo.  CANNOT be captured here (`dist.all_reduce` under
                      NPUGraph hangs, D14), so it is timed eager and reported
                      separately -- never subtracted from a captured number.

WHAT FUSION CAN ACTUALLY REMOVE.  Not the barrier (every rank still waits for its
peers) and not the peer reads ((w-1)*n bytes have to cross the fabric either way).
Only two things: one kernel launch, and the round trip of y through GM -- today the
GEMM writes y and the all-reduce reads it back, whereas a fused epilogue would write
its tile straight into the staging slot.  So

    fusion_ceiling ~ 2n of local traffic + one launch

and `2n` is measured, not guessed: a world=1 call does exactly 4n of local traffic
(read y, write staging, read own staging, write out) and no peer traffic at all, so
half of it is the round trip fusion deletes.

usage: OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 \
       python npu_ops/python/bench_fused_layer.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "python"))

HIDDEN = 5120           # QwQ-32B
FFN = 27648
KEY_LEN = 256
SPIN_LIMIT = 200_000_000
GROUP = 1024

CAPTURE_CALLS = 10
GRAPH_REPLAYS = 20
EAGER_ITERS = 200
WARMUP = 10

# Row-parallel projections at TP=4: K is sharded, N is not, and the output needs an
# all-reduce.  These are the only two ops in the model that pay a collective.
LAYERS = [("o_proj  ", HIDDEN // 4, HIDDEN), ("down_proj", FFN // 4, HIDDEN)]
BATCHES = [1, 16, 64]


def _ag_int(v, rank, world, dev):
    t = torch.zeros(world, dtype=torch.int64, device=dev)
    t[rank] = v
    dist.all_reduce(t)
    return [int(x) for x in t.tolist()]


def _ag_key(key, rank, world, dev):
    buf = torch.zeros(world, KEY_LEN, dtype=torch.uint8, device=dev)
    raw = key.encode()
    buf[rank, : len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=dev)
    dist.all_reduce(buf)
    return [bytes(buf[r].tolist()).rstrip(b"\x00").decode() for r in range(world)]


def _ag_float(v, rank, world, dev):
    t = torch.zeros(world, dtype=torch.float32, device=dev)
    t[rank] = float(v)
    dist.all_reduce(t)
    return [float(x) for x in t.tolist()]


def _time_graph(fn, dev) -> float:
    """Capture CAPTURE_CALLS calls, replay, return us/call.  Inputs stay alive in
    the caller -- a captured graph replays against frozen ADDRESSES (D17)."""
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(CAPTURE_CALLS):
            fn()
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(GRAPH_REPLAYS):
        g.replay()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / (GRAPH_REPLAYS * CAPTURE_CALLS) * 1e6
    del g
    return us


def _time_eager(fn) -> float:
    torch.npu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(EAGER_ITERS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / EAGER_ITERS * 1e6


def worker(rank: int, world: int) -> None:
    import torch_npu  # noqa: F401
    import w4a8_ops

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29677")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    w4a8_ops.load()
    dev = f"npu:{rank}"
    torch.ops.npu.enable_peer_access([d for d in range(world) if d != rank])

    cap = max(m * HIDDEN for m in BATCHES)
    elems = torch.ops.npu.peer_staging_elems(cap)
    staging = torch.zeros(elems, device=dev, dtype=torch.float32)
    pids = _ag_int(torch.ops.npu.ipc_get_bare_tgid(), rank, world, dev)
    key = torch.ops.npu.ipc_export(staging,
                                   [p for i, p in enumerate(pids) if i != rank])
    keys = _ag_key(key, rank, world, dev)
    ptrs = [staging.data_ptr() if r == rank else torch.ops.npu.ipc_import(keys[r])
            for r in range(world)]
    dist.barrier()

    # ---- barrier floor: one 32 B block, no payload to hide behind -------------
    tiny = torch.ones(16, device=dev, dtype=torch.bfloat16)
    floor = _time_graph(
        lambda: torch.ops.npu.peer_allreduce(tiny, staging, ptrs, rank, cap,
                                             SPIN_LIMIT), dev)
    floor = max(_ag_float(floor, rank, world, dev))

    if rank == 0:
        print(f"world={world}  barrier floor (32 B) = {floor:.1f} us\n", flush=True)
        print(f"{'layer':<10} {'M':>4} {'K':>6} | {'gemm':>8} {'+ours':>8} "
              f"{'marginal':>9} {'local':>7} | {'eager gemm':>10} {'+hccl':>9} "
              f"{'+ours':>8} | {'fusion<=':>9}")
        print("-" * 110, flush=True)

    for name, K, N in LAYERS:
        # Random weights in kernel layout: timing does not depend on the values,
        # and the correctness gate below compares two REDUCTIONS of the same GEMM
        # output, so the weights only have to be identical across arms.
        G = w4a8_ops.num_groups(K, GROUP)
        kpad = w4a8_ops.k_pad_elems(K, GROUP)
        wq = torch.randint(-128, 127, (N, kpad // 2), device=dev, dtype=torch.int8)
        ws = torch.rand(G, N, device=dev, dtype=torch.bfloat16) * 0.01 + 0.001
        wk = torch.randint(-64, 64, (G, N), device=dev, dtype=torch.int32)
        wz = torch.randint(-8, 7, (G, N), device=dev).to(torch.bfloat16)

        for M in BATCHES:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.05

            def gemm():
                a_hi, a_lo, a_s, a_k = torch.ops.npu.midgroup_quant_a(x)
                return torch.ops.npu.midgroup_w4a8_gemm(a_hi, a_lo, a_s, a_k,
                                                        wq, ws, wk, wz, K)

            def gemm_ours():
                y = gemm()
                return torch.ops.npu.peer_allreduce(y[:M].reshape(-1), staging,
                                                    ptrs, rank, cap, SPIN_LIMIT)

            def gemm_hccl():
                y = gemm()[:M].reshape(-1).contiguous()
                dist.all_reduce(y)
                return y

            # ---- correctness: ours must equal an f32-accumulated reference ----
            # (not HCCL's bf16 output -- that would grade us on their rounding)
            y = gemm()[:M].reshape(-1)
            ref = y.detach().clone().float()
            dist.all_reduce(ref)
            ref = ref.to(torch.bfloat16)
            got = torch.ops.npu.peer_allreduce(y, staging, ptrs, rank, cap,
                                               SPIN_LIMIT)
            torch.npu.synchronize()
            exact = min(_ag_float(1.0 if torch.equal(got, ref) else 0.0,
                                  rank, world, dev)) > 0.5

            # world=1: same kernel, same local pipeline, zero peer traffic and no
            # waiting.  This isolates the local cost from the fabric cost.
            def ours_local():
                return torch.ops.npu.peer_allreduce(y, staging, [ptrs[rank]], 0,
                                                    cap, SPIN_LIMIT)

            t_local = max(_ag_float(_time_graph(ours_local, dev), rank, world, dev))
            t_gemm = max(_ag_float(_time_graph(gemm, dev), rank, world, dev))
            t_ours = max(_ag_float(_time_graph(gemm_ours, dev), rank, world, dev))
            e_gemm = max(_ag_float(_time_eager(gemm), rank, world, dev))
            e_hccl = max(_ag_float(_time_eager(gemm_hccl), rank, world, dev))
            e_ours = max(_ag_float(_time_eager(gemm_ours), rank, world, dev))

            marginal = t_ours - t_gemm
            # 4n of local traffic in t_local; fusion deletes the y round trip (2n)
            # plus one launch (~4 us on this box, P4).
            ceiling = t_local / 2.0 + 4.0
            if rank == 0:
                mark = "" if exact else "  <-- NOT EXACT"
                print(f"{name:<10} {M:>4} {K:>6} | {t_gemm:>7.1f}u {t_ours:>7.1f}u "
                      f"{marginal:>8.1f}u {t_local:>7.1f}u | {e_gemm:>9.1f}u "
                      f"{e_hccl:>8.1f}u {e_ours:>7.1f}u | {ceiling:>8.1f}u"
                      f"{mark}", flush=True)
            dist.barrier()

    if rank == 0:
        print()
        print("marginal = (gemm+ours) - gemm, both captured: what our all-reduce "
              "actually costs in situ.")
        print("local    = the same op with world=1: all local traffic (4n), no "
              "peer traffic, no waiting.")
        print("fusion<= = local/2 + 4 us: fusion deletes y's round trip through GM "
              "(2n) and one launch.")
        print("           It cannot delete the barrier or the peer reads, so this "
              "is an upper bound.")
        print("eager columns are host-bound and are here only to show that "
              "+hccl >> +ours end to end;")
        print("           they are NOT comparable with the captured columns.")

    try:
        torch.ops.npu.ipc_close(key)
    except Exception:                                         # noqa: BLE001
        pass
    dist.destroy_process_group()


def main() -> int:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    world = len(vis.split(",")) if vis else 1
    if world < 2:
        print("need >=2 devices")
        return 1
    mp.start_processes(worker, args=(world,), nprocs=world, join=True,
                       start_method="spawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
