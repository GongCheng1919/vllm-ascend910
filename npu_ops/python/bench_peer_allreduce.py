"""What does our own all-reduce COST? (P6.5 D17)

`test_peer_barrier.py` proved the in-kernel-barrier op is correct and capturable.
That says nothing about whether it is worth shipping.  The whole TP line is priced
off one number (D13): the 3632 us/step of unoverlapped communication is what stands
between "TP is exactly break-even" and 1.78x.  Our op only earns that if it is at
least as cheap as the `dist.all_reduce` it replaces -- a single-core kernel that
costs 200 us per call would kill the line no matter how correct it is.

MEASUREMENT NOTES (both of these bit us before):

  * eager loops on this box are host-bound (~900 us/layer of python dispatch), so
    OUR side is timed inside a captured graph: R calls are captured into one graph,
    the graph is replayed G times, and the wall clock is divided by R*G.  Nothing
    synchronises inside, so what is left is device time.
  * HCCL cannot be captured here (`dist.all_reduce` under NPUGraph hangs at
    "Waiting for pending NCCL work", D14/§2), so its baseline is a back-to-back
    eager loop with a single sync at the end -- the host runs ahead and the device
    queue is the bottleneck.  This method FAVOURS HCCL slightly (no capture
    overhead), which is the right way round for a claim of the form "ours is
    cheaper".
  * every rank is timed; the reported figure is the MAX over ranks, because a
    collective costs what its slowest participant costs.

The engine's tensors are bf16, ours is f32 -- twice the bytes.  HCCL is therefore
timed at both dtypes: f32 is the byte-matched comparison, bf16 is what the engine
actually pays today.  At decode sizes both should land on the same latency floor
(D11: 10 KB and 640 KB cost the same); if they do not, the ratio is the honest one.

usage: OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 \
       python npu_ops/python/bench_peer_allreduce.py
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO_CANDIDATES = ("build-venv/libvllm_w4a8_npu_ops.so", "build/libvllm_w4a8_npu_ops.so")

HIDDEN = 5120           # QwQ-32B
KEY_LEN = 256
SPIN_LIMIT = 200_000_000

# (label, elements).  8 is the latency floor -- one 32 B block, so whatever it
# costs is pure barrier + launch, with no payload to hide behind.
CASES = [
    ("floor  (16 el, 32 B)", 16),
    ("M=1    (5120 el)", HIDDEN),
    ("M=4    (20480 el)", 4 * HIDDEN),
    ("M=16   (81920 el)", 16 * HIDDEN),
    ("M=64   (327680 el)", 64 * HIDDEN),
]
CAP = max(n for _, n in CASES)

CAPTURE_CALLS = 20      # calls recorded into one graph
GRAPH_REPLAYS = 25      # replays of that graph  => 500 timed calls
HCCL_ITERS = 500
WARMUP = 20


def _load_so() -> str:
    for rel in SO_CANDIDATES:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            torch.ops.load_library(p)
            return p
    raise FileNotFoundError("no .so built -- run npu_ops/build.sh")


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
    # float32, not float64: HCCL here rejects kDouble ("Unsupported data type").
    t = torch.zeros(world, dtype=torch.float32, device=dev)
    t[rank] = v
    dist.all_reduce(t)
    return [float(x) for x in t.tolist()]


def worker(rank: int, world: int) -> None:
    import torch_npu  # noqa: F401

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29673")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    _load_so()
    dev = f"npu:{rank}"
    torch.ops.npu.enable_peer_access([d for d in range(world) if d != rank])

    elems = torch.ops.npu.peer_staging_elems(CAP)
    staging = torch.zeros(elems, device=dev, dtype=torch.float32)
    pids = _ag_int(torch.ops.npu.ipc_get_bare_tgid(), rank, world, dev)
    key = torch.ops.npu.ipc_export(staging,
                                   [p for i, p in enumerate(pids) if i != rank])
    keys = _ag_key(key, rank, world, dev)
    ptrs = [staging.data_ptr() if r == rank else torch.ops.npu.ipc_import(keys[r])
            for r in range(world)]
    dist.barrier()

    if rank == 0:
        print(f"world={world}  staging={elems * 4 / 1024:.0f} KiB  "
              f"ours: {CAPTURE_CALLS} calls/graph x {GRAPH_REPLAYS} replays, "
              f"HCCL: {HCCL_ITERS} eager iters\n", flush=True)
        print(f"{'case':<22} {'bf16 B':>8} {'ours bf':>9} {'ours f32':>8} "
              f"{'ourseag':>10} {'hccl bf':>8} {'hccl32':>8} {'speedup':>8} "
              f"{'ours GB/s':>9}")
        print("-" * 104, flush=True)

    rows = []
    for label, n in CASES:
        ours = {}
        exact = {}
        for dt in (torch.float32, torch.bfloat16):
            x = torch.full((n,), float(rank + 1), device=dev, dtype=dt)

            # ---- correctness first: a fast wrong kernel is not a result ------
            # bf16 is graded against an F32-ACCUMULATED reference rounded once --
            # what the kernel computes, and strictly better than a
            # bf16-accumulating collective.  Grading it against HCCL's bf16 output
            # would grade us on THEIR rounding.
            # .clone() FIRST: for an f32 input `x.float()` returns x itself, so
            # the all_reduce below would reduce the input in place and every
            # comparison after it would be against a corrupted x.
            ref = x.detach().clone().float()
            dist.all_reduce(ref)
            ref = ref.to(dt)
            got = torch.ops.npu.peer_allreduce(x, staging, ptrs, rank, CAP,
                                               SPIN_LIMIT)
            torch.npu.synchronize()
            exact[dt] = bool(torch.equal(got, ref))

            # ---- captured graph, no host sync inside ------------------------
            for _ in range(WARMUP):
                torch.ops.npu.peer_allreduce(x, staging, ptrs, rank, CAP,
                                             SPIN_LIMIT)
            torch.npu.synchronize()
            dist.barrier()

            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                for _ in range(CAPTURE_CALLS):
                    torch.ops.npu.peer_allreduce(x, staging, ptrs, rank, CAP,
                                                 SPIN_LIMIT)
            torch.npu.synchronize()
            g.replay()                  # one replay to settle, then time
            torch.npu.synchronize()
            dist.barrier()

            t0 = time.perf_counter()
            for _ in range(GRAPH_REPLAYS):
                g.replay()
            torch.npu.synchronize()
            ours[dt] = ((time.perf_counter() - t0)
                        / (GRAPH_REPLAYS * CAPTURE_CALLS) * 1e6)
            del g, x
            dist.barrier()

        # ---- ours, EAGER (bf16): dispatch included, so the host cost is visible
        # rather than implied.
        x = torch.full((n,), float(rank + 1), device=dev, dtype=torch.bfloat16)
        torch.npu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(HCCL_ITERS):
            torch.ops.npu.peer_allreduce(x, staging, ptrs, rank, CAP, SPIN_LIMIT)
        t_issue = time.perf_counter()
        torch.npu.synchronize()
        ours_eager = (time.perf_counter() - t0) / HCCL_ITERS * 1e6
        ours_issue = (t_issue - t0) / HCCL_ITERS * 1e6
        ours_us = ours[torch.bfloat16]

        # ---- HCCL baseline, both dtypes -------------------------------------
        # ISSUE time is recorded separately: if issuing the calls costs as much as
        # the whole loop, this probe is measuring python, not HCCL, and the eager
        # number must NOT be quoted as a device cost.  (The engine pays no such
        # cost -- it captures its collectives.)
        hccl = {}
        hccl_issue = {}
        for dt in (torch.float32, torch.bfloat16):
            y = torch.full((n,), 1.0, device=dev, dtype=dt)
            for _ in range(WARMUP):
                dist.all_reduce(y)
            torch.npu.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(HCCL_ITERS):
                dist.all_reduce(y)
            t_issue = time.perf_counter()
            torch.npu.synchronize()
            hccl[dt] = (time.perf_counter() - t0) / HCCL_ITERS * 1e6
            hccl_issue[dt] = (t_issue - t0) / HCCL_ITERS * 1e6

        # slowest rank is what a collective actually costs
        ours_f32 = max(_ag_float(ours[torch.float32], rank, world, dev))
        ours_us = max(_ag_float(ours_us, rank, world, dev))
        ours_eager = max(_ag_float(ours_eager, rank, world, dev))
        ours_issue = max(_ag_float(ours_issue, rank, world, dev))
        h32 = max(_ag_float(hccl[torch.float32], rank, world, dev))
        hbf = max(_ag_float(hccl[torch.bfloat16], rank, world, dev))
        hbf_issue = max(_ag_float(hccl_issue[torch.bfloat16], rank, world, dev))
        ex_f32 = min(_ag_float(1.0 if exact[torch.float32] else 0.0, rank, world,
                                dev)) > 0.5
        ex_bf16 = min(_ag_float(1.0 if exact[torch.bfloat16] else 0.0, rank, world,
                                dev)) > 0.5
        all_exact = ex_f32 and ex_bf16

        # each rank moves its own n floats out and (world-1)*n floats in
        gbs = n * 2 * world / (ours_us * 1e-6) / 1e9
        rows.append((label, n, ours_us, h32, hbf, all_exact, gbs, ours_f32))
        if rank == 0:
            mark = ("" if all_exact else
                    f"  <-- NOT EXACT (f32 {ex_f32}, bf16 {ex_bf16})")
            print(f"{label:<22} {n * 2:>8} {ours_us:>8.1f}u {ours_f32:>8.1f}u "
                  f"{ours_eager:>9.1f}u {hbf:>8.1f}u {h32:>8.1f}u "
                  f"{hbf / ours_us:>7.2f}x {gbs:>8.1f}{mark}", flush=True)
        dist.barrier()

    if rank == 0:
        print()
        bad = [r[0] for r in rows if not r[5]]
        if bad:
            print(f"WARNING: not bitwise equal to dist.all_reduce: {bad}")
        # The decode-relevant verdict: M=1 is what a batch-1 step pays per layer.
        m1 = next(r for r in rows if r[1] == HIDDEN)
        print(f"decode verdict (M=1, {HIDDEN} el, bf16 -- what the GEMM emits): "
              f"ours {m1[2]:.1f} us vs HCCL bf16 {m1[4]:.1f} us "
              f"= {m1[4] / m1[2]:.1f}x")
        print(f"  bf16 vs our own f32 path: {m1[7]:.1f} -> {m1[2]:.1f} us "
              f"({m1[7] / m1[2]:.2f}x from halving the bytes)")
        print("  NOTE: do not multiply this by layer count -- the engine profile "
              "measured 113 us/call")
        print("        device-side (3632/32) while this microbench measures ~250; "
              "the RATIO is what transfers.")

    try:
        torch.ops.npu.ipc_close(key)
    except Exception:                                         # noqa: BLE001
        pass
    dist.destroy_process_group()


def main() -> int:
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    world = len(vis.split(",")) if vis else 1
    if world < 2:
        print("need >=2 devices, set ASCEND_RT_VISIBLE_DEVICES")
        return 1
    mp.start_processes(worker, args=(world,), nprocs=world, join=True,
                       start_method="spawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
