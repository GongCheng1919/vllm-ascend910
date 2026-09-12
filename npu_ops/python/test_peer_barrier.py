"""Does the IN-KERNEL barrier actually synchronise the ranks? (P6.5 D17)

D16's probe read peer memory correctly, but the ranks were held together by the
CALLER's `dist.barrier()`. A captured graph cannot contain a host-side collective,
so the real op has to synchronise itself. `peer_allreduce_f32` does; this checks it.

THE ONLY TEST THAT PROVES ANYTHING IS THE SKEWED ONE. Run the ranks in lockstep and
a completely broken barrier still passes, because everyone happens to be ready. So
every case here deliberately *desynchronises* the ranks first, and the values are
chosen so that reading a stale slot gives a visibly wrong sum rather than a
plausible one.

  1. lockstep        -- sanity only; passing here means nothing on its own
  2. skewed launch   -- rank r sleeps r * 25 ms before launching
  3. skewed + values change every iteration, so a stale read is detectable
  4. long skewed run -- 200 iterations with a random skew each time
  5. graph capture   -- the point of the exercise: captured, no host barrier at all
  6. MANY graphs     -- vLLM's FULL_DECODE_ONLY keeps dozens of graphs alive, and
                        D16 left an unresolved ~1/3 failure on the 13th graph of
                        the OLD probe (`peer_read_sum`, which has no in-kernel
                        barrier).  Same axis, real op, one graph per decode batch
                        bucket.

usage: OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 \
       python npu_ops/python/test_peer_barrier.py
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

KEY_LEN = 256
N = 5120            # QwQ-32B hidden
CAP = 8192          # per-slot capacity, >= N
SPIN_LIMIT = 200_000_000
ITERS_LONG = 200
GRAPH_REPLAYS = 20
GRAPHS_ALIVE = 24     # more than D16's 13-graph failure point

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO_CANDIDATES = ("build-venv/libvllm_w4a8_npu_ops.so", "build/libvllm_w4a8_npu_ops.so")


def _load_so() -> str:
    for rel in SO_CANDIDATES:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            torch.ops.load_library(p)
            return p
    raise FileNotFoundError("no .so built")


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


def run_gates(rank, world, dev, dtype, staging, ptrs) -> list:
    """The six gates, for one wire dtype.

    VALUES ARE CHOSEN TO BE EXACT IN BF16 -- multiples of 0.25 with a sum under 64.
    The obvious scheme ((rank+1)*1000 + step) is NOT: 10000 needs more than bf16's 8
    significant bits, so every "wrong" result would just be rounding and the gate
    would prove nothing about the barrier. Consecutive steps still differ by 0.25,
    which is far above the 1e-3 tolerance, so a stale slot is still visible.
    """
    failures: list[str] = []
    x = torch.empty(N, device=dev, dtype=dtype)

    def allreduce(step: int) -> torch.Tensor:
        """No dist.barrier() anywhere -- the kernel is supposed to do it."""
        return torch.ops.npu.peer_allreduce(x, staging, ptrs, rank, CAP, SPIN_LIMIT)

    def value(r: int, step: int) -> float:
        return (r + 1) * 4.0 + (step % 16) * 0.25

    def expect(step: int) -> float:
        return float(sum(value(r, step) for r in range(world)))

    def fill(step: int) -> None:
        x.fill_(value(rank, step))

    # ---- 1. lockstep (sanity only) -------------------------------------------
    fill(0)
    dist.barrier()
    got = allreduce(0)
    torch.npu.synchronize()
    ok = abs(got[0].item() - expect(0)) < 1e-3
    if not ok:
        failures.append("lockstep: wrong sum")
    if rank == 0:
        print(f"[1/6] lockstep (proves little) : {'OK' if ok else 'WRONG'} "
              f"got={got[0].item()} want={expect(0)}", flush=True)

    # ---- 2. skewed launch -----------------------------------------------------
    # Ranks enter the kernel up to 75 ms apart. A barrier that does not really wait
    # lets a fast rank read a slot its slow peer has not written yet.
    fill(1)
    dist.barrier()
    time.sleep(0.025 * rank)
    got = allreduce(1)
    torch.npu.synchronize()
    ok = abs(got[0].item() - expect(1)) < 1e-3
    if not ok:
        failures.append(f"skewed: got {got[0].item()}, want {expect(1)}")
    if rank == 0:
        print(f"[2/6] skewed launch (25ms/rank): {'OK' if ok else 'WRONG'} "
              f"got={got[0].item()} want={expect(1)}", flush=True)

    # ---- 3. skewed, values change every step ---------------------------------
    # Now a stale read is visible: last step's value differs from this step's.
    bad = 0
    for step in range(2, 12):
        fill(step)
        dist.barrier()
        time.sleep(0.01 * ((rank + step) % world))
        got = allreduce(step)
        torch.npu.synchronize()
        if abs(got[0].item() - expect(step)) > 1e-3:
            bad += 1
            if rank == 0 and bad <= 2:
                print(f"      step {step}: got {got[0].item()} want {expect(step)}",
                      flush=True)
    if bad:
        failures.append(f"changing values: {bad}/10 steps wrong")
    if rank == 0:
        print(f"[3/6] skewed + changing values : {10 - bad}/10 exact"
              f"{'' if not bad else '  <-- FAIL'}", flush=True)

    # ---- 4. long skewed run ---------------------------------------------------
    bad = 0
    rng = random.Random(1234 + rank)
    t0 = time.perf_counter()
    for step in range(12, 12 + ITERS_LONG):
        fill(step)
        if rng.random() < 0.1:
            time.sleep(rng.random() * 0.005)
        got = allreduce(step)
        if step % 50 == 0:
            torch.npu.synchronize()
            if abs(got[0].item() - expect(step)) > 1e-3:
                bad += 1
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / ITERS_LONG * 1e6
    if bad:
        failures.append(f"long run: {bad} checked steps wrong")
    if rank == 0:
        print(f"[4/6] {ITERS_LONG} iters, random skew, NO host barrier : "
              f"{'OK' if not bad else 'FAILED'}  ({dt:.1f} us/iter)", flush=True)

    # ---- 5. graph capture, no host barrier at all -----------------------------
    # The kernel bumps its sequence counter in device memory, so each replay is a
    # new round and the barrier has to work on replay too -- which is exactly what
    # a captured decode step needs.
    try:
        fill(500)
        dist.barrier()
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(3):
                allreduce(500)
        torch.npu.current_stream().wait_stream(s)
        torch.npu.synchronize()
        dist.barrier()

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            captured = allreduce(500)
        torch.npu.synchronize()
        for _ in range(GRAPH_REPLAYS):
            g.replay()
        torch.npu.synchronize()
        ok = abs(captured[0].item() - expect(500)) < 1e-3
        if not ok:
            failures.append(f"graph: got {captured[0].item()}, want {expect(500)}")
        if rank == 0:
            print(f"[5/6] captured + {GRAPH_REPLAYS} replays : "
                  f"{'OK' if ok else 'WRONG'} got={captured[0].item()} "
                  f"want={expect(500)}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        failures.append(f"graph: {type(e).__name__}: {str(e)[:110]}")
        if rank == 0:
            print(f"[5/6] captured : FAILED {type(e).__name__}: {str(e)[:150]}",
                  flush=True)

    # ---- 6. many graphs alive at once ----------------------------------------
    # One graph per shape, ALL kept alive, then replayed in order.  The ranks are
    # barriered BETWEEN replays (never inside one): the protocol assumes every rank
    # issues the same sequence of calls -- which vLLM guarantees, since every rank
    # walks the same layers -- and this test is about graph count, not about the
    # barrier, which cases 2-4 already cover.
    try:
        # multiples of 16: one 32 B DataCopy block is 8 floats but 16 bf16, so a
        # shape legal for f32 can be illegal for bf16.
        shapes = [16 * (i + 1) for i in range(4)] + [512, 1024, 2048, 2560, 5120]
        shapes = (shapes * 3)[:GRAPHS_ALIVE]
        graphs = []
        for i, n in enumerate(shapes):
            xi = torch.full((n,), value(rank, 0), device=dev, dtype=dtype)
            for _ in range(2):      # warm the shape before capturing it
                torch.ops.npu.peer_allreduce(xi, staging, ptrs, rank, CAP,
                                             SPIN_LIMIT)
            torch.npu.synchronize()
            dist.barrier()
            gi = torch.npu.NPUGraph()
            with torch.npu.graph(gi):
                oi = torch.ops.npu.peer_allreduce(xi, staging, ptrs, rank, CAP,
                                                  SPIN_LIMIT)
            torch.npu.synchronize()
            # xi is kept alive DELIBERATELY: a captured graph replays against the
            # frozen input ADDRESS, so letting the input fall out of scope hands
            # its memory to the next allocation and the replay reduces whatever
            # landed there.  That is what the first version of this gate did, and
            # it looked exactly like "many graphs corrupt each other" -- wrong
            # values on 10 of 24 graphs, no error.
            graphs.append((n, gi, oi, xi))
            dist.barrier()

        want = expect(0)
        bad = []
        for i, (n, gi, oi, _xi) in enumerate(graphs):
            gi.replay()
            torch.npu.synchronize()
            if abs(oi[0].item() - want) > 1e-3:
                bad.append(i)
            dist.barrier()
        if bad:
            failures.append(f"many graphs: wrong value on graph(s) {bad}")
        if rank == 0:
            print(f"[6/6] {len(graphs)} graphs alive, captured + replayed : "
                  f"{'OK' if not bad else 'WRONG on ' + str(bad)}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        failures.append(f"many graphs: {type(e).__name__}: {str(e)[:110]}")
        if rank == 0:
            print(f"[6/6] many graphs FAILED: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)

    return failures


def worker(rank: int, world: int) -> None:
    import torch_npu  # noqa: F401

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29671")
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
        print(f"world={world}  staging={elems} elems  N={N} cap={CAP}", flush=True)

    failures: list[str] = []
    for dtype in (torch.float32, torch.bfloat16):
        if rank == 0:
            print(f"\n--- dtype={str(dtype).split('.')[-1]} "
                  f"{'(what the GEMM emits, and what the engine reduces)' if dtype == torch.bfloat16 else ''}",
                  flush=True)
        failures += run_gates(rank, world, dev, dtype, staging, ptrs)

    if rank == 0:
        print()
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print(f"  - {f}")
        else:
            print("RESULT: PASS -- the in-kernel barrier holds under skew and "
                  "under graph capture")
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
