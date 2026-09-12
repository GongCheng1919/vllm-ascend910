"""Gate for the fused GEMM + all-reduce line: can a SELF-WRITTEN kernel do
cross-rank communication here, and survive the two conditions that killed CANN's?

BACKGROUND (P6.5 D15).  `npu_mm_all_reduce_base` (CANN's MC2 fused
matmul+all-reduce) microbenches green on this box -- hundreds of iterations, 4
ranks, both dtypes -- and then hangs inside vLLM, in FULL_DECODE_ONLY graph mode
AND with `--eager`.  So "the primitive works" turned out not to transfer, and the
question for our own fused op is no longer "do the primitives exist" but "does a
cross-rank kernel survive at system scale".

This file answers that for the design that shares the LEAST with MC2: peer HBM read
directly via `aclrtDeviceEnablePeerAccess`, no HCCL object, no comm-resource
allocation, no task queue.  Four checks, in dependency order -- each one only means
something if the previous passed:

  0. one mapping   -- a single persistent staging buffer, exported/imported ONCE.
                      Re-exporting per shape exhausts the IPC registry (507899) and
                      is not what a real op would do: it would map a staging buffer
                      at init and slice it forever after.
  1. correctness   -- out == sum of every rank's buffer, vs dist.all_reduce
  2. many shapes   -- dozens of distinct sizes back to back.  MC2's failure is
                      keyed on tiling, and a real model has dozens of them; this
                      is the axis a two-shape microbench cannot see.
  3. long run      -- hundreds of consecutive iterations, no hang
  4. GRAPH CAPTURE -- captured into NPUGraph and replayed.  This is the one that
                      decides the project: every decode win we have lives inside
                      FULL_DECODE_ONLY, and falling back to PIECEWISE measured
                      9.3 -> 16.0 ms, far more than the 3.6 ms the fusion saves.

Check 4 is expected to be the hard one and is deliberately last, so a failure there
still leaves 1-3 as a usable result.

usage:  OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 \
        python npu_ops/python/test_peer_allreduce.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SO_CANDIDATES = ("build-venv/libvllm_w4a8_npu_ops.so", "build/libvllm_w4a8_npu_ops.so")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # npu_ops/

# Sizes are multiples of 8 floats (the kernel's 32 B DataCopy granularity).
# 5120 = QwQ-32B hidden, 640 = its TP=8 shard: the shapes a row-parallel all-reduce
# would actually see, plus deliberate odd ones.
SHAPES_MANY = [8, 16, 64, 128, 256, 512, 640, 1024, 1280, 2048, 2560, 4096,
               5120, 8192, 10240, 20480, 40960, 24, 40, 72, 136, 264, 520,
               1032, 2056, 4104, 5128, 8200, 648, 1288]
MAX_N = max([8, 16, 64, 128, 256, 512, 640, 1024, 1280, 2048, 2560, 4096,
             5120, 8192, 10240, 20480, 40960, 24, 40, 72, 136, 264, 520,
             1032, 2056, 4104, 5128, 8200, 648, 1288])
LONG_RUN_ITERS = 400
GRAPH_REPLAYS = 50


def _load_so() -> str:
    for rel in SO_CANDIDATES:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            torch.ops.load_library(p)
            return p
    raise FileNotFoundError(f"no .so found under {HERE} in {SO_CANDIDATES}")


KEY_LEN = 256
_MY_KEY = [""]


def _all_gather_int(value: int, rank: int, world: int, dev: str) -> list[int]:
    t = torch.zeros(world, dtype=torch.int64, device=dev)
    t[rank] = value
    dist.all_reduce(t)                          # each slot written by exactly one rank
    return [int(v) for v in t.tolist()]


def _all_gather_key(key: str, rank: int, world: int, dev: str) -> list[str]:
    """All-gather fixed-width IPC key blobs as bytes, no pickling / object gather."""
    buf = torch.zeros(world, KEY_LEN, dtype=torch.uint8, device=dev)
    raw = key.encode()
    assert len(raw) < KEY_LEN, f"IPC key too long: {len(raw)}"
    buf[rank, : len(raw)] = torch.tensor(list(raw), dtype=torch.uint8, device=dev)
    dist.all_reduce(buf)
    out = []
    for r in range(world):
        b = bytes(buf[r].tolist()).rstrip(b"\x00")
        out.append(b.decode())
    return out


def _exchange_ptrs(buf: torch.Tensor, rank: int, world: int) -> list[int]:
    """Map every rank's staging buffer into THIS rank's address space.

    Raw `data_ptr()` values cannot be used across processes here: all four ranks
    report the identical address (0x12c041200000 measured), because there is no
    unified virtual address space across devices. Handing that number to the kernel
    reads LOCAL memory silently -- the first version of this test scored
    world x (own value) on every shape and reported no error at all.

    So each rank exports an IPC key for its own buffer, whitelisting the other
    ranks' bare tgids, and imports the peers' keys with ENABLE_PEER_ACCESS. The
    imported addresses are valid locally and genuinely distinct.
    """
    dev = str(buf.device)
    my_pid = torch.ops.npu.ipc_get_bare_tgid()
    pids = _all_gather_int(my_pid, rank, world, dev)
    peers_pids = [p for i, p in enumerate(pids) if i != rank]

    my_key = torch.ops.npu.ipc_export(buf, peers_pids)
    _MY_KEY[0] = my_key      # kept so it can be closed at exit (507899)
    keys = _all_gather_key(my_key, rank, world, dev)

    ptrs = []
    for r in range(world):
        ptrs.append(buf.data_ptr() if r == rank
                    else torch.ops.npu.ipc_import(keys[r]))

    # The bug this test was built to avoid: identical addresses look like a
    # working all-reduce that is really a local read x world. Refuse to measure.
    assert len(set(ptrs)) == world, (
        f"rank {rank}: peer addresses are not distinct ({[hex(p) for p in ptrs]}) "
        "-- the IPC import did not produce a real mapping")
    return ptrs


def worker(rank: int, world: int) -> None:
    import torch_npu  # noqa: F401

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29651")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    dev = f"npu:{rank}"
    so = _load_so()
    if rank == 0:
        print(f"world={world}  so={so}", flush=True)

    # Every rank must be able to reach every other rank's HBM.
    torch.ops.npu.enable_peer_access([d for d in range(world) if d != rank])
    if rank == 0:
        print("[1/4] enable_peer_access OK on all ranks", flush=True)

    failures: list[str] = []

    def allreduce(buf: torch.Tensor, ptrs: list[int]) -> torch.Tensor:
        return torch.ops.npu.peer_read_sum(buf, ptrs)

    # ---- the single mapping, done once ---------------------------------------
    # Every shape below is a VIEW into this one buffer, so the peer pointers stay
    # valid and no key is ever re-exported.
    staging = torch.zeros(MAX_N, device=dev, dtype=torch.float32)
    ptrs = _exchange_ptrs(staging, rank, world)
    if rank == 0:
        print(f"      mapped peers: {[hex(x) for x in ptrs]}", flush=True)

    # ---- check 1+2: correctness across many distinct shapes -------------------
    # Distinct sizes through ONE mapping is the microbench analogue of "a real
    # model has dozens of distinct tilings on one communicator".
    bad = 0
    for n in SHAPES_MANY:
        staging[:n].fill_(float(rank + 1))
        buf = staging[:n]
        dist.barrier()                      # caller-side sync: no in-kernel barrier yet
        got = allreduce(buf, ptrs)
        torch.npu.synchronize()

        ref = buf.clone()
        dist.all_reduce(ref)
        if not torch.equal(got, ref):
            bad += 1
            if rank == 0 and bad <= 3:
                d = (got - ref).abs().max().item()
                print(f"      MISMATCH n={n} max|diff|={d:.3e} "
                      f"got[0]={got[0].item()} ref[0]={ref[0].item()}", flush=True)
    if bad:
        failures.append(f"correctness: {bad}/{len(SHAPES_MANY)} shapes wrong")
    if rank == 0:
        print(f"[2/4] {len(SHAPES_MANY) - bad}/{len(SHAPES_MANY)} shapes exact"
              f"{'' if not bad else '  <-- FAIL'}", flush=True)

    # ---- check 3: long consecutive run ---------------------------------------
    n = 5120
    staging[:n].fill_(float(rank + 1))
    buf = staging[:n]
    expect = float(world * (world + 1) // 2)
    t0 = time.perf_counter()
    hung = False
    for i in range(LONG_RUN_ITERS):
        dist.barrier()
        out = allreduce(buf, ptrs)
        if i % 100 == 0:
            torch.npu.synchronize()
            if abs(out[0].item() - expect) > 1e-5:
                failures.append(f"long run: drifted at iter {i}")
                hung = True
                break
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / LONG_RUN_ITERS * 1e6
    if rank == 0:
        print(f"[3/4] {LONG_RUN_ITERS} consecutive iterations "
              f"{'OK' if not hung else 'FAILED'}  ({dt:.1f} us/iter incl. barrier)",
              flush=True)

    # ---- check 4: NPUGraph capture + replay -----------------------------------
    # THE decisive one.  Note there is no dist.barrier() inside the captured region:
    # a host-side collective cannot be captured, which is precisely why the real op
    # will need an in-kernel barrier.  Here the inputs are constant across replays,
    # so the sum is well-defined without one, and the question under test is purely
    # "can this kernel be captured and replayed at all".
    try:
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(3):
                allreduce(buf, ptrs)
        torch.npu.current_stream().wait_stream(s)
        torch.npu.synchronize()
        dist.barrier()

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            captured = allreduce(buf, ptrs)
        torch.npu.synchronize()

        for _ in range(GRAPH_REPLAYS):
            g.replay()
        torch.npu.synchronize()

        ok = abs(captured[0].item() - expect) < 1e-5
        if not ok:
            failures.append(f"graph: replay gave {captured[0].item()}, want {expect}")
        if rank == 0:
            print(f"[4/4] NPUGraph capture + {GRAPH_REPLAYS} replays: "
                  f"{'OK' if ok else 'WRONG VALUE'}  (out[0]={captured[0].item()})",
                  flush=True)
    except Exception as e:                                    # noqa: BLE001
        failures.append(f"graph: {type(e).__name__}: {str(e)[:120]}")
        if rank == 0:
            print(f"[4/4] NPUGraph capture FAILED: {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            traceback.print_exc()

    if rank == 0:
        print()
        if failures:
            print("RESULT: FAIL")
            for f in failures:
                print(f"  - {f}")
        else:
            print("RESULT: PASS -- a self-written cross-rank kernel works here, "
                  "including under graph capture")
    try:
        torch.ops.npu.ipc_close(_MY_KEY[0])
    except Exception:                                         # noqa: BLE001
        pass
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
    sys.exit(main())
