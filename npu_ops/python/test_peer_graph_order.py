"""Pin the loose end from P6.5 D16: under what conditions does capture stop working?

The clean probe captures and replays the peer all-reduce correctly and does so
REPRODUCIBLY -- 3 runs x 4 configurations, 12/12, right answers. So "a self-written
cross-rank kernel can be captured" is established.

What is NOT established is the condition vLLM actually imposes:
`FULL_DECODE_ONLY` captures MANY graphs, and it captures them AFTER the engine has
already executed. Two earlier harnesses failed exactly there --

  * `test_peer_allreduce.py` check 4: capture after 30 shapes + 400 iterations
    -> "The DDR address of the MTE instruction is out of range"
  * an earlier version of this file: every case -> "the model stream execute failed"

-- but each of those harnesses had extra machinery of its own, so neither failure
can be attributed to the axis it was meant to test. This file therefore grows the
axis INSIDE the scaffold that is known to work, changing one thing at a time:

    A  capture immediately                    (the known-good baseline)
    B  N ops, then capture                    N = 1, 30, 400
    C  capture, run ops, capture again        (interleaving)
    D  K graphs alive at once                 K = 8   (what vLLM does)

Everything runs in ONE process so the baseline and the variants share device state;
A running first and passing is what makes a later failure attributable.

usage: OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 \
       python npu_ops/python/test_peer_graph_order.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

KEY_LEN, N, MAXN = 256, 5120, 40960
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


def capture(buf, ptrs, expect, rank, label, keep):
    """Capture one graph, replay it, report. `keep` holds graphs alive (case D)."""
    try:
        s = torch.npu.Stream()
        s.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(s):
            for _ in range(3):
                torch.ops.npu.peer_read_sum(buf, ptrs)
        torch.npu.current_stream().wait_stream(s)
        torch.npu.synchronize()

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            out = torch.ops.npu.peer_read_sum(buf, ptrs)
        torch.npu.synchronize()
        for _ in range(10):
            g.replay()
        torch.npu.synchronize()
        keep.append((g, out))
        ok = abs(out[0].item() - expect) < 1e-5
        if rank == 0:
            print(f"  {label:<28}: {'OK' if ok else 'WRONG'}  out[0]={out[0].item()}",
                  flush=True)
        return ok
    except Exception as e:                                    # noqa: BLE001
        if rank == 0:
            print(f"  {label:<28}: FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
        return False


def run_ops(buf, ptrs, n):
    for _ in range(n):
        dist.barrier()
        torch.ops.npu.peer_read_sum(buf, ptrs)
    torch.npu.synchronize()


def worker(rank: int, world: int) -> None:
    import torch_npu  # noqa: F401

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29663")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    _load_so()
    dev = f"npu:{rank}"

    torch.ops.npu.enable_peer_access([d for d in range(world) if d != rank])
    staging = torch.zeros(MAXN, device=dev, dtype=torch.float32)
    staging[:N].fill_(float(rank + 1))
    pids = _ag_int(torch.ops.npu.ipc_get_bare_tgid(), rank, world, dev)
    key = torch.ops.npu.ipc_export(staging,
                                   [p for i, p in enumerate(pids) if i != rank])
    keys = _ag_key(key, rank, world, dev)
    ptrs = [staging.data_ptr() if r == rank else torch.ops.npu.ipc_import(keys[r])
            for r in range(world)]
    buf = staging[:N]
    expect = float(world * (world + 1) // 2)
    dist.barrier()

    keep: list = []            # graphs stay alive -- case D depends on it
    if rank == 0:
        print(f"world={world}  one process, baseline first", flush=True)

    capture(buf, ptrs, expect, rank, "A  capture immediately", keep)
    dist.barrier()

    for n in (1, 30, 400):
        run_ops(buf, ptrs, n)
        dist.barrier()
        capture(buf, ptrs, expect, rank, f"B  {n} ops then capture", keep)
        dist.barrier()

    run_ops(buf, ptrs, 10)
    dist.barrier()
    capture(buf, ptrs, expect, rank, "C  capture after interleave", keep)
    dist.barrier()

    # A barrier between captures, like A/B/C have.  Without it the "limit on live
    # graphs" looked real and landed at 11, 13 and 5 on three runs -- i.e. it was
    # never a limit, it was the ranks drifting apart while capturing.
    good = 0
    for i in range(8):
        dist.barrier()
        if not capture(buf, ptrs, expect, rank, f"D  graph #{i + 2} alive", keep):
            break
        good = i + 1
    if rank == 0:
        print(f"  -> {len(keep)} graphs alive at exit", flush=True)

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
