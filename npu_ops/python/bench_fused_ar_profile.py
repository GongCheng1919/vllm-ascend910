"""Device-side price of fused matmul+all-reduce vs the split pair.

WHY A THIRD SCRIPT.  `bench_fused_allreduce.py` can only time wall clock, and at
decode sizes this box spends ~450 us/iteration in eager dispatch on BOTH arms
(`npu-eager-bench-is-host-bound`), which swamps the ~100 us of device work we care
about.  The two ways out both failed:

  * NPUGraph capture of `dist.all_reduce` hangs here (P6.5 D15);
  * vllm-ascend's `MatmulAllreduceRowParallelOp` dies inside vLLM's graph capture,
    because the first MC2 call on a communicator always fails and vLLM has no retry
    (D15) -- and a retry does not help once the failure lands inside capture.

So this measures the device directly: run each arm under `torch_npu.profiler` and
sum the on-device durations from `kernel_details.csv`.  Host dispatch does not
appear in that sum, which is exactly the point.

The question it answers: does fusing REMOVE the collective's device time, or does
the fused kernel just pay it internally?  D9 measured 3632 us/step of
non-overlapped communication at TP=4 with Overlapped=0; that is the budget a fused
op is trying to reclaim.

usage: ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 python npu_ops/python/bench_fused_ar_profile.py
"""
from __future__ import annotations

import csv
import glob
import os
import shutil
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SHAPES = [("o_proj", 1280, 5120), ("down_proj", 6912, 5120)]
MS = [1, 16]
ITERS = 30
PROF_ROOT = os.environ.get(
    "MC2_PROF_DIR",
    "/tmp/claude-1002/-home-gongcheng-PureInt8LLMPretraining-vllm/"
    "01e437ab-d7b1-4369-b46c-790ceb7ea2b2/scratchpad/mc2_prof")


def _retry(fn, tries: int = 4):
    """First MC2 call on a comm fails; retry with nothing else on the comm (D15)."""
    for _ in range(tries):
        try:
            out = fn()
            torch.npu.synchronize()
            return out
        except Exception:                                     # noqa: BLE001
            continue
    return None


def _device_us(prof_dir: str, iters: int) -> dict[str, float]:
    """Per-iteration device us per kernel name, from the profiler's kernel_details."""
    import torch_npu

    for rank_dir in sorted(glob.glob(os.path.join(prof_dir, "*ascend_pt"))):
        if not os.path.isdir(os.path.join(rank_dir, "ASCEND_PROFILER_OUTPUT")):
            torch_npu.profiler.profiler.analyse(rank_dir)
    tot: dict[str, float] = defaultdict(float)
    for path in glob.glob(os.path.join(prof_dir, "*", "ASCEND_PROFILER_OUTPUT",
                                       "kernel_details.csv")):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = row.get("Name") or row.get("Op Name") or "?"
                dur = row.get("Duration(us)") or row.get("Duration (us)") or "0"
                try:
                    tot[name] += float(dur)
                except ValueError:
                    pass
    return {k: v / iters for k, v in tot.items()}


def _profile(fn, tag: str, rank: int) -> dict[str, float]:
    import torch_npu

    d = os.path.join(PROF_ROOT, f"{tag}_r{rank}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    # Both activities: with NPU alone the export has no device tasks at all
    # ("Failed to get acl to npu flow events", and the db has no COMPUTE_TASK
    # table).  export_type=Text is what produces kernel_details.csv -- the 2.8
    # default is Db, which is why the first run summed to 0.0 us.
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(d))
    with prof:
        for _ in range(ITERS):
            fn()
        torch.npu.synchronize()
    return _device_us(d, ITERS)


def worker(rank: int, world: int) -> None:
    import torch_npu

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29643")
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    pg = dist.distributed_c10d._get_default_group()
    hcom = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    dev = f"npu:{rank}"

    if rank == 0:
        print(f"world={world}  device us/iteration, summed over kernels", flush=True)
        print(f"  {'shape':>10} {'M':>3} | {'split dev':>9} {'fused dev':>9} "
              f"{'saved':>7} {'ratio':>6}", flush=True)

    for name, K, N in SHAPES:
        w = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
        ws = (torch.rand(N, device=dev) * 0.002 + 0.0005).float()
        for M in MS:
            x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
            qx, pts = torch_npu.npu_dynamic_quant(x)

            def split():
                y = torch_npu.npu_quant_matmul(qx, w, ws, pertoken_scale=pts,
                                               output_dtype=torch.float16)
                dist.all_reduce(y)
                return y

            def fused():
                return torch_npu.npu_mm_all_reduce_base(
                    qx, w, hcom, dequant_scale=ws, pertoken_scale=pts)

            if _retry(fused) is None:
                if rank == 0:
                    print(f"  {name:>10} {M:>3} | fused unavailable", flush=True)
                continue

            tag = f"{name}_M{M}"
            s_us = _profile(split, tag + "_split", rank)
            f_us = _profile(fused, tag + "_fused", rank)
            s_tot, f_tot = sum(s_us.values()), sum(f_us.values())
            if rank == 0:
                print(f"  {name:>10} {M:>3} | {s_tot:>9.1f} {f_tot:>9.1f} "
                      f"{s_tot - f_tot:>7.1f} "
                      f"{(f_tot / s_tot if s_tot else float('nan')):>6.3f}",
                      flush=True)
                for label, d in (("split", s_us), ("fused", f_us)):
                    top = sorted(d.items(), key=lambda kv: -kv[1])[:4]
                    parts = "  ".join(f"{k[:38]}={v:.1f}" for k, v in top)
                    print(f"      {label}: {parts}", flush=True)
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
