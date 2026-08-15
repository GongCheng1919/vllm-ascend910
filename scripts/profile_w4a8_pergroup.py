#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile CANN per-group INT4 WeightQuantBatchMatmulV2."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m-values", default="1,8,16,32,128,1024")
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=6912)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--timed-iters", type=int, default=20)
    return parser.parse_args()


def weight_quant_matmul(x, weight, scale, group_size):
    return torch_npu.npu_weight_quant_batchmatmul(
        x,
        weight,
        antiquant_scale=scale,
        antiquant_group_size=group_size,
    )


def synchronize():
    torch_npu.npu.synchronize()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    m_values = [int(value) for value in args.m_values.split(",")]

    if args.k % args.group_size:
        raise ValueError("K must be divisible by group size")
    if args.n % 64:
        raise ValueError("N must be divisible by 64")

    torch.manual_seed(20260730)
    torch_npu.npu.set_device(0)

    raw_weight = torch.randint(
        -8, 8, (args.k, args.n), dtype=torch.int32, device="cpu"
    ).npu()
    packed_weight = torch_npu.npu_convert_weight_to_int4pack(raw_weight)
    scale = torch.ones(
        (args.k // args.group_size, args.n),
        dtype=torch.bfloat16,
        device="npu",
    )
    del raw_weight
    synchronize()

    metadata = {
        "device": torch_npu.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "m_values": m_values,
        "k": args.k,
        "n": args.n,
        "group_size": args.group_size,
        "activation_dtype": "bfloat16",
        "packed_weight_dtype": str(packed_weight.dtype),
        "packed_weight_shape": list(packed_weight.shape),
        "packed_weight_format": torch_npu.get_npu_format(packed_weight),
        "inner_precise": 0,
        "profiler_schedule": {"wait": 0, "warmup": 5, "active": 5},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for m in m_values:
        x = torch.randn((m, args.k), dtype=torch.bfloat16, device="npu")
        for _ in range(5):
            output = weight_quant_matmul(
                x, packed_weight, scale, args.group_size
            )
        synchronize()

        start = time.perf_counter()
        for _ in range(args.timed_iters):
            output = weight_quant_matmul(
                x, packed_weight, scale, args.group_size
            )
        synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        profile_dir = output_dir / f"m{m}"
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            export_type=torch_npu.profiler.ExportType.Text,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=5, active=5, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(profile_dir), analyse_flag=True
            ),
            record_shapes=True,
            experimental_config=experimental_config,
        ) as profiler:
            for _ in range(10):
                output = weight_quant_matmul(
                    x, packed_weight, scale, args.group_size
                )
                profiler.step()
        synchronize()

        row = {
            "m": m,
            "k": args.k,
            "n": args.n,
            "group_size": args.group_size,
            "average_ms": elapsed_ms / args.timed_iters,
            "output_dtype": str(output.dtype),
            "output_shape": "x".join(str(size) for size in output.shape),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    with (output_dir / "latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
