#!/usr/bin/env python3
"""Build a structurally valid random QwQ-32B W4A8_DYNAMIC checkpoint.

This artifact is intended only for runtime throughput, memory, and operator
profiling. It does not preserve model quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANTIZED_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
ROW_PARALLEL_WEIGHT_SUFFIXES = (
    ".self_attn.o_proj.weight",
    ".mlp.down_proj.weight",
)
DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("models/QwQ-32B"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/QwQ-32B-W4A8-Random")
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-shard-size-gib", type=float, default=2.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before construction.",
    )
    return parser.parse_args()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def read_source_schema(source: Path) -> OrderedDict[str, dict[str, Any]]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing source index: {index_path}")
    index = json.loads(index_path.read_text())
    files = sorted(set(index["weight_map"].values()))
    schema: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for filename in files:
        with safe_open(source / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                schema[name] = {
                    "shape": list(tensor_slice.get_shape()),
                    "dtype": tensor_slice.get_dtype(),
                }
    return OrderedDict(sorted(schema.items()))


def is_quantized_weight(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in QUANTIZED_WEIGHT_SUFFIXES)


def make_float_tensor(
    name: str, shape: list[int], dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    if name == "model.embed_tokens.weight":
        tensor = torch.empty(shape, dtype=dtype)
        return tensor.uniform_(-0.02, 0.02, generator=generator)
    if name.endswith("norm.weight"):
        return torch.ones(shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def make_quantized_module(
    name: str,
    shape: list[int],
    group_size: int,
    generator: torch.Generator,
) -> OrderedDict[str, torch.Tensor]:
    if len(shape) != 2:
        raise ValueError(f"Expected a matrix for {name}, found {shape}")
    output_size, input_size = shape
    if output_size % 2:
        raise ValueError(f"Output size must be even for packed INT4: {name} {shape}")
    if input_size % group_size:
        raise ValueError(
            f"Input size must be divisible by group size {group_size}: "
            f"{name} {shape}"
        )

    prefix = name.removesuffix(".weight")
    scale_bias_width = (
        16 if any(name.endswith(s) for s in ROW_PARALLEL_WEIGHT_SUFFIXES) else 1
    )
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    # Each byte holds two signed INT4 values along the original output axis.
    tensors[name] = torch.randint(
        -128,
        128,
        (output_size // 2, input_size),
        dtype=torch.int8,
        generator=generator,
    )
    tensors[f"{prefix}.weight_scale"] = torch.ones(
        (output_size, 1), dtype=torch.float32
    )
    tensors[f"{prefix}.weight_offset"] = torch.zeros(
        (output_size, 1), dtype=torch.float32
    )
    tensors[f"{prefix}.weight_scale_second"] = torch.full(
        (output_size, input_size // group_size), 0.001, dtype=torch.float16
    )
    tensors[f"{prefix}.weight_offset_second"] = torch.zeros(
        (output_size, input_size // group_size), dtype=torch.float16
    )
    tensors[f"{prefix}.scale_bias"] = torch.zeros(
        (output_size, scale_bias_width), dtype=torch.float32
    )
    return tensors


def copy_support_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        if path.name in {".msc", ".mdl", ".mv", "README.md"}:
            continue
        shutil.copy2(path, output / path.name)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.group_size not in (64, 128):
        raise ValueError("910B4 MSD Per-Group W4 supports group size 64 or 128")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    schema = read_source_schema(source)
    copy_support_files(source, output)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    random.seed(args.seed)

    shard_limit = math.floor(args.max_shard_size_gib * 1024**3)
    shard: OrderedDict[str, torch.Tensor] = OrderedDict()
    shard_bytes = 0
    shard_paths: list[Path] = []
    temporary_weight_map: dict[str, str] = {}
    quant_description: OrderedDict[str, Any] = OrderedDict(
        (
            ("version", "1.0.0"),
            ("model_quant_type", "W4A8_DYNAMIC"),
            ("group_size", args.group_size),
        )
    )
    total_size = 0

    def flush_shard() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        shard_number = len(shard_paths) + 1
        filename = f"quant_model_weight_w4a8_dynamic-{shard_number:05d}.safetensors"
        path = output / filename
        save_file(shard, path, metadata={"format": "pt"})
        for tensor_name in shard:
            temporary_weight_map[tensor_name] = filename
        shard_paths.append(path)
        print(
            f"wrote {filename}: {path.stat().st_size / 1024**3:.2f} GiB, "
            f"{len(shard)} tensors",
            flush=True,
        )
        shard = OrderedDict()
        shard_bytes = 0

    def add_tensor(name: str, tensor: torch.Tensor, quant_type: str) -> None:
        nonlocal shard_bytes, total_size
        size = tensor_nbytes(tensor)
        if shard and shard_bytes + size > shard_limit:
            flush_shard()
        shard[name] = tensor.contiguous()
        shard_bytes += size
        total_size += size
        quant_description[name] = quant_type

    for name, spec in schema.items():
        if is_quantized_weight(name):
            tensors = make_quantized_module(
                name, spec["shape"], args.group_size, generator
            )
            for tensor_name, tensor in tensors.items():
                add_tensor(tensor_name, tensor, "W4A8_DYNAMIC")
        else:
            dtype_name = spec["dtype"]
            if dtype_name not in DTYPE_MAP:
                raise ValueError(f"Unsupported source dtype {dtype_name}: {name}")
            tensor = make_float_tensor(
                name, spec["shape"], DTYPE_MAP[dtype_name], generator
            )
            add_tensor(name, tensor, "FLOAT")
    flush_shard()

    shard_count = len(shard_paths)
    final_weight_map: dict[str, str] = {}
    for old_path in shard_paths:
        shard_number = int(old_path.stem.rsplit("-", 1)[1])
        final_name = (
            "quant_model_weight_w4a8_dynamic-"
            f"{shard_number:05d}-of-{shard_count:05d}.safetensors"
        )
        final_path = old_path.with_name(final_name)
        os.rename(old_path, final_path)
        for tensor_name, filename in temporary_weight_map.items():
            if filename == old_path.name:
                final_weight_map[tensor_name] = final_name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(final_weight_map.items())),
    }
    (output / "quant_model_weight_w4a8_dynamic.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    (output / "quant_model_description.json").write_text(
        json.dumps(quant_description, indent=2) + "\n"
    )
    (output / "README.md").write_text(
        "# QwQ-32B Random W4A8 Dynamic\n\n"
        "This checkpoint has the QwQ-32B architecture and deterministic random "
        "W4A8 Dynamic weights. It is only for throughput, HBM, and operator "
        "profiling. It must not be used for quality or accuracy evaluation.\n\n"
        f"- seed: `{args.seed}`\n"
        f"- group size: `{args.group_size}`\n"
        "- quantized modules: all attention and MLP Linear weights\n"
        "- format version: `1.0.0` (two INT4 values packed in each INT8 byte)\n"
    )
    print(
        f"complete: {output}, {shard_count} shards, "
        f"{total_size / 1024**3:.2f} GiB tensor payload"
    )


if __name__ == "__main__":
    main()
