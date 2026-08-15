#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist
import torch_npu


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    value = torch.tensor(
        [float(dist.get_rank() + 1)], dtype=torch.float32, device="npu"
    )
    dist.all_reduce(value)
    torch.npu.synchronize()
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    actual = float(value.cpu())
    if actual != expected:
        raise RuntimeError(f"all-reduce mismatch: {actual} != {expected}")
    print(
        f"rank={dist.get_rank()} local_rank={local_rank} "
        f"all_reduce={actual:.1f} pass"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

