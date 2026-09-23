"""Qwen3.8-27B decode throughput under vLLM, graph capture on vs off.

HF transformers gives 4.8 tok/s on this model because the step is 99.9%
host-bound (~9,900 aten ops at ~21 us each to issue). The memory-bound floor is
51.0 GiB / 1.1 TB/s = 46.4 ms, i.e. 22 tok/s. This asks how much of that gap
graph capture actually closes. No ANS anywhere -- this is the bf16 baseline.

Usage::

    python3 tests/vllm_baseline.py --model <hf dir> [--eager] [--out N]
"""
import argparse, os, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eager", action="store_true", help="disable graph capture")
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--util", type=float, default=0.93)
    # vllm_ascend 0.23.0's libvllm_ascend_kernels.so was built against a newer
    # CANN and fails to load here on ONE missing symbol,
    # `rtFunctionGetMetaInfoSize`, so `torch.ops._C_ascend.npu_gemma_rms_norm`
    # and friends do not exist. "none" makes every vLLM CustomOp use
    # forward_native instead of forward_oot. Graph capture is unaffected --
    # it is what we are actually measuring.
    ap.add_argument("--custom-ops", default="all")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(model=args.model,
              dtype="bfloat16",
              max_model_len=args.max_len,
              max_num_seqs=1,
              gpu_memory_utilization=args.util,
              enforce_eager=args.eager,
              limit_mm_per_prompt={"image": 0, "video": 0},
              compilation_config={"custom_ops": [args.custom_ops]},
              # vllm-ascend reads its fusion switches from ITS OWN
              # additional_config["ascend_compilation_config"], not from vLLM's
              # compilation_config.pass_config. The norm-quant pass builds its
              # match pattern out of npu_add_rms_norm_bias ->
              # aclnnAddRmsNormBias, which CANN 8.5.0 does not have. It fuses a
              # norm with a QUANTISATION, so in bf16 it has nothing to do.
              additional_config={
                  "ascend_compilation_config": {"fuse_norm_quant": False},
              },
              trust_remote_code=True)
    print(f"\n[vllm] engine up in {time.perf_counter() - t0:.0f} s "
          f"(graph capture {'OFF' if args.eager else 'ON'})", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.out, ignore_eos=True)
    prompt = "The capital of France is"
    llm.generate([prompt], sp)                       # warm up / capture
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp)
    dt = time.perf_counter() - t0
    n = len(outs[0].outputs[0].token_ids)
    print(f"\n[vllm] {'eager' if args.eager else 'graph'}: {n} tokens in {dt:.2f} s "
          f"= {n / dt:.1f} tok/s  ({dt / n * 1e3:.1f} ms/token)")
    print(f"       HF transformers on the same model: 4.8 tok/s (207.9 ms/token)")
    print(f"       memory-bound floor:               22.0 tok/s (46.4 ms/token)")

if __name__ == "__main__":
    main()
