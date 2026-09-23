"""Qwen3.8-27B decode step, linears only: ANS fused vs bf16 matmul, device time.

No 52 GiB checkpoint needed -- the shapes below are read off the published
``model.safetensors.index.json`` and the timing does not depend on the weight
VALUES, only on the shapes and on the wide-group fraction (which is what the
`wide%` column reports, so a synthetic run says when it is lying).

The model is hybrid: 48 ``linear_attention`` layers and 16 ``full_attention``
layers out of 64, every 4th layer full. Both carry the same dense MLP. The
lm_head is 248320x5120 and runs once per token, which is 2.4 GiB of bf16 --
too big to leave out of a decode-step projection.

Usage::

    python3 tests/bench_qwen38_layer.py bind/librans.so [device] [M]
"""
import sys, time, numpy as np, torch, torch_npu

torch.ops.load_library(sys.argv[1])
torch.npu.set_device(int(sys.argv[2]) if len(sys.argv) > 2 else 0); dev = "npu:0"
sys.path.insert(0, ".")
from codec.fixed_codec import encode_fixed_gemm

def bf(a):
    f = np.ascontiguousarray(a, dtype=np.float32); u = f.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def to_f32(u): return (np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def u16(a, m=0):
    a = np.ascontiguousarray(a, dtype=np.uint16)
    if a.size < m: a = np.zeros(m, dtype=np.uint16)
    return torch.from_numpy(a.view(np.int16)).to(dev)
def i32(a): return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)
def bench(fn, warmup=10, rep=30, trials=5):
    """Min over trials, not mean: a single-op wall clock on this box has ~50%
    run-to-run spread and the mean of a noisy floor is not the floor."""
    for _ in range(warmup): fn()
    out = []
    for _ in range(trials):
        torch.npu.synchronize(); t0 = time.perf_counter()
        for _ in range(rep): fn()
        torch.npu.synchronize(); out.append((time.perf_counter() - t0) / rep * 1e6)
    return min(out)

def enqueue_us(fn, rep=30):
    """Host time alone: how long the launches take to return, before any sync."""
    for _ in range(10): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn()
    t = (time.perf_counter() - t0) / rep * 1e6
    torch.npu.synchronize()
    return t

H = 5120
# name -> (n_out, n_in), exactly as in the index. count = times per decode step.
LINEAR_ATTN = [("linear_attn.in_proj_qkv", 10240, H),
               ("linear_attn.in_proj_z",    6144, H),
               ("linear_attn.in_proj_a",      48, H),
               ("linear_attn.in_proj_b",      48, H),
               ("linear_attn.out_proj",        H, 6144)]
FULL_ATTN  = [("self_attn.q_proj", 12288, H),
              ("self_attn.k_proj",  1024, H),
              ("self_attn.v_proj",  1024, H),
              ("self_attn.o_proj",     H, 6144)]
MLP        = [("mlp.gate_proj", 17408, H), ("mlp.up_proj", 17408, H),
              ("mlp.down_proj", H, 17408)]
HEAD       = [("lm_head", 248320, H)]
N_LINEAR, N_FULL = 48, 16
PLAN = ([(s, N_LINEAR) for s in LINEAR_ATTN] + [(s, N_FULL) for s in FULL_ATTN]
        + [(s, N_LINEAR + N_FULL) for s in MLP] + [(s, 1) for s in HEAD])

# Matches ans_linear.MIN_BYTES: below this the op is host-bound and compressing
# is a straight loss, so those projections stay bf16 and are timed as bf16.
MIN_MIB = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(3)
print(f"Qwen3.8-27B decode step, M={M}  (48 linear-attn + 16 full-attn layers + lm_head)")
print(f"anything under {MIN_MIB:g} MiB stays bf16 -- the op is host-bound there\n")
print(f"  {'proj':<24}{'x':>4}{'N':>8}{'K':>7}{'MB bf16':>9}{'MB ans':>8}{'wide%':>7}"
      f"{'bf16 us':>10}{'ans us':>9}{'host us':>9}{'ratio':>8}")
tb = ta = mb = ma = th = 0.0
for (name, n_out, n_in), count in PLAN:
    w = bf(rng.normal(0, 0.02, n_out * n_in)).reshape(n_out, n_in)
    enc = encode_fixed_gemm(w.ravel(), n_out, n_in, max_bits=64.0); meta = enc.meta
    n_pad, k_pad = meta["n_pad"], meta["k_pad"]
    wpad = np.zeros((n_pad, k_pad), np.uint16); wpad[:n_out, :n_in] = w
    x = bf(rng.normal(0, 1.0, M * k_pad)).reshape(M, k_pad)
    xt = torch.from_numpy(to_f32(x)).to(torch.bfloat16).to(dev)
    wtT = torch.from_numpy(to_f32(wpad)).to(torch.bfloat16).to(dev).t().contiguous()
    args = (u16(enc.lplane), u16(enc.eplane), u16(enc.dplane), u16(enc.sbase),
            i32(enc.wide_group if enc.wide_group.size else np.zeros(8, np.int32)),
            u16(enc.wide_l, 16), u16(enc.wide_exp, 16), i32(meta["wide_tile_offsets"]))
    t_mm = bench(lambda: torch.matmul(xt, wtT))
    ans = lambda: torch.ops.rans.ans_fixed_decode_gemm(xt, *args, n_pad, k_pad, 0)
    t_ans = bench(ans)
    t_host = enqueue_us(ans)
    b_mb, a_mb = n_pad * k_pad * 2 / 2**20, enc.payload_bytes / 2**20
    wide = 100.0 * enc.wide_group.size / max(enc.n_groups, 1)
    kept = b_mb >= MIN_MIB          # compressed; below the threshold it stays bf16
    tb += t_mm * count
    ta += (t_ans if kept else t_mm) * count
    mb += b_mb * count
    ma += (a_mb if kept else b_mb) * count
    th += (t_host if kept else 0.0) * count
    print(f"  {name:<24}{count:>4}{n_out:>8}{n_in:>7}{b_mb:>9.0f}{a_mb:>8.0f}{wide:>7.3f}"
          f"{t_mm:>10.1f}{t_ans:>9.1f}{t_host:>9.1f}"
          f"{(f'{t_ans / t_mm:.2f}x' if kept else 'bf16'):>8}")
    del wtT, args, w, wpad, enc
    torch.npu.empty_cache()
print(f"\n  whole decode step{'':<19}{mb / 1024:>8.1f}G{ma / 1024:>7.1f}G{'':>7}"
      f"{tb / 1e3:>9.1f}m{ta / 1e3:>8.1f}m{th / 1e3:>8.1f}m{ta / tb:>7.2f}x")
print(f"\n  weights (linears only): bf16 {mb / 1024:.1f} GiB -> ans {ma / 1024:.1f} GiB"
      f"  ({mb / ma:.3f}x smaller, {16 / (mb / ma):.2f} bits/weight)")
print(f"  host launch time alone: {th / 1e3:.1f} ms of the {ta / 1e3:.1f} ms\n  projected tok/s from linears alone: bf16 {1e6 / tb:.1f} -> ans {1e6 / ta:.1f}")
