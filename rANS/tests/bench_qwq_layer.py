"""One QwQ-32B decoder layer's linears: ANS fused vs bf16 matmul, device time.

No 62 GB model needed -- these are the real shapes, and at M=1 a decode step is
exactly these seven GEMMs per layer, 64 times.
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
def bench(fn, warmup=5, rep=30):
    for _ in range(warmup): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(rep): fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6

H, I, L = 5120, 27648, 64
LAYER = [("q_proj", 5120, H), ("k_proj", 1024, H), ("v_proj", 1024, H),
         ("o_proj", H, 5120), ("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(3)
print(f"QwQ-32B decoder layer, M={M}\n")
print(f"  {'proj':<11}{'N':>7}{'K':>7}{'MB bf16':>9}{'MB ans':>8}{'bf16 us':>10}{'ans us':>9}{'ratio':>8}")
tb = ta = mb = ma = 0.0
for name, n_out, n_in in LAYER:
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
    t_ans = bench(lambda: torch.ops.rans.ans_fixed_decode_gemm(xt, *args, n_pad, k_pad, 0))
    b_mb, a_mb = n_pad * k_pad * 2 / 2**20, enc.payload_bytes / 2**20
    tb += t_mm; ta += t_ans; mb += b_mb; ma += a_mb
    print(f"  {name:<11}{n_out:>7}{n_in:>7}{b_mb:>9.0f}{a_mb:>8.0f}{t_mm:>10.1f}{t_ans:>9.1f}"
          f"{t_ans / t_mm:>7.2f}x")
    del wtT, args
    torch.npu.empty_cache()
print(f"\n  per layer     {mb:>21.0f}{ma:>8.0f}{tb:>10.1f}{ta:>9.1f}{ta / tb:>7.2f}x")
print(f"  x{L} layers   {mb * L / 1024:>20.1f}G{ma * L / 1024:>7.1f}G{tb * L / 1e3:>9.1f}m{ta * L / 1e3:>8.1f}m")
print(f"\n  projected decode step (linears only): bf16 {tb * L / 1e3:.1f} ms -> ans {ta * L / 1e3:.1f} ms")
print(f"  projected tok/s from linears alone:   bf16 {1e3 / (tb * L / 1e3):.1f} -> ans {1e3 / (ta * L / 1e3):.1f}")
