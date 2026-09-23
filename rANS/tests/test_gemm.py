"""Fused decode+GEMM: y = x @ W^T with W arriving as a fixed-rate payload.

The reference is the DECODED weight, not the original float: the tiles test
already proves the decode is bit-exact, so any difference here is the Cube's or
the handshake's, not the codec's.
"""
import sys, numpy as np, torch, torch_npu
torch.ops.load_library(sys.argv[1])
torch.npu.set_device(0); dev = "npu:0"
sys.path.insert(0, ".")
from codec.fixed_codec import encode_fixed_gemm, GEMM_N0, GEMM_K0

def bf(a):
    f = np.ascontiguousarray(a, dtype=np.float32); u = f.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def to_f32(u16):
    return (np.ascontiguousarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def u16(a, minel=0):
    a = np.ascontiguousarray(a, dtype=np.uint16)
    if a.size < minel: a = np.zeros(minel, dtype=np.uint16)
    return torch.from_numpy(a.view(np.int16)).to(dev)
def i32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

rng = np.random.default_rng(7); ok = True
CASES = [(1, 64, 128), (1, 256, 512), (16, 64, 128), (128, 256, 256),
         (1, 1024, 1024), (32, 512, 1024), (128, 1024, 2048), (7, 320, 768),
         (1, 5120, 5120), (256, 2048, 2048)]
print(f"tile = {GEMM_N0} x {GEMM_K0}\n")
print(f"  {'M':>5} {'N':>6} {'K':>6} {'tiles':>10} {'wide':>7}  {'err/tol':>9}  result")
for m, n_out, n_in, in CASES:
    w = bf(rng.normal(0, 0.02, n_out * n_in)).reshape(n_out, n_in)
    enc = encode_fixed_gemm(w.ravel(), n_out, n_in, max_bits=64.0); meta = enc.meta
    n_pad, k_pad = meta["n_pad"], meta["k_pad"]
    # exact padded weight the kernel will hand the Cube
    wpad = np.zeros((n_pad, k_pad), dtype=np.uint16); wpad[:n_out, :n_in] = w
    x = bf(rng.normal(0, 1.0, m * k_pad)).reshape(m, k_pad)
    x[:, n_in:] = 0
    xt = torch.from_numpy(to_f32(x)).to(torch.bfloat16).to(dev)
    wt = torch.from_numpy(to_f32(wpad)).to(torch.bfloat16).to(dev)
    ref = torch.matmul(xt, wt.t())          # same dtype + fp32 accumulate as the kernel

    args = (u16(enc.lplane), u16(enc.eplane), u16(enc.dplane), u16(enc.sbase),
            i32(enc.wide_group if enc.wide_group.size else np.zeros(8, np.int32)),
            u16(enc.wide_l, 16), u16(enc.wide_exp, 16), i32(meta["wide_tile_offsets"]))
    # Three back-to-back launches: a stale cross-core flag left by the previous
    # call would desync the ping-pong and show up on the 2nd or 3rd, not the 1st.
    # Criterion. Two things separate "different" from "wrong" here:
    #
    #  * a ONE-ULP difference in the bf16 output is a rounding tie. Checked
    #    against an fp64 dot product: |ours - truth| == |torch - truth| exactly
    #    on every such element.
    #  * a dot product that CANCELS resolves below the fp32 accumulator's
    #    noise. On 256x2048x2048 seven results of magnitude ~1e-5 come out of
    #    rows whose RMS scale is ~41 -- six orders of cancellation. Absolute
    #    error there is ~1e-7 for both, and ours is the closer of the two on
    #    three of the seven. Summation order, not arithmetic.
    #
    # So bound the difference by the accumulator's resolution for that dot
    # product, ||x[r]|| * ||w[c]||, and not by the result's own magnitude. A
    # genuinely wrong tile would be off by O(rms), seven orders above this.
    scale = np.outer(np.linalg.norm(to_f32(x).astype(np.float64), axis=1),
                     np.linalg.norm(to_f32(wpad).astype(np.float64), axis=1))
    rf = ref.float().cpu().numpy()
    ulp = np.ldexp(1.0, np.maximum(np.frexp(np.abs(rf))[1], -126) - 8)
    tol = ulp + 32 * 2.0**-24 * scale
    worst, bad = 0.0, 0
    for _ in range(3):
        y = torch.ops.rans.ans_fixed_decode_gemm(xt, *args, n_pad, k_pad)
        torch.npu.synchronize()
        d = np.abs(y.float().cpu().numpy() - rf)
        bad = max(bad, int((d > tol).sum()))
        worst = max(worst, float((d / tol).max()))
    good = bad == 0
    ok &= good
    print(f"  {m:5d} {n_out:6d} {n_in:6d} {meta['tiles_n']:4d}x{meta['tiles_k']:<5d} "
          f"{meta['n_wide']:7d}  {worst:9.2f}  {'PASS' if good else f'FAIL {bad} elems'}")
print(f"\n[status] {'PASS' if ok else 'FAIL'}")
