import sys, numpy as np, torch, torch_npu
torch.ops.load_library(sys.argv[1])
torch.npu.set_device(0); dev = "npu:0"
sys.path.insert(0, ".")
from codec.fixed_codec import encode_fixed_gemm, GEMM_N0, GEMM_K0
def bf(a):
    f = np.ascontiguousarray(a, dtype=np.float32); u = f.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
def u16(a, minel=0):
    a = np.ascontiguousarray(a, dtype=np.uint16)
    if a.size < minel: a = np.zeros(minel, dtype=np.uint16)
    return torch.from_numpy(a.view(np.int16)).to(dev)
def i32(a): return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)
rng = np.random.default_rng(2026); ok = True
print(f"shipped ans_fixed_decode_tiles, tile = {GEMM_N0} x {GEMM_K0}\n")
print(f"  {'shape':>15}{'tiles':>12}{'wide':>7}  result")
SHAPES = [(1024, 5120, False), (512, 1024, True), (64, 128, False), (300, 700, False),
          (4096, 64, False), (3584, 4608, False), (1, 1, False), (65, 129, False),
          # QwQ-32B / Qwen3.5-27B projection shapes (both are hidden 5120)
          (5120, 5120, False), (1024, 5120, False), (27648, 5120, False), (5120, 27648, True),
          (17408, 5120, False), (5120, 17408, True)]
for n_out, n_in, spike in SHAPES:
    vals = rng.normal(0, 0.02, n_out * n_in)
    if spike: vals[::997] = 3e30
    w = bf(vals).reshape(n_out, n_in)
    enc = encode_fixed_gemm(w.ravel(), n_out, n_in, max_bits=64.0); m = enc.meta
    out = torch.empty(m["n_pad"], m["k_pad"], dtype=torch.bfloat16, device=dev)
    out.fill_(float("nan"))
    torch.ops.rans.ans_fixed_decode_tiles(
        u16(enc.lplane), u16(enc.eplane), u16(enc.dplane), u16(enc.sbase),
        i32(enc.wide_group if enc.wide_group.size else np.zeros(8, np.int32)),
        u16(enc.wide_l, 16), u16(enc.wide_exp, 16), i32(m["wide_tile_offsets"]), out)
    torch.npu.synchronize()
    got = out.view(torch.uint16).cpu().numpy()[:n_out, :n_in]
    bad = int(np.count_nonzero(got != w)); ok &= bad == 0
    print(f"  {n_out:6d}x{n_in:<8d}{m['tiles_n']:5d}x{m['tiles_k']:<6d}{m['n_wide']:7d}  "
          f"{'PASS' if bad == 0 else f'FAIL {bad}/{w.size}'}")
print(f"\n[status] {'PASS' if ok else 'FAIL'}")
