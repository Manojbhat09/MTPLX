"""Exactness gate + microbench for int4_simd_mma kernels vs stock quantized_matmul."""
import sys
import time

import mlx.core as mx

sys.path.insert(0, "/Users/dyson/challenges/MTPLX")
from mtplx.kernels.int4_simd_mma import int4_prefill_qmm, int4_vocab_qmm  # noqa: E402

mx.set_default_device(mx.gpu)


def make(m, k, n, bits, gs, dtype=mx.bfloat16, seed=0):
    wq = mx.random.uniform(0, 2**32 - 1, (n, k // 8), key=mx.random.key(seed)).astype(mx.uint32)
    scales = (mx.random.normal((n, k // gs), key=mx.random.key(seed + 1)) * 0.02).astype(dtype)
    biases = (mx.random.normal((n, k // gs), key=mx.random.key(seed + 2)) * 0.02).astype(dtype)
    x = (mx.random.normal((m, k), key=mx.random.key(seed + 3)) * 0.5).astype(dtype)
    return x, wq, scales, biases


def stock(x, wq, s, b, gs):
    return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=4)


def dmax(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    mx.eval()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


fail = 0
print("=== exactness: vocab lane (N=6400 slice of real shape) ===")
for m in (1, 3, 4, 5, 8, 12, 16):
    for gs in (32, 64):
        x, wq, s, b = make(m, 512, 6400, 4, gs, seed=m * 7 + gs)
        y_ref = stock(x, wq, s, b, gs)
        y_new = int4_vocab_qmm(x, wq, s, b, group_size=gs)
        mx.eval(y_ref, y_new)
        d = dmax(y_ref, y_new)
        tol = max(0.05, 0.02 * float(mx.abs(y_ref).max()))
        ok = d <= tol
        fail += not ok
        print(f"m={m:3d} gs={gs}  dmax={d:.5f} tol={tol:.4f}  {'OK' if ok else 'FAIL'}")

print("=== exactness: prefill lane ===")
for m in (128, 512):
    for (k, n) in ((512, 34816), (6144, 5120)):
        gs = 32
        x, wq, s, b = make(m, k, n, 4, gs, seed=m + k)
        y_ref = stock(x, wq, s, b, gs)
        y_new = int4_prefill_qmm(x, wq, s, b, group_size=gs)
        mx.eval(y_ref, y_new)
        d = dmax(y_ref, y_new)
        ok = d <= 0.05
        fail += not ok
        print(f"m={m} k={k} n={n}  dmax={d:.5f}  {'OK' if ok else 'FAIL'}")

if fail:
    print(f"\n{fail} EXACTNESS FAILURES — skip bench")
    sys.exit(1)

print("\n=== bench: vocab projection N=248320 K=5120 g64 4bit ===")
for m in (1, 4, 8, 16):
    x, wq, s, b = make(m, 5120, 248320, 4, 64, seed=m)
    ms_stock = bench(lambda: stock(x, wq, s, b, 64))
    ms_new = bench(lambda: int4_vocab_qmm(x, wq, s, b, group_size=64))
    gb = (248320 * 5120 / 2 + 248320 * 80 * 2 * 2) / 1e9
    print(f"m={m:3d}  stock {ms_stock:8.2f} ms ({gb/ms_stock*1e3:6.1f} GB/s)   mma {ms_new:8.2f} ms ({gb/ms_new*1e3:6.1f} GB/s)   speedup {ms_stock/ms_new:5.2f}x")

print("\n=== bench: prefill 4bit g32 ===")
for m in (512, 1024, 2048):
    for (k, n, tag) in ((5120, 34816, "gate_up"), (5120, 8192, "qkv"), (6144, 5120, "o_proj"), (17408, 5120, "down")):
        gs = 32
        x, wq, s, b = make(m, k, n, 4, gs, seed=m + k)
        ms_stock = bench(lambda: stock(x, wq, s, b, gs), iters=20)
        ms_new = bench(lambda: int4_prefill_qmm(x, wq, s, b, group_size=gs), iters=20)
        print(f"m={m:5d} {tag:8s} K={k:5d} N={n:6d}  stock {ms_stock:8.2f} ms  mma {ms_new:8.2f} ms  speedup {ms_stock/ms_new:5.2f}x")
