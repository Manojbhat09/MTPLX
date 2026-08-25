"""Prefill-lane A/B vs stock across M (== ctx tokens per chunk) up to 8K rows."""
import sys
import time

import mlx.core as mx

mx.set_default_device(mx.gpu)
sys.path.insert(0, "/Users/dyson/challenges/MTPLX")
from mtplx.kernels.int4_simd_mma import int4_prefill_qmm, prefill_mma_eligible  # noqa: E402

f = mx.load("/Volumes/medusa-1tb/models/qwen38-mtplx-drafter/mtp.safetensors")
shapes = [
    ("q_proj", "mtp.layers.0.self_attn.q_proj", 12288, 5120),
    ("up_proj", "mtp.layers.0.mlp.up_proj", 17408, 5120),
    ("down_proj", "mtp.layers.0.mlp.down_proj", 5120, 17408),
]


def bench(fn, iters, warmup):
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


print(f"{'shape':9s} {'m':>6s} {'exact':>6s} {'stock ms':>10s} {'mma ms':>10s} {'speedup':>8s}")
for name, base, n, k in shapes:
    wq = f[f"{base}.weight"]
    s = f[f"{base}.scales"]
    b = f[f"{base}.biases"]
    for m in (128, 256, 512, 1024, 2048, 4096, 8192):
        iters = 20 if m <= 1024 else (10 if m <= 2048 else 5)
        x = (mx.random.normal((m, k), key=mx.random.key(m + n)) * 0.3).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        new = int4_prefill_qmm(x, wq, s, b, group_size=64)
        mx.eval(ref, new)
        exact = bool(mx.array_equal(new, ref))
        del ref, new
        ms_ref = bench(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4), iters, 2)
        ms_new = bench(lambda: int4_prefill_qmm(x, wq, s, b, group_size=64), iters, 2)
        print(f"{name:9s} {m:6d} {str(exact):>6s} {ms_ref:10.1f} {ms_new:10.1f} {ms_ref/ms_new:7.2f}x")
        del x
