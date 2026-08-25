import sys
import time

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)
sys.path.insert(0, "/Users/dyson/challenges/MTPLX")
from mtplx.kernels.int4_simd_mma import int4_prefill_qmm  # noqa: E402

f = mx.load("/Volumes/medusa-1tb/models/qwen38-mtplx-drafter/mtp.safetensors")

shapes = [
    ("q_proj", "mtp.layers.0.self_attn.q_proj", 12288, 5120),
    ("o_proj", "mtp.layers.0.self_attn.o_proj", 5120, 6144),
    ("up_proj", "mtp.layers.0.mlp.up_proj", 17408, 5120),
    ("down_proj", "mtp.layers.0.mlp.down_proj", 5120, 17408),
]


def bench(fn, iters=20, warmup=5):
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


print("=== real W4/G64 production weights: prefill lane ===")
for name, base, n, k in shapes:
    wq = f[f"{base}.weight"]
    s = f[f"{base}.scales"]
    b = f[f"{base}.biases"]
    for m in (128, 256):
        x = (mx.random.normal((m, k), key=mx.random.key(m + n)) * 0.3).astype(mx.bfloat16)
        ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        new = int4_prefill_qmm(x, wq, s, b, group_size=64)
        mx.eval(ref, new)
        bit_exact = bool(mx.array_equal(new, ref))
        dmax = float(mx.abs(new.astype(mx.float32) - ref.astype(mx.float32)).max())
        ms_ref = bench(lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
        ms_new = bench(lambda: int4_prefill_qmm(x, wq, s, b, group_size=64))
        print(
            f"{name:9s} ({n:5d}x{k:5d}) m={m:4d}: bit-exact={bit_exact} dmax={dmax:.6f}  "
            f"stock {ms_ref:8.2f} ms  mma {ms_new:8.2f} ms  speedup {ms_ref / ms_new:5.2f}x"
        )
