import os
import sys

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.gpu)
sys.path.insert(0, "/Users/dyson/challenges/MTPLX")
os.environ["MTPLX_INT4_MMA"] = "all"

from mtplx.kernels.int4_simd_mma import (  # noqa: E402
    install_int4_mma_qlinear_patch,
    mma_dispatch_counter_snapshot,
    uninstall_int4_mma_qlinear_patch,
)

print(install_int4_mma_qlinear_patch())

k, n = 512, 6400
wq = mx.random.uniform(0, 2**32 - 1, (n, k // 8), key=mx.random.key(1)).astype(mx.uint32)
s = (mx.random.normal((n, k // 64), key=mx.random.key(2)) * 0.02).astype(mx.bfloat16)
b = (mx.random.normal((n, k // 64), key=mx.random.key(3)) * 0.02).astype(mx.bfloat16)

layer = nn.QuantizedLinear(k, n, bias=False)
layer.weight = wq
layer.scales = s
layer.biases = b
layer.bits = 4
layer.group_size = 64

for m in (8, 16):
    x = (mx.random.normal((m, k), key=mx.random.key(m)) * 0.5).astype(mx.bfloat16)
    y = layer(x)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y, ref)
    d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
    snap = {kk: vv for kk, vv in mma_dispatch_counter_snapshot().items() if kk.startswith("vocab_m") or kk == "vocab"}
    print(f"vocab lane m={m}: dmax={d:.5f} counters={snap}")

k2, n2 = 5120, 34816
wq2 = mx.random.uniform(0, 2**32 - 1, (n2, k2 // 8), key=mx.random.key(11)).astype(mx.uint32)
s2 = (mx.random.normal((n2, k2 // 32), key=mx.random.key(12)) * 0.02).astype(mx.bfloat16)
b2 = (mx.random.normal((n2, k2 // 32), key=mx.random.key(13)) * 0.02).astype(mx.bfloat16)
layer2 = nn.QuantizedLinear(k2, n2, bias=False)
layer2.weight = wq2
layer2.scales = s2
layer2.biases = b2
layer2.bits = 4
layer2.group_size = 32

x2 = (mx.random.normal((128, k2), key=mx.random.key(14)) * 0.5).astype(mx.bfloat16)
y2 = layer2(x2)
ref2 = mx.quantized_matmul(x2, wq2, s2, b2, transpose=True, group_size=32, bits=4)
mx.eval(y2, ref2)
print("prefill lane m=128: bit-exact =", bool(mx.array_equal(y2, ref2)), " counters:", {kk: vv for kk, vv in mma_dispatch_counter_snapshot().items() if kk == "prefill"})

uninstall_int4_mma_qlinear_patch()
x3 = (mx.random.normal((8, k), key=mx.random.key(21)) * 0.5).astype(mx.bfloat16)
before = mma_dispatch_counter_snapshot().get("vocab", 0)
_ = layer(x3)
after = mma_dispatch_counter_snapshot().get("vocab", 0)
print("after uninstall, vocab counter delta:", after - before, "(expect 0)")
