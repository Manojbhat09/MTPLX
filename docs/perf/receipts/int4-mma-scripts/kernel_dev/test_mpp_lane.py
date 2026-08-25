import os
import sys

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.gpu)
sys.path.insert(0, "/Users/dyson/challenges/MTPLX")
os.environ["MTPLX_INT4_MMA"] = "vocab,mpp"

import mtplx.kernels.int4_simd_mma as mod  # noqa: E402
from mtplx.kernels.int4_simd_mma import (  # noqa: E402
    install_int4_mma_qlinear_patch,
    mma_dispatch_counter_snapshot,
    mpp_available,
    mpp_vocab_eligible,
)

print("hardware gate on this M4:", mod._mpp_hardware_available(), "| mpp_available:", mpp_available())
print("eligibility k=5120 n=248320 g64 m=8:", mpp_vocab_eligible(8, 5120, 248320, 4, 64, mx.bfloat16))
print("eligibility ng>128 guard k=16384 g64:", mpp_vocab_eligible(8, 16384, 6400, 4, 64, mx.bfloat16))

print(mod.install_int4_mma_qlinear_patch())

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

# 1) Natural path on this machine: hardware gate False -> straight to vocab lane.
x = (mx.random.normal((8, k), key=mx.random.key(8)) * 0.5).astype(mx.bfloat16)
ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
y1 = layer(x)
mx.eval(y1, ref)
c1 = mma_dispatch_counter_snapshot()
print("gate-off path: dmax", float(mx.abs(y1 - ref).max()), "| vocab count", c1.get("vocab", 0), "| mpp count", c1.get("mpp_vocab", 0))

# 2) Simulate G17/26.4: force gate True -> mpp attempted -> compile fails -> cached -> vocab fallback.
mod._mpp_hardware_available.cache_clear()
mod._mpp_hardware_available.__wrapped__  # noqa: B018
orig = mod._mpp_hardware_available
mod._mpp_hardware_available = lambda: True  # type: ignore[assignment]
mod._MPP_STATE["available"] = None
try:
    y2 = layer(x)
    mx.eval(y2)
    c2 = mma_dispatch_counter_snapshot()
    print("forced-gate path: dmax vs stock", float(mx.abs(y2 - ref).max()), "| vocab delta", c2.get("vocab", 0) - c1.get("vocab", 0), "| mpp count", c2.get("mpp_vocab", 0))
    print("mpp_available after failure cache:", mpp_available())

    # 3) Second call must skip mpp entirely (cached failure).
    y3 = layer(x)
    mx.eval(y3)
    c3 = mma_dispatch_counter_snapshot()
    print("second call: vocab delta", c3.get("vocab", 0) - c2.get("vocab", 0))
finally:
    mod._mpp_hardware_available = orig  # type: ignore[assignment]
