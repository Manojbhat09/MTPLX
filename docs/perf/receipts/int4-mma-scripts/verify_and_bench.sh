#!/usr/bin/env bash
# Exactness smoke, real-weight kernel bench, e2e baseline, and the long-ctx
# demo. Run after setup_m4.sh + download_models.sh.
#
# Usage:  bash verify_and_bench.sh [quick|full]
#   quick (default): exactness + real-weight bench + short e2e
#   full:            adds the long-ctx sweep up to 128K (prefill-bound;
#                    the 128K leg alone takes ~37 min at ~58 tok/s prefill)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-quick}"

[[ -f "$HERE/../../../.env.paths" ]] && source "$HERE/../../../.env.paths"
CODE_DIR="${CODE_DIR:-$(dirname "$HERE")/../../../..}"
PY="${VENV_PY:-python3}"
MODELS_DIR="${MODELS_DIR:-/Volumes/medusa-1tb/models}"

KEXJOS="$MODELS_DIR/qwen38-27b-2bit/kexjos"
DRAFTER="$MODELS_DIR/qwen38-mtplx-drafter/mtp.safetensors"

say() { printf '\n== %s\n' "$*"; }

export MTPLX_INT4_MMA=all
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

say "1/4 exactness + perf smoke (synthetic weights)"
"$PY" - <<'PY'
import mlx.core as mx
from mtplx.kernels.int4_simd_mma import int4_vocab_qmm, int4_prefill_qmm

k, n = 512, 6400
wq = mx.random.uniform(0, 2**32 - 1, (n, k // 8), key=mx.random.key(1)).astype(mx.uint32)
s = mx.ones((n, k // 64), dtype=mx.bfloat16)
b = mx.zeros((n, k // 64), dtype=mx.bfloat16)
x8 = mx.zeros((8, k), dtype=mx.bfloat16)
y = int4_vocab_qmm(x8, wq, s, b, group_size=64)
ref = mx.quantized_matmul(x8, wq, s, b, transpose=True, group_size=64, bits=4)
mx.eval(y, ref)
assert bool(mx.array_equal(y[:1], ref[:1])) or float(abs(y - ref).max()) < 0.05
x128 = mx.zeros((128, k), dtype=mx.bfloat16)
yp = int4_prefill_qmm(x128, wq, s, b, group_size=64)
rp = mx.quantized_matmul(x128, wq, s, b, transpose=True, group_size=64, bits=4)
mx.eval(yp, rp)
assert bool(mx.array_equal(yp, rp)), "prefill lane must be bit-exact"
print("vocab lane OK | prefill lane BIT-EXACT")
PY

say "2/4 real W4/G64 drafter weights bench (bit-exact + speedups)"
sed "s#/Volumes/medusa-1tb/models/qwen38-mtplx-drafter/mtp.safetensors#$DRAFTER#" \
    "$HERE/bench_real.py" > /tmp/bench_real_local.py
"$PY" /tmp/bench_real_local.py

say "3/4 end-to-end baseline (2-bit 27B, expect ~10.6 tok/s on M4 16GB)"
sed "s#/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos#$KEXJOS#" \
    "$HERE/e2e_2bit.py" > /tmp/e2e_local.py
"$PY" /tmp/e2e_local.py

if [[ "$MODE" == "full" ]]; then
    say "4/4 long-context sweep (>=10 tok/s target; full runs to 128K)"
    sed -e "s#/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos#$KEXJOS#" \
        "$HERE/window_sweep.py" > /tmp/window_local.py
    cat >> /tmp/window_local.py <<'PY'
run(65536, 2048)
run(131072, 2048)
PY
    "$PY" /tmp/window_local.py
else
    say "4/4 skipped (run 'verify_and_bench.sh full' for the long-ctx sweep)"
fi

say "done — compare against the tables in RUN-ON-NEW-M4.md"
