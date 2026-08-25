# Reproducing this work on any new M4-class Mac (16 GB)

Everything here was measured on: **Mac mini, base M4, 16 GB unified,
macOS 26.x, MLX 0.32 / mlx-lm 0.31.3**. The recipe is hardware-agnostic for
any Apple-silicon Mac; only the absolute numbers shift with RAM/bandwidth.

## What you get

1. `perf/int4-mma-lanes` — INT4 simdgroup-MMA kernels (vocab lane ~1.3–1.5×
   at m=8/16; prefill retile bit-exact 1.06–1.09× flat to M=8192) behind
   `MTPLX_INT4_MMA`, plus an opt-in native MPP packed-INT4 lane that
   auto-engages on G17-class GPUs with macOS ≥ 26.4.
2. Long-context serving of the 27B 2-bit model at **≥10 tok/s from 4K out
   to 128K** in ≤12 GB peak memory.

## Setup (three commands)

```bash
bash docs/perf/receipts/int4-mma-scripts/setup_m4.sh          # clone + venv + deps
MODELS_DIR=/Volumes/<your-drive>/models \
    bash docs/perf/receipts/int4-mma-scripts/download_models.sh   # ~8.3 GB
bash docs/perf/receipts/int4-mma-scripts/verify_and_bench.sh full
```

`setup_m4.sh` creates `$HOME/MTPLX` (notes branch: these docs) and
`$HOME/MTPLX-code` (worktree on the code branch), plus `.venv` with mlx +
mlx-lm. It saves both paths to `.env.paths`, which the other scripts source.
Any external drive works for models (`MODELS_DIR` needs ≥15 GB free);
the internal disk should not be used if it has <20 GB free.

## Expected results on a base M4 / 16 GB

| check | expected |
|---|---|
| vocab lane exactness | dmax ≤ 0.0625 bf16 vs stock |
| prefill lane exactness | **bit-exact** (`mx.array_equal`) |
| real W4/G64 prefill speedup | 1.04–1.10× at m ∈ {128,256} |
| e2e decode (2-bit kexjos, short ctx) | ~10.6 tok/s |
| long ctx (window=2048) | ≥10 tok/s at any ctx up to 128K; step time flat |
| peak memory, long ctx | ≤12 GB |

On an M4 Pro/Max the same steps run faster; on M5-class silicon with
macOS ≥ 26.4 the native `'mpp'` lane additionally engages (probe it:
`python -c "from mtplx.kernels.int4_mma import *"` after setting
`MTPLX_INT4_MMA=vocab,mpp` and calling `mpp_available()`).

## How the long-context trick works

mlx_lm's whole-sequence forward materializes logits for every position:
`(ctx × 248320)` — a single allocation that exceeds Metal's ~9.5 GB
per-buffer cap near 16K and thrashes RAM well below it. Fix:

```python
lm = model.language_model
for i in range(0, len(ids), 2048):                 # chunked backbone-only prefill
    h = lm.model(mx.array([ids[i:i+2048]]), cache); mx.eval(h)
logits = lm.lm_head(h[:, -1:, :])                  # logits for LAST position only
```

plus a rotating window on the 16 full-attention layers only:

```python
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
cache = [ArraysCache(size=2) if l.is_linear else RotatingKVCache(max_size=2048, keep=4)
         for l in lm.model.layers]
```

GDN linear-attention layers carry global state in constant-size arrays, so
step time stays flat as context grows (measured: +1.1 ms per 1K tokens
without the window; flat with it).

## Troubleshooting / known limits

- **401 from `hf download`**: a stale `HF_TOKEN` env breaks the CLI. Use
  `env -u HF_TOKEN -u HF_OWNER hf download ...` (all repos are public).
- **`[metal::malloc] ... greater than maximum allowed buffer size`**: some op
  tried a >9.5 GB single allocation — almost always full-vocab logits over
  many positions. Use the sliced-lm_head pattern above.
- **Custom kernel "works" but outputs garbage**: remember MLX
  `metal_kernel(grid=...)` takes TOTAL THREADS, not threadgroups, and MLX
  compiles custom kernels lazily (failures surface at the caller's next
  `mx.eval`). Both bit this project; see the worklog's "Environment
  gotchas" section.
- **Prefill is slow (~58 tok/s)** regardless of chunk size — GDN chunked
  kernel-bound in mlx-lm 0.31.3. A 128K prompt takes ~37 min to ingest.
- **QuantizedKVCache**: measured slower than fp16 KV under mlx-lm 0.31.3;
  don't bother.
- **majentik VL checkpoint**: needs ≥24 GB; OOMs on 16 GB even text-only.
- Close other GPU-heavy apps when running: peak sits at ~12 GB and macOS
  paging will silently halve your tok/s before OOM ever appears.
