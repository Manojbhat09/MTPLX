# Qwen3.8-27B on a base M4 (16 GB): worklog, hardware observations, results

Date: 2026-08-25
Companion code: branch `perf/int4-mma-lanes`
(`mtplx/kernels/int4_simd_mma.py`); detailed kernel receipts in
`docs/perf/receipts/int4-mma-vocab-prefill-g13plus.md`.
Scripts: `docs/perf/receipts/int4-mma-scripts/`.

## Machine profile

- Mac mini, Apple M4 (base), 16 GB unified memory, macOS 26.2 (25C56)
- GPU `applegpu_g16g`, GPU family 9; Metal toolchain
  (GPUCompiler 32023.850) — **pre-26.4 SDK: no packed-numeric format
  types**, so native int4b TensorOps cannot compile here
- Effective bandwidth ~100 GB/s streaming weights; Metal per-buffer cap
  measured at **9.53 GB** (single allocation above this fails hard)
- MLX 0.32.0 / mlx-lm 0.31.3 via miniconda (`/Users/dyson/miniconda`)

## Environment gotchas learned the hard way

1. **MLX `metal_kernel(grid=...)` takes TOTAL THREADS**, not threadgroups.
   A grid of `(m_tiles, n_tiles)` runs one thread per tile — kernels appear
   to execute but only tid 0 does any work, producing deterministic-looking
   garbage that changes when unrelated code edits change register
   allocation. Correct repo-wide convention:
   `grid=(threads_per_tg * tg_count_x, tg_count_y, 1)` and derive tile
   counts inside MSL from scalar size args, never from `grid_size`.
2. **MLX compiles custom kernels lazily.** A compile failure surfaces at
   the *caller's next* `mx.eval`, outside any try/except around the kernel
   call. Any optional fast path needs an eager one-time probe that caches
   availability for process life.
3. **Metal single-buffer cap (~9.5 GB):** materializing full-sequence vocab
   logits `(ctx x 248320)` fails at ctx≈16K outright, and thrashes RAM well
   below that. Slice hidden states before lm_head.
4. Stale `HF_TOKEN` env breaks the `hf` CLI (401). Workaround:
   `env -u HF_TOKEN -u HF_OWNER hf download ...` or anonymous curl.
5. Internal disk nearly full (~12 GiB); all model artifacts live on
   `/Volumes/medusa-1tb`.

## Artifacts on /Volumes/medusa-1tb

| path | what |
|------|------|
| `models/qwen38-27b-2bit/kexjos` | keXjos/Qwen3.8-27B-mlx-2Bit, 2-bit g64 affine, text-only (7.9 GB) |
| `models/qwen38-27b-2bit/majentik` | majentik/Qwen3.8-27B-MLX-2bit, 2-bit g32 + bf16 vision tower (10 GB) |
| `models/qwen38-mtplx-drafter` | Youssofal mtp.safetensors DFlash2 drafter, W4/G64 (239 MB) + mtplx_runtime.json |

majentik loads (11.0 GB active) but generation OOMs even text-only on
16 GB; it needs >=24 GB. The vision tower itself is bf16 matmuls (the INT4
recipe does not apply inside it); image tokens enter the trunk as one
large-M prefill batch, squarely in the prefill lane's M>=128 regime.

## Kernel results (base M4)

Real W4/G64 drafter weights vs stock `mx.quantized_matmul`:

- prefill lane: **bit-exact** (`mx.array_equal`) everywhere; flat
  1.06-1.09x across all four trunk shapes from M=128 through M=8192 rows.
  No crossover, no cliff; speedup independent of context.
- vocab lane: bf16-rounding exact (same distance to fp32 ground truth as
  stock); **1.47x @ m=8, 1.25x @ m=16** (up_proj N=17408);
  synthetic N=248320 g64: 1.28x @ m=8, 1.22x @ m=16.
  Stock GEMV wins below M=8 (98.6 GB/s at m=1) -> lane gated to M>=8.

Native MPP lane: control flow verified (gate/probe/fallback/counters);
numerics unverified until a G17-class host runs the probe.

## Long-context achievement (decode >= 10 tok/s)

Baseline: naive whole-sequence forward collapsed to 1.25 tok/s at 4K and
hard-failed at 16K (12.9 GB logits alloc > Metal buffer cap).

Recipe (no model edits):
1. backbone-only chunked prefill (`model.language_model.model`),
   lm_head applied to last position only,
2. 2048-token chunks,
3. rotating 2048-token KV window on the 16 full-attention layers only
   (GDN linear layers keep constant-size global state).

| ctx | ms/step | decode | peak GB |
|-------|---------|--------|---------|
| 4096 | 98 | 10.24 | 11.9 |
| 32768 | 94 | 10.67 | 11.9 |
| 65536 | 97 | 10.35 | 11.9 |
| 131072 | 90 | **11.11** | 11.9 |

Step time is FLAT vs context once windowed (KV reads were the marginal
+1.1 ms per 1K tokens).

### Recall validation (needle tests)

- needle 600 tokens back (inside 2048 window): **HIT**
- needle 2600 back with window: miss
- needle 2600 back with FULL UNWINDOWED KV: **also miss**
- needle 6000 back, full KV: miss

Conclusion: long-range retrieval is already broken in this 2-bit
checkpoint regardless of attention history — the rotating window costs
nothing measurable in quality here.

### Negative results

- `QuantizedKVCache` (8-bit and 4-bit) is SLOWER than fp16 KV at every
  size tested under mlx-lm 0.31.3 — eager dequant overhead exceeds the
  bandwidth saved.
- Quantized KV also did not reduce peak memory enough to matter once the
  window was in place.
- The vocab MMA lane loses to stock GEMV at m<=4 (0.59-0.64x); eligibility
  gate starts at m=8.
- majentik VL cannot generate on 16 GB (OOM before first token).

## End-to-end baselines on this machine

- kexjos 2-bit, plain decode, short ctx: **10.62 tok/s**
  (prior llama.cpp GGUF attempts: 6.4-7.4 tok/s)
- prefill throughput ~58 tok/s regardless of chunk size — GDN chunked
  kernel-bound; ingesting a 128K prompt costs ~37 minutes
- PR #335 reference point: 113.25 tok/s decode headline on its >=24 GB
  benchmark rig (different hardware class entirely)

## Where the wins would land upstream

The MTPLX optimized checkpoint keeps lm_head at 8-bit g64, so the 4-bit
vocab lane applies to trunk layers during M=8/16 verify rounds rather than
lm_head; projected e2e decode uplift there is ~+25-35% at verify batch 8.
True end-to-end numbers need the big rig running the PR harness with
`MTPLX_INT4_MMA=all`. On G17 + macOS 26.4 hosts the opt-in 'mpp' lane adds
the native packed-INT4 path (upstream measured +2.1% decode isolated).
