"""Long-ctx runner: backbone-only chunked prefill + last-position lm_head.

Avoids the (ctx, 248320) full-vocab logits tensor that capped ctx at ~4-16K.
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_path = "/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos"
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

model, tokenizer = load(model_path)
lm = model.language_model

base_ids = tokenizer.encode(
    "The theory of general relativity describes gravity as the curvature of "
    "spacetime produced by mass and energy. Light bends around massive objects, "
    "clocks run slower in stronger gravitational fields, and the universe expands. "
)[5:]

GEN = 24


def run_ctx(target: int) -> bool:
    ids = (base_ids * (target // len(base_ids) + 1))[:target]
    try:
        cache = make_prompt_cache(model)
        mx.clear_cache()
        mx.reset_peak_memory()

        t0 = time.perf_counter()
        for i in range(0, len(ids), CHUNK):
            piece = mx.array([ids[i : i + CHUNK]])
            h = lm.model(piece, cache)
            mx.eval(h)
        last = h[:, -1:, :]
        logits = lm.lm_head(last)
        mx.eval(logits)
        pf = time.perf_counter() - t0

        n_out = 0
        t0 = time.perf_counter()
        for _ in range(GEN):
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            _ = int(nid)
            logits = model(nid[:, None], cache=cache)
            mx.eval(logits)
            n_out += 1
        dec = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 1e9
        active = mx.get_active_memory() / 1e9
        ok = n_out / dec >= 10.0
        print(
            f"ctx={len(ids):6d} chunk={CHUNK}: prefill {len(ids)/pf:6.0f} tok/s | "
            f"decode {n_out/dec:5.2f} tok/s | peak {peak:5.1f} GB active {active:5.1f} GB "
            f"{'OK' if ok else '<10'}",
            flush=True,
        )
        return ok
    except Exception as e:
        print(f"ctx={target} chunk={CHUNK}: FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
        return False
    finally:
        mx.clear_cache()


for tgt in (8192, 16384, 32768, 65536, 98304):
    if not run_ctx(tgt):
        break
