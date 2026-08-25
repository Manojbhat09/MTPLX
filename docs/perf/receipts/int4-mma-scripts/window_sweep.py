"""Rotating-window full-attn caches: hold decode >=10 tok/s at large ctx."""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

model_path = "/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos"
model, tokenizer = load(model_path)
lm = model.language_model
n_layers = len(lm.model.layers)

base_ids = tokenizer.encode(
    "The theory of general relativity describes gravity as the curvature of "
    "spacetime produced by mass and energy. Light bends around massive objects, "
    "clocks run slower in stronger fields, and the universe expands. "
)[5:]

GEN = 20


def build_window_cache(window):
    cache = []
    for idx, layer in enumerate(lm.model.layers):
        if layer.is_linear:
            cache.append(ArraysCache(size=2))
        else:
            cache.append(RotatingKVCache(max_size=window, keep=4))
    return cache


def run(target, window, chunk=2048):
    ids = (base_ids * (target // len(base_ids) + 1))[:target]
    cache = build_window_cache(window)
    mx.clear_cache()
    mx.reset_peak_memory()
    try:
        for i in range(0, len(ids), chunk):
            h = lm.model(mx.array([ids[i : i + chunk]]), cache)
            mx.eval(h)
        logits = lm.lm_head(h[:, -1:, :])
        mx.eval(logits)

        out_tokens = []
        for _ in range(GEN):
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            out_tokens.append(int(nid))
            logits = model(nid[:, None], cache=cache)
        mx.eval(logits)
        sample = tokenizer.decode(out_tokens[:GEN])

        t0 = time.perf_counter()
        for _ in range(GEN):
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            _ = int(nid)
            logits = model(nid[:, None], cache=cache)
            mx.eval(logits)
        ms = (time.perf_counter() - t0) / GEN * 1e3
        ok = 1000 / ms >= 10.0
        print(
            f"ctx={target:6d} window={window:5d}: {ms:6.1f} ms/step "
            f"({1000/ms:5.2f} tok/s) peak {mx.get_peak_memory()/1e9:5.1f} GB "
            f"{'OK' if ok else '<10'} | gen: {sample[:60]!r}",
            flush=True,
        )
        return ok
    except Exception as e:
        print(f"ctx={target} window={window}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False
    finally:
        mx.clear_cache()


run(4096, 2048)
run(32768, 2048)
run(32768, 4096)
