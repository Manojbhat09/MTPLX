"""Decode-step cost vs ctx, with plain / 8-bit / 4-bit quantized KV."""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, QuantizedKVCache

model_path = "/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos"
model, tokenizer = load(model_path)
lm = model.language_model

base_ids = tokenizer.encode(
    "The theory of general relativity describes gravity as the curvature of "
    "spacetime produced by mass and energy. Light bends around massive objects, "
    "clocks run slower in stronger fields, and the universe expands. "
)[5:]

GEN = 20


def build_cache(kv_bits):
    cache = model.make_cache()
    if kv_bits is not None:
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                cache[i] = QuantizedKVCache(group_size=64, bits=kv_bits)
    return cache


def run(target, kv_bits, chunk=2048):
    ids = (base_ids * (target // len(base_ids) + 1))[:target]
    cache = build_cache(kv_bits)
    mx.clear_cache()
    mx.reset_peak_memory()
    try:
        for i in range(0, len(ids), chunk):
            h = lm.model(mx.array([ids[i : i + chunk]]), cache)
            mx.eval(h)
        logits = lm.lm_head(h[:, -1:, :])
        mx.eval(logits)

        # warmup 3 then time GEN steps
        for _ in range(3):
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            _ = int(nid)
            logits = model(nid[:, None], cache=cache)
        mx.eval(logits)
        t0 = time.perf_counter()
        for _ in range(GEN):
            nid = mx.argmax(logits[:, -1, :], axis=-1)
            _ = int(nid)
            logits = model(nid[:, None], cache=cache)
            mx.eval(logits)
        ms = (time.perf_counter() - t0) / GEN * 1e3
        print(
            f"ctx={target:6d} kv={'fp16' if kv_bits is None else str(kv_bits)+'bit':>6s}: "
            f"{ms:7.1f} ms/step ({1000/ms:6.2f} tok/s) peak {mx.get_peak_memory()/1e9:5.1f} GB",
            flush=True,
        )
    except Exception as e:
        print(f"ctx={target} kv={kv_bits}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    finally:
        mx.clear_cache()


for ctx in (1024, 4096, 16384):
    for bits in (None, 8, 4):
        run(ctx, bits)
