"""Recall controls: needle inside-window vs full-KV baseline."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

model_path = "/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos"
model, tokenizer = load(model_path)
lm = model.language_model

filler_ids = tokenizer.encode(
    "The observatory sits on a ridge above the valley. Astronomers calibrate "
    "the mirrors each night before the dome opens. Weather stations record "
    "humidity and wind along the trail. "
)[5:]
NEEDLE = "The maintenance tunnel passcode is ZEBRA-PURPLE-4721."


def build(window):
    if window is None:
        return [
            ArraysCache(size=2) if l.is_linear else KVCache() for l in lm.model.layers
        ]
    return [
        ArraysCache(size=2) if l.is_linear else RotatingKVCache(max_size=window, keep=4)
        for l in lm.model.layers
    ]


def trial(target, back_tokens, window):
    """Place needle `back_tokens` before the question."""
    ids = filler_ids * (target // len(filler_ids) + 1)
    ids = ids[: max(target - back_tokens - 20, 0)]
    ids += tokenizer.encode(NEEDLE)[5:]
    while len(ids) < target:
        ids += filler_ids
    ids = ids[:target]
    q = "\nQuestion: What is the maintenance tunnel passcode?\nAnswer:"
    ids = ids + tokenizer.encode(q)[5:]

    cache = build(window)
    mx.clear_cache()
    for i in range(0, len(ids), 2048):
        h = lm.model(mx.array([ids[i : i + 2048]]), cache)
        mx.eval(h)
    logits = lm.lm_head(h[:, -1:, :])
    mx.eval(logits)
    out = []
    for _ in range(10):
        nid = int(mx.argmax(logits[:, -1, :], axis=-1))
        out.append(nid)
        logits = model(mx.array([[nid]]), cache=cache)
        mx.eval(logits)
    text = tokenizer.decode(out)
    hit = "ZEBRA" in text or "4721" in text or "PURPLE" in text
    wtag = "fullKV" if window is None else f"w{window}"
    print(
        f"ctx={len(ids):6d} {wtag:>7s} needle_{back_tokens}_back: {text[:40]!r} {'HIT' if hit else 'miss'}",
        flush=True,
    )


# inside 2048 window
trial(8192, 600, 2048)
# just outside window
trial(8192, 2600, 2048)
# full-KV control, same distance as the 'outside' case
trial(8192, 2600, None)
# full-KV control, deep
trial(8192, 6000, None)
