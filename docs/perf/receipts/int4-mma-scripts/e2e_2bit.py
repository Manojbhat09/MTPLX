"""End-to-end decode TPS baseline for the 2-bit 27B on this machine."""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/medusa-1tb/models/qwen38-27b-2bit/kexjos"
prompt = sys.argv[2] if len(sys.argv) > 2 else "Explain the theory of general relativity in detail:"
gen_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

print(f"loading {model_path} ...")
t0 = time.perf_counter()
model, tokenizer = load(model_path)
print(f"load: {time.perf_counter() - t0:.1f}s")

cache = make_prompt_cache(model)
ids = tokenizer.encode(prompt)
t0 = time.perf_counter()
logits = model(mx.array([ids]), cache=cache)
mx.eval(logits)
prefill_t = time.perf_counter() - t0
print(f"prefill {len(ids)} tokens: {prefill_t*1e3:.1f} ms ({len(ids)/prefill_t:.0f} tok/s)")

tokens = []
t0 = time.perf_counter()
for _ in range(gen_tokens):
    nid = mx.argmax(logits[:, -1, :], axis=-1)
    tokens.append(int(nid))
    logits = model(nid[:, None], cache=cache)
    mx.eval(logits)
dec_t = time.perf_counter() - t0
tps = gen_tokens / dec_t
text = tokenizer.decode(tokens)
print(f"decode: {tps:.2f} tok/s over {gen_tokens} tokens")
print("sample:", text[:200].replace("\n", " "))
