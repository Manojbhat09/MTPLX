# Fix: async SessionBank commit for tool-call responses

## Summary

`_schedule_idle_postcommit_snapshot` was scaffolded but never implemented. When the generation-final compatibility check rejects a response as unsafe to commit (typically because it contained `tool_calls`), control falls through to this function. The unpatched implementation just logs `"abandoned_foreground_busy"` and returns without ever calling `bank.put`. Result: any session whose responses include tool calls (every modern coding agent: opencode, Claude Code, Codex, Aider) **never** gets committed to the SessionBank, even when its session id is stable. Every turn pays full cold prefill.

This patch fills in the function: poll for the foreground to clear (bounded by a 30s deadline), then run the existing `_store_retokenized_history_snapshot` synchronously inside the background executor. That function already canonicalises the conversation (including `assistant_tool_calls`) into the exact prefix the next request will send, so the commit is byte-for-byte safe - it just had no caller.

## Reproduction (before this patch)

Run any tool-using OpenAI-compatible client against `mtplx --port 8088` with a stable `x-mtplx-session-id` header:

```bash
# opencode example
opencode run --model mtplx/qwen36-27b ...
```

Watch `GET /admin/sessions` and `GET /metrics`:

- per-session `bytes` stays at `0`
- `last_cache_miss_reason` stays `"new_session"`
- `boundaries[].bank_token_hash` is `null`, `nbytes` is `0` for every snapshot
- per-turn `cached_tokens` is `0` even after dozens of turns
- TTFT scales with full context length, not the delta

## Repro after this patch

Same workload, same monitoring:

- per-session `bytes > 0` after the first response stream completes
- `cached_tokens` grows monotonically with `context_len` from turn 2 onward
- TTFT drops to delta-prefill cost (single digits in seconds) as the cache extends

## Empirical numbers from a live opencode session

Seven-turn `researcher` subagent sequence on a 27B Qwen model:

| turn | ctx     | cached  | hit % | ttft   |
|------|---------|---------|-------|--------|
| 1    | 6,562   | 0       | 0%    | 22 s   |
| 2    | 13,961  | 0       | 0%    | 29 s   |
| 3    | 20,495  | 13,961  | 68%   | 33 s   |
| 4    | 22,479  | 0       | 0%    | 86 s   |
| 5    | 23,186  | 22,479  | **97%** | 42 s |
| 6    | 24,254  | 23,186  | 96%   | 45 s   |
| 7    | 24,956  | 23,186  | 93%   | 50 s   |

Compare to a same-length session under unpatched 0.1.6 (same hardware, same workload), which never showed `cached > 0` and held TTFT at 80-100 s for context > 30K. The remaining `cached=0` rows after the fix (turns 2 and 4) are postcommit-lag artifacts: the next turn arrived before the async commit acquired the model lock. They self-heal once a quiet moment lands. The lag is a candidate for a follow-up optimisation (start the commit during stream tail rather than after lock release) but is not a regression vs. today.

## Behavioural contract (preserved)

The original "never extend stream latency for a postcommit" contract is preserved:

1. The synchronous `pending` return (`{"stored": False, "mode": "async_pending", "reason": <unsafe_reason>}`) is unchanged.
2. The async work runs in `state.postcommit_executor` (or `state.generation_executor` as fallback), exactly as before.
3. If the foreground stays busy past `_IDLE_POSTCOMMIT_MAX_WAIT_S = 30s`, the background commit logs `abandoned_foreground_busy` and returns. The model lock is never blocked indefinitely.
4. All exceptions from the inner commit are caught and logged as `async_error` so the executor never propagates faults.

## Risk

- **Low.** The fix calls existing code (`_store_retokenized_history_snapshot`) that already had a synchronous caller (the `inline` postcommit path). The only behavioural change is that the *async* path now invokes it instead of being a stub. Bank `put` semantics, eviction, and per-session caps are unchanged.
- The bounded wait + caught exceptions ensure no new failure modes for the request path itself; the worst the patched function can do is fail to commit (the prior behaviour for 100% of tool-call sessions).

## Tests

- Updated `tests/test_openai_bridge.py::test_idle_async_postcommit_returns_pending_and_dispatches_retokenized_commit` to assert the new contract (commit is attempted, not silently abandoned).
- Added `test_idle_async_postcommit_attempts_commit_for_tool_call_responses` - regression test for the tool-call case specifically. Verifies `assistant_tool_calls` propagates into `_store_retokenized_history_snapshot`.
- Added `test_idle_async_postcommit_abandons_when_foreground_stays_busy` - covers the deadline-abandon path so the bounded-wait guarantee is enforced.
- All existing `tests/test_session_bank.py` and `tests/test_server_openai.py` tests still pass.

## Files changed

- `mtplx/server/openai.py` — fill in `_schedule_idle_postcommit_snapshot` body (~67 lines added, 16 removed).
- `tests/test_openai_bridge.py` — replace one stale test, add two regression tests.
- `CHANGELOG.md` — add an "Unreleased" entry describing the fix.
