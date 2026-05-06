# Changelog

All notable user-facing changes are recorded here.

## v0.1.5

### Added

- Added explicit Sustained, Sustained Max, and Burst mode semantics across the CLI, onboarding wizard, quickstart paths, docs, and browser UI.
- Added long-context Sustained runtime telemetry for paged-attention routing, dense-fallback avoidance, large-query fallback behavior, and phase-aware prefill/decode diagnostics.
- Added opt-in Apple Silicon long-context QA coverage for Sustained memory and fallback regression checks.

### Fixed

- Fixed the 16K-32K long-context memory balloon by using chunked prefill, final-token logits, request-sized paged KV, dynamic paged-cache growth, and oversized SessionBank snapshot protection.
- Fixed normal Sustained long-context prefill from silently materializing dense full K/V state after the paged threshold.
- Fixed the 16K Sustained TPS regression by routing large-query paged attention through bounded paths instead of the old dense fallback.
- Fixed stale mode wording in help/docs and surfaced the selected runtime mode in `/health` and the browser chat UI.

### Release Notes

- Sustained is the default long-context native-MTP user path. Sustained Max adds explicit fan boost. Burst remains the old performance-cold max-fan headline lane for short prompts and benchmarks.
- Real QA showed the 32K Sustained path staying below the 35 GiB hard guard and the 16K Sustained Max decode gap recovering to within the release budget.
- This release does not claim the v0.2 no-fan long-response decay target or a proven 200K-token production ceiling.

## v0.1.4

### Fixed

- Fixed streaming completions hanging after visible generation finished by committing token-safe SessionBank state from the generation final state, and by moving unsafe postcommit work to an idle-only fallback that does not block the client.
- Fixed false web UI stall aborts during long active generations by adding server-side SSE progress heartbeats and heartbeat-aware browser status handling.
- Fixed the local install/release naming problem by moving from the preview/rc package line to stable `0.1.4` / `v0.1.4` naming.

### Release Notes

- Issue #7 and issue #8 are the user-visible fixes in this release.
- No sampler, decode-loop, MTP acceptance, kernel, or model-weight behavior changed for this release.
- Sustained no-fan long-context throughput remains the v0.2 performance track; v0.1.4 fixes serving liveness and release packaging, not the thermal/decay target.

## v0.1.0-preview.3

### Fixed

- Corrected the package and CLI version constants so fresh installs report `mtplx 0.1.0-preview.3 (0.1.0rc3)`. Preview 3 supersedes Preview 2, whose artifacts contained the OpenClaw and WebUI fixes but still printed the Preview 1 version string.

## v0.1.0-preview.2

### Added

- Added OpenAI-compatible tool-call support for agent clients such as OpenClaw: MTPLX now accepts `tools` / `tool_choice`, feeds tool schemas into the Qwen chat template, returns structured `message.tool_calls`, streams `delta.tool_calls`, and preserves tool-result history across turns.
- Added target-only AR switching without unloading the runtime: use `--no-mtp`, `/mtp off` in terminal chat, `"generation_mode":"ar"` in API requests, or the browser chat MTP toggle to compare against native-MTP generation.

### Fixed

- Fixed agent clients printing raw Qwen `<tool_call>` markup instead of executing tools.
- Fixed malformed generated tool-call markup leaking to clients; MTPLX now returns an explicit protocol error.

## v0.1.0-preview.1

### Added

- Added `install_preview_global.sh` to the private GitHub release path so the preview wheel installs into a durable `~/.mtplx/preview-venv` and exposes a normal global `mtplx` launcher.

### Fixed

- Added `mtplx help` as a first-class alias for `mtplx --help`.
- Added nested help aliases such as `mtplx help run` and `mtplx help qa exactness`.

## v0.1.0-preview

### Added

- Lazy package imports so `import mtplx` does not import MLX.
- No-MLX-safe `mtplx --help`, `doctor`, `inspect`, and `init` surface.
- Fresh-venv wheel smoke script for the Phase 0 install gate.
- Public benchmark dry-run paths that do not import heavy runtime modules.
- Packaged OpenAI server entrypoint with API-key guard, rate-limit knob, stream interval, warmup metadata, `/health`, `/metrics`, and `/v1/models` fake-state tests.
- No-MLX-safe `mtplx max` thermal-control surface with explicit ThermalForge/TG Pro detection and opt-in `--max` wiring.
- Baseline Anthropic `/v1/messages` translator, including non-stream responses and `stream=true` SSE events.

### Known Caveats

- Sustained no-fan long-context throughput is below the 50+ tok/s target.
- `performance-cold` is opt-in and may require the MTPLX MLX fork.
- The curated release repository is private-first until QA passes.

### Roadmap

- v0.2: kernel ladder for sustained no-fan throughput.
- v0.3: additional MTP architectures and broader serving polish.
