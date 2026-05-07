# MTPLX v0.2.1

v0.2.1 is an emergency hotfix for the public server quickstart path.

It includes the full v0.2.0 Pi/client work: `mtplx start pi`, Pi config
generation, automatic Pi launch, live MTPLX server-console controls, and the
OpenAI tool-streaming fixes.

## Fixed

- `mtplx quickstart --max`, `mtplx serve --max`, and `mtplx start --max` now keep
  the v0.2 Sustained Max default even when an older `~/.mtplx/config.toml`
  contains `profile = "performance-cold"`.
- Non-Sustained MTP prefill above 16K prompt tokens now fails with a clear
  configuration error instead of taking the full hidden/logits path that can
  allocate hundreds of GB at 64K+ context.

## Immediate User Guidance

For long-context server benchmarks, use:

```bash
mtplx config set profile sustained
mtplx quickstart --profile sustained --max
```

`--profile performance-cold --max` remains available as the short-context Burst
lane, but it is not the long-context v0.2 Sustained prefill path.
