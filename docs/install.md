# Install

See [INSTALL.md](../INSTALL.md) for the short path.

MTPLX v0.1-preview is Apple-Silicon-first:

- macOS 14.0 or newer
- native arm64 Python 3.10 or newer
- `python3 -m pip install mlx` in that same environment
- enough unified memory and disk for the selected model/profile, checked by `mtplx doctor`

The first-run default model is `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` and the first-run default profile is `performance-cold`. `stable` remains available as the conservative compatibility alias.

Do not install model weights into the source checkout. Use the MTPLX model cache or a Hugging Face cache.
