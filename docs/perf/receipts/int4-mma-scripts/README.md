# int4-mma-scripts

Reproduction assets for the int4-MMA + long-context work (M4 16 GB).

- `kernel/int4_simd_mma.py` — the kernel module itself, mirrored here so this
  branch is self-contained. Canonical copy lives on branch
  `perf/int4-mma-lanes` at `mtplx/kernels/int4_simd_mma.py` (the PR branch);
  import from there when running benchmarks (`PYTHONPATH=<code-worktree>`).
- `kernel_dev/` — the development-time validation suites:
  - `bench_mma.py` — full exactness gate + synthetic benchmark (vocab lane
    M{1..16}, prefill lane real shapes)
  - `sweep_m.py` — prefill-lane A/B vs stock across M=128..8192 on real
    W4/G64 weights (flat ~1.08x, bit-exact)
  - `test_shim.py` — QuantizedLinear installer: routing, counters,
    uninstall restore
  - `test_mpp_lane.py` — native MPP lane gate/probe/fallback control flow
- top-level scripts — end-to-end and long-context experiments
  (`e2e_2bit`, `bench_real`, `kv_sweep`, `window_sweep`, `long_ctx`,
  `needle_controls`) plus the new-machine bootstrap trio
  (`setup_m4.sh`, `download_models.sh`, `verify_and_bench.sh`); see
  `../RUN-ON-NEW-M4.md`.

Paths inside the python scripts default to `/Volumes/medusa-1tb/models/...`;
override with `MODELS_DIR` (shell scripts) or edit the constants for a
different mount.
