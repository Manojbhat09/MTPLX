# Profiles

| Profile | Purpose |
|---|---|
| `performance-cold` | Medium mode: native-MTP speed path, about 2.2x burst over the same model with MTP off, not sustained without fan control. |
| `stable` | Hidden conservative alias for the exact/staged long-reply path and compatibility fallback. |
| `exact` | QA and release exactness checks. |
| `max-diagnostic` | Max-style fan-controlled diagnostics. Product Max is `performance-cold` plus explicit `--max`. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.
