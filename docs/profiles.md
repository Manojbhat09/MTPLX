# Profiles

| Profile | Purpose |
|---|---|
| `performance-cold` | Default first-run profile. Preserves the verified 60 tok/s class for short/cold replies without fan control. |
| `stable` | Public conservative alias for predictable long replies and compatibility fallback. |
| `exact` | QA and release exactness checks. |
| `max-diagnostic` | Fan-controlled diagnostics only. Not a product headline mode. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.
