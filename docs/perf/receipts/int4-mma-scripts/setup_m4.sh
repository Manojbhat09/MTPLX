#!/usr/bin/env bash
# Bootstrap any Apple-silicon Mac (target: base M4, 16 GB) to reproduce the
# MTPLX int4-mma lanes + long-context results.
#
# Usage:  bash setup_m4.sh
# Env overrides:
#   REPO_URL   git clone source   (default: https://github.com/Manojbhat09/MTPLX.git)
#   WORKDIR    where to clone     (default: ~/MTPLX)
#
# Layout created:
#   $WORKDIR        this repo on branch notes/m4-16gb-qwen38-worklog (docs + scripts)
#   $WORKDIR-code   worktree on branch perf/int4-mma-lanes           (kernel code)
#   $WORKDIR/.venv  python env with mlx + mlx-lm
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Manojbhat09/MTPLX.git}"
WORKDIR="${WORKDIR:-$HOME/MTPLX}"
CODE_BRANCH="perf/int4-mma-lanes"
NOTES_BRANCH="notes/m4-16gb-qwen38-worklog"

say() { printf '\n== %s\n' "$*"; }

say "0/5 host sanity"
[[ "$(uname -s)" == "Darwin" ]] || { echo "need macOS"; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || { echo "need Apple silicon"; exit 1; }
echo "macOS $(sw_vers -productVersion) | $(sysctl -n machdep.cpu.brand_string)"
echo "RAM: $(sysctl -n hw.memsize | awk '{print int($1/1073741824)" GB"}')"
echo "NOTE: results in this worklog were measured on macOS 26.x, 16 GB unified."

say "1/5 clone"
if [[ ! -d "$WORKDIR/.git" ]]; then
    git clone "$REPO_URL" "$WORKDIR"
fi
git -C "$WORKDIR" fetch origin "$CODE_BRANCH" "$NOTES_BRANCH"
git -C "$WORKDIR" checkout "$NOTES_BRANCH"

say "2/5 code worktree ($CODE_BRANCH)"
if [[ ! -d "$WORKDIR-code/.git" ]] && ! git -C "$WORKDIR" worktree list | grep -q "$WORKDIR-code"; then
    git -C "$WORKDIR" worktree add "$WORKDIR-code" "origin/$CODE_BRANCH"
fi
ls "$WORKDIR-code/mtplx/kernels/int4_simd_mma.py" >/dev/null && echo "kernel module present"

say "3/5 python env"
cd "$WORKDIR"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet "mlx>=0.32" "mlx-lm>=0.31" numpy
./.venv/bin/python - <<'PY'
import mlx.core as mx, mlx_lm
print("mlx", mx.__version__, "| mlx_lm", mlx_lm.__version__)
print("device:", mx.default_device(), "| arch:", mx.device_info().get("architecture"))
PY

say "4/5 disk space check (models need ~12 GB)"
cat > "$WORKDIR/.env.paths" <<EOF
export CODE_DIR="$WORKDIR-code"
export VENV_PY="$WORKDIR/.venv/bin/python"
EOF
echo "paths saved to $WORKDIR/.env.paths"

say "5/5 next steps"
cat <<'EOF'
  source "$PWD/.env.paths"
  MODELS_DIR="/Volumes/<your-drive>/models" bash docs/perf/receipts/int4-mma-scripts/download_models.sh
  bash docs/perf/receipts/int4-mma-scripts/verify_and_bench.sh

MODELS_DIR needs >= 15 GB free. Default expectation table lives in
docs/perf/receipts/RUN-ON-NEW-M4.md.
EOF
