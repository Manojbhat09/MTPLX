#!/usr/bin/env bash
# Download the model artifacts used by the worklog.
#
# Usage:  MODELS_DIR=/Volumes/medusa-1tb/models bash download_models.sh
# Env:    MODELS_DIR (default /Volumes/medusa-1tb/models)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/Volumes/medusa-1tb/models}"
mkdir -p "$MODELS_DIR"

say() { printf '\n== %s\n' "$*"; }

# A stale HF_TOKEN breaks the hf CLI with 401s; unset it and download
# anonymously (all source repos are public).
HF_UNSET=(env -u HF_TOKEN -u HF_OWNER)

say "1/2 keXjos/Qwen3.8-27B-mlx-2Bit (~8 GB, text gen / long-ctx runs)"
if [[ -f "$MODELS_DIR/qwen38-27b-2bit/kexjos/model-00002-of-00002.safetensors" ]]; then
    echo "already present, skipping"
else
    "${HF_UNSET[@]}" hf download keXjos/Qwen3.8-27B-mlx-2Bit \
        --local-dir "$MODELS_DIR/qwen38-27b-2bit/kexjos"
fi

say "2/2 Youssofal drafter files (~240 MB, real W4/G64 kernel benches)"
if [[ -f "$MODELS_DIR/qwen38-mtplx-drafter/mtp.safetensors" ]]; then
    echo "already present, skipping"
else
    "${HF_UNSET[@]}" hf download Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed \
        mtp.safetensors mtplx_runtime.json \
        --local-dir "$MODELS_DIR/qwen38-mtplx-drafter"
fi

say "done"
du -sh "$MODELS_DIR/qwen38-27b-2bit/kexjos" "$MODELS_DIR/qwen38-mtplx-drafter"
cat <<'EOF'

Optional (needs >=24 GB machine, OOMs on 16 GB):
  env -u HF_TOKEN -u HF_OWNER hf download majentik/Qwen3.8-27B-MLX-2bit \
      --local-dir "$MODELS_DIR/qwen38-27b-2bit/majentik"
EOF
