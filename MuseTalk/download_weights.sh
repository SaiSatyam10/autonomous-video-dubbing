#!/usr/bin/env bash

set -euo pipefail

CHECKPOINTS_DIR="${1:-models}"

mkdir -p \
  "${CHECKPOINTS_DIR}/musetalk" \
  "${CHECKPOINTS_DIR}/musetalkV15" \
  "${CHECKPOINTS_DIR}/syncnet" \
  "${CHECKPOINTS_DIR}/dwpose" \
  "${CHECKPOINTS_DIR}/face-parse-bisent" \
  "${CHECKPOINTS_DIR}/sd-vae" \
  "${CHECKPOINTS_DIR}/whisper"

if ! command -v hf >/dev/null 2>&1; then
  python -m pip install -U "huggingface_hub[hf_xet]"
fi

download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  hf download "${repo_id}" "$@" --local-dir "${local_dir}"
}

echo "Downloading MuseTalk weights into ${CHECKPOINTS_DIR}"

# MuseTalk 1.0
download TMElyralab/MuseTalk "${CHECKPOINTS_DIR}" \
  "musetalk/musetalk.json" \
  "musetalk/pytorch_model.bin"

# MuseTalk 1.5
download TMElyralab/MuseTalk "${CHECKPOINTS_DIR}" \
  "musetalkV15/musetalk.json" \
  "musetalkV15/unet.pth"

# SD VAE
download stabilityai/sd-vae-ft-mse "${CHECKPOINTS_DIR}/sd-vae" \
  "config.json" \
  "diffusion_pytorch_model.bin"

# Whisper tiny
download openai/whisper-tiny "${CHECKPOINTS_DIR}/whisper" \
  "config.json" \
  "pytorch_model.bin" \
  "preprocessor_config.json"

# DWPose checkpoint used by the current MuseTalk codebase
download yzd-v/DWPose "${CHECKPOINTS_DIR}/dwpose" \
  "dw-ll_ucoco_384.pth"

# SyncNet
download ByteDance/LatentSync "${CHECKPOINTS_DIR}/syncnet" \
  "latentsync_syncnet.pt"

# Face parsing and ResNet backbone
download ManyOtherFunctions/face-parse-bisent "${CHECKPOINTS_DIR}/face-parse-bisent" \
  "79999_iter.pth" \
  "resnet18-5c106cde.pth"

required_files=(
  "${CHECKPOINTS_DIR}/musetalk/musetalk.json"
  "${CHECKPOINTS_DIR}/musetalk/pytorch_model.bin"
  "${CHECKPOINTS_DIR}/musetalkV15/musetalk.json"
  "${CHECKPOINTS_DIR}/musetalkV15/unet.pth"
  "${CHECKPOINTS_DIR}/sd-vae/config.json"
  "${CHECKPOINTS_DIR}/sd-vae/diffusion_pytorch_model.bin"
  "${CHECKPOINTS_DIR}/whisper/config.json"
  "${CHECKPOINTS_DIR}/whisper/pytorch_model.bin"
  "${CHECKPOINTS_DIR}/whisper/preprocessor_config.json"
  "${CHECKPOINTS_DIR}/dwpose/dw-ll_ucoco_384.pth"
  "${CHECKPOINTS_DIR}/syncnet/latentsync_syncnet.pt"
  "${CHECKPOINTS_DIR}/face-parse-bisent/79999_iter.pth"
  "${CHECKPOINTS_DIR}/face-parse-bisent/resnet18-5c106cde.pth"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -s "${file}" ]]; then
    echo "Missing or empty file: ${file}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Weight download completed with missing files." >&2
  exit 1
fi

echo "All MuseTalk weights downloaded successfully."
