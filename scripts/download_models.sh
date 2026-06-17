#!/usr/bin/env bash
#
# Download the v1 image stack into ~/models/sd/ (paths match config.py).
# Only the minimal fp16 files are fetched (~4.8 GB total).
#
# NOTE: HF serves large files via cas-bridge.xethub.hf.co (Xet CDN). If that host
# is blocked/throttled on your network the big files will stall at 0 bytes — run
# this from a network/VPN where it is reachable, or use a browser (URLs below).
#
# Usage:   bash scripts/download_models.sh
# Resume:  just re-run it (wget -c continues partial files).

set -euo pipefail
SD="$HOME/models/sd"
HF="${HF_ENDPOINT:-https://huggingface.co}"

dl () {  # repo  relpath  destdir
  local url="$HF/$1/resolve/main/$2"
  local out="$3/$2"
  mkdir -p "$(dirname "$out")"
  echo ">> $1 :: $2"
  wget -c -q --show-progress -O "$out" "$url"
}

echo "### DreamShaper V8 (fp16) -> $SD/dreamshaper-8"
R="Lykon/dreamshaper-8"; D="$SD/dreamshaper-8"
for f in \
  model_index.json \
  scheduler/scheduler_config.json \
  tokenizer/merges.txt tokenizer/special_tokens_map.json \
  tokenizer/tokenizer_config.json tokenizer/vocab.json \
  feature_extractor/preprocessor_config.json \
  text_encoder/config.json text_encoder/model.fp16.safetensors \
  unet/config.json unet/diffusion_pytorch_model.fp16.safetensors \
  vae/config.json vae/diffusion_pytorch_model.fp16.safetensors ; do
  dl "$R" "$f" "$D"
done

echo "### IP-Adapter (+ CLIP image encoder) -> $SD/ip-adapter"
R="h94/IP-Adapter"; D="$SD/ip-adapter"
for f in \
  models/ip-adapter_sd15.bin \
  models/image_encoder/config.json \
  models/image_encoder/model.safetensors ; do
  dl "$R" "$f" "$D"
done

echo "### LCM-LoRA (SD1.5) -> $SD/lcm-lora-sdv1-5"
R="latent-consistency/lcm-lora-sdv1-5"; D="$SD/lcm-lora-sdv1-5"
dl "$R" "pytorch_lora_weights.safetensors" "$D"

echo
echo "DONE. Sizes:"
du -sh "$SD"/*/
