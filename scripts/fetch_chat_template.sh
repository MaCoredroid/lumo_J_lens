#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${CHAT_TEMPLATE:-$ROOT/configs/qwen3-openai-codex.jinja}
URL=${CHAT_TEMPLATE_URL:-https://raw.githubusercontent.com/MaCoredroid/Lumo_FlyWheel/9fd1b40287748c8b6b8a9075fc383f454a30b0e0/docker/chat_templates/qwen3-openai-codex.jinja}
EXPECTED_SHA256=${CHAT_TEMPLATE_SHA256:-c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da}

if [[ -f "$DEST" ]] && [[ $(sha256sum "$DEST" | awk '{print $1}') == "$EXPECTED_SHA256" ]]; then
  echo "chat template already verified: $DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$URL" -o "$tmp"
actual=$(sha256sum "$tmp" | awk '{print $1}')
[[ "$actual" == "$EXPECTED_SHA256" ]] || {
  echo "chat template SHA-256 mismatch: expected $EXPECTED_SHA256, got $actual" >&2
  exit 1
}
mv "$tmp" "$DEST"
echo "chat template fetched and verified: $DEST"
