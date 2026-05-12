#!/usr/bin/env bash
set -euo pipefail

MODAL=".venv/bin/modal"
APP="repomind-vllm"
VOLUMES=("huggingface-cache" "vllm-cache")

echo "==> Stopping Modal app: $APP"
$MODAL app stop "$APP" --yes
echo "    Stopped."

echo "==> Deleting volumes..."
for vol in "${VOLUMES[@]}"; do
    echo "    Deleting volume: $vol"
    $MODAL volume delete "$vol" --yes && echo "    Deleted." || echo "    Not found, skipping."
done

echo "==> Done. All Modal resources for '$APP' have been removed."
