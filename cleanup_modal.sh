#!/usr/bin/env bash
set -euo pipefail

MODAL="$(command -v modal)"
APP="qwen-7b-service"
VOLUMES=("qwen-7b-weights-cache" "bge-small-embed-cache")

echo "==> Stopping Modal app: $APP"
$MODAL app stop "$APP" --yes
echo "    Stopped."

echo "==> Deleting volumes..."
for vol in "${VOLUMES[@]}"; do
    echo "    Deleting volume: $vol"
    $MODAL volume delete "$vol" --yes && echo "    Deleted." || echo "    Not found, skipping."
done

echo "==> Done. All Modal resources for '$APP' have been removed."
