#!/usr/bin/env bash
set -euo pipefail

# Optional maintainer helper. Preview runtime uses bundled code under
# backend/app/preview/vendor. Fine training uses the embedded trainer under
# worker/trainer/dash_deblur_group_gs; this script only refreshes upstream
# comparison copies for bundled preview code.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/repo-cache"
mkdir -p "$CACHE"

clone_checkout() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local path="$CACHE/$name"
  if [ ! -d "$path/.git" ]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch --all --tags
  git -C "$path" checkout "$commit"
}

clone_checkout "LiteVGGT-repo" "https://github.com/GarlicBa/LiteVGGT-repo.git" "4767c17f8b6f176bb751566e92f60eb885040033"
clone_checkout "spark" "https://github.com/sparkjsdev/spark.git" "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf"

echo "Optional repository cache is ready at $CACHE"
