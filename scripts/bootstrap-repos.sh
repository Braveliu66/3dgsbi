#!/usr/bin/env bash
set -euo pipefail

# Optional maintainer helper. Runtime does not use repo-cache; bundled preview
# code lives in backend/app/preview/vendor. This script only refreshes upstream
# comparison copies when updating the vendored implementation.

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
  if [ "$name" = "EDGS" ]; then
    git -C "$path" submodule update --init --recursive submodules/gaussian-splatting
  fi
}

clone_checkout "LiteVGGT-repo" "https://github.com/GarlicBa/LiteVGGT-repo.git" "4767c17f8b6f176bb751566e92f60eb885040033"
clone_checkout "EDGS" "https://github.com/CompVis/EDGS.git" "9a897645eb47c1b24d4f9e4428cd745927bf1ee1"
clone_checkout "lingbot-map" "https://github.com/Robbyant/lingbot-map.git" "f720b421c6c50af3adc63272033226aa4811ef42"
clone_checkout "spark" "https://github.com/sparkjsdev/spark.git" "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf"

echo "Optional repository cache is ready at $CACHE"
