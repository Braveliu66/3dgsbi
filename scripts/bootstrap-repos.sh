#!/usr/bin/env bash
set -euo pipefail

# Optional maintainer helper. Runtime does not use repo-cache; bundled preview
# code lives in backend/app/preview/vendor. This script only refreshes upstream
# comparison copies when updating the vendored implementation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/repo-cache"
LMRS_COMPACT_BOX_PATCH="$ROOT/worker/patches/lmrs-fastgs-compact-box.patch"
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
  elif [ "$name" = "lm-rs" ]; then
    git -C "$path" submodule update --init --recursive submodules/simple-knn submodules/diff-gaussian-rasterization
    git -C "$path/submodules/diff-gaussian-rasterization" checkout c2529d3bb13bc38271710785c015a89d9d623237
    git -C "$path/submodules/diff-gaussian-rasterization" submodule update --init --recursive
    if ! git -C "$path/submodules/diff-gaussian-rasterization" apply --reverse --check "$LMRS_COMPACT_BOX_PATCH" >/dev/null 2>&1; then
      git -C "$path/submodules/diff-gaussian-rasterization" apply "$LMRS_COMPACT_BOX_PATCH"
    fi
  fi
}

clone_checkout "LiteVGGT-repo" "https://github.com/GarlicBa/LiteVGGT-repo.git" "4767c17f8b6f176bb751566e92f60eb885040033"
clone_checkout "EDGS" "https://github.com/CompVis/EDGS.git" "9a897645eb47c1b24d4f9e4428cd745927bf1ee1"
clone_checkout "lingbot-map" "https://github.com/Robbyant/lingbot-map.git" "f720b421c6c50af3adc63272033226aa4811ef42"
clone_checkout "spark" "https://github.com/sparkjsdev/spark.git" "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf"
clone_checkout "lm-rs" "https://github.com/hamzapehlivan/lm-rs.git" "cb40c7c06c2a60f8314ce095ad7b4513fbb33319"

echo "Optional repository cache is ready at $CACHE"
