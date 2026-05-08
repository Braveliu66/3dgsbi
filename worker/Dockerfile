# syntax=docker/dockerfile:1.7

ARG WORKER_BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
FROM ${WORKER_BASE_IMAGE} AS worker-common

WORKDIR /app

ARG APT_MIRROR_HOST=https://mirrors.tuna.tsinghua.edu.cn
ARG NODEJS_DIST_BASE_URL=https://npmmirror.com/mirrors/node
ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV REPO_CACHE_DIR=/repo-cache
ENV MODEL_CACHE_DIR=/model-cache
ENV TORCH_HOME=/model-cache/torch
ENV HF_HOME=/model-cache/huggingface
ENV HF_HUB_CACHE=/model-cache/huggingface/hub
ENV XDG_CACHE_HOME=/model-cache/xdg
ENV TORCH_EXTENSIONS_DIR=/model-cache/torch_extensions

RUN --mount=type=cache,id=three-dgs-worker-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
    -e "s|http://archive.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|https://archive.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|http://security.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|https://security.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|http://ports.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|https://ports.ubuntu.com|${APT_MIRROR_HOST}|g" \
    -e "s|http://deb.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|https://deb.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|http://security.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|https://security.debian.org|${APT_MIRROR_HOST}|g" \
    {} + && \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    mkdir -p /var/cache/apt/archives/partial /var/lib/apt/lists/partial && \
    apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl gnupg xz-utils build-essential cmake ninja-build \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    if ! find /usr/local/cuda/include /usr/include /usr -path '*/cudnn.h' -print -quit 2>/dev/null | grep -q .; then \
      apt-get update && apt-get install -y --no-install-recommends libcudnn9-dev-cuda-12; \
    fi && \
    find /usr/local/cuda/include /usr/include /usr -path '*/cudnn.h' -print -quit 2>/dev/null | grep -q .

RUN --mount=type=cache,id=three-dgs-worker-pip-bootstrap-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --upgrade pip; then
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "pip install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

COPY worker/constraints.txt /tmp/worker/constraints.txt

RUN <<'EOF'
set -eu
case "$(dpkg --print-architecture)" in
  amd64) node_arch=x64 ;;
  arm64) node_arch=arm64 ;;
  *) echo "unsupported Node.js architecture: $(dpkg --print-architecture)" >&2; exit 1 ;;
esac
node_tmp="$(mktemp -d)"
curl -fsSL "${NODEJS_DIST_BASE_URL}/latest-v22.x/SHASUMS256.txt" -o "$node_tmp/SHASUMS256.txt"
node_tar="$(awk "/linux-${node_arch}\\.tar\\.xz$/ {print \$2; exit}" "$node_tmp/SHASUMS256.txt")"
test -n "$node_tar"
curl -fsSL "${NODEJS_DIST_BASE_URL}/latest-v22.x/${node_tar}" -o "$node_tmp/$node_tar"
(cd "$node_tmp" && grep " ${node_tar}$" SHASUMS256.txt | sha256sum -c -)
tar -xJf "$node_tmp/$node_tar" -C /usr/local --strip-components=1
ln -sf /usr/local/bin/node /usr/local/bin/nodejs
rm -rf "$node_tmp"
node --version && npm --version
EOF

COPY backend/app/preview/tools/package*.json /opt/preview-tools/
COPY backend/app/preview/tools/spark_transcode_spz.mjs /opt/preview-tools/spark_transcode_spz.mjs
RUN --mount=type=cache,id=three-dgs-worker-npm-cache,target=/root/.npm,sharing=locked \
    cd /opt/preview-tools && \
    npm config set registry ${NPM_CONFIG_REGISTRY} && \
    npm config set replace-registry-host always && \
    npm ci --omit=dev --cache /root/.npm --prefer-offline

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV SPARK_SPZ_CLI=/opt/preview-tools/spark_transcode_spz.mjs

FROM worker-common AS worker

ARG LMRS_REPO_URL=https://github.com/hamzapehlivan/lm-rs.git
ARG LMRS_REPO_COMMIT=cb40c7c06c2a60f8314ce095ad7b4513fbb33319
ARG LMRS_RASTERIZER_COMMIT=c2529d3bb13bc38271710785c015a89d9d623237

ENV FINE_RECONSTRUCTION_ENABLED=true
ENV LMRS_ROOT=/opt/lm-rs
ENV MAX_JOBS=2
ENV CUDA_PATH=/usr/local/cuda
ENV NVTE_FRAMEWORK=pytorch

COPY backend/requirements.txt /tmp/backend/requirements.txt
COPY worker/requirements.txt /tmp/worker/requirements.txt
COPY worker/patches/lmrs-fastgs-compact-box.patch /tmp/lmrs-fastgs-compact-box.patch
RUN <<'EOF'
set -eu
command -v nvcc
python - <<'PY'
from pathlib import Path
import torch
import torchvision

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torchvision.__version__.startswith("0.22.1"), torchvision.__version__
headers = [
    Path("/usr/local/cuda/include/cudnn.h"),
    Path("/usr/include/cudnn.h"),
    Path("/usr/include/x86_64-linux-gnu/cudnn.h"),
]
if not any(path.exists() for path in headers):
    found = next(Path("/usr").glob("**/cudnn.h"), None)
    assert found is not None, "cudnn.h not found"
print("CUDA/cuDNN compile preflight ok")
PY
EOF

RUN --mount=type=cache,id=three-dgs-worker-transformer-engine-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
cudnn_include="$(
python - <<'PY'
from pathlib import Path
for path in (
    Path("/usr/local/cuda/include/cudnn.h"),
    Path("/usr/include/cudnn.h"),
    Path("/usr/include/x86_64-linux-gnu/cudnn.h"),
):
    if path.exists():
        print(path.parent)
        raise SystemExit(0)
found = next(Path("/usr").glob("**/cudnn.h"), None)
if found is None:
    raise SystemExit("cudnn.h not found")
print(found.parent)
PY
)"
case "$cudnn_include" in
  */include) export CUDNN_PATH="${cudnn_include%/include}" ;;
  *) export CUDNN_PATH="$(dirname "$cudnn_include")" ;;
esac
export CUDA_PATH=/usr/local/cuda
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS=1
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --no-build-isolation --constraint /tmp/worker/constraints.txt 'einops==0.8.0' 'transformer-engine[pytorch]==2.4.0'; then
    python - <<'PY'
import einops
import transformer_engine.pytorch as te
print("transformer_engine.pytorch import ok")
PY
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "transformer-engine install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

RUN --mount=type=cache,id=three-dgs-worker-runtime-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --no-build-isolation --constraint /tmp/worker/constraints.txt -r /tmp/worker/requirements.txt; then
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "worker runtime requirements install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

RUN --mount=type=cache,id=three-dgs-worker-pycolmap-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --constraint /tmp/worker/constraints.txt 'pycolmap==3.12.6'; then
    python - <<'PY'
import pycolmap
print("pycolmap import ok")
PY
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "pycolmap install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

COPY backend/app ./app

RUN --mount=type=cache,id=three-dgs-worker-runtime-extension-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-lmrs-git-cache,target=/repo-cache/lm-rs,sharing=locked <<'EOF'
set -eu
retry_pip() {
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if python -m pip install --constraint /tmp/worker/constraints.txt "$@"; then
      return 0
    fi
    if [ "$attempt" -eq 5 ]; then
      return 1
    fi
    echo "extension install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
    sleep $((attempt * 20))
    attempt=$((attempt + 1))
  done
}
CACHED_LMRS=/repo-cache/lm-rs/source
if [ ! -d "$CACHED_LMRS/.git" ]; then
  rm -rf "$CACHED_LMRS"
  git clone "$LMRS_REPO_URL" "$CACHED_LMRS"
fi
git -C "$CACHED_LMRS" fetch --all --tags --prune
git -C "$CACHED_LMRS" checkout "$LMRS_REPO_COMMIT"
git -C "$CACHED_LMRS" submodule update --init --recursive submodules/simple-knn submodules/diff-gaussian-rasterization
git -C "$CACHED_LMRS/submodules/diff-gaussian-rasterization" checkout "$LMRS_RASTERIZER_COMMIT"
git -C "$CACHED_LMRS/submodules/diff-gaussian-rasterization" submodule update --init --recursive
LMRS_GLM="$CACHED_LMRS/submodules/diff-gaussian-rasterization/third_party/glm"
VENDORED_GLM=/app/app/preview/vendor/edgs/gaussian_splatting/submodules/diff-gaussian-rasterization/third_party/glm
if [ ! -f "$LMRS_GLM/glm/glm.hpp" ]; then
  rm -rf "$LMRS_GLM"
  mkdir -p "$(dirname "$LMRS_GLM")"
  cp -a "$VENDORED_GLM" "$LMRS_GLM"
fi
test -f "$LMRS_GLM/glm/glm.hpp"
rm -rf "$LMRS_ROOT"
mkdir -p "$(dirname "$LMRS_ROOT")"
cp -a "$CACHED_LMRS" "$LMRS_ROOT"
git -C "$LMRS_ROOT/submodules/diff-gaussian-rasterization" apply /tmp/lmrs-fastgs-compact-box.patch
retry_pip --no-build-isolation "$LMRS_ROOT/submodules/diff-gaussian-rasterization"
retry_pip --no-build-isolation "$LMRS_ROOT/submodules/simple-knn"
retry_pip --no-build-isolation /app/app/preview/vendor/edgs/gaussian_splatting/submodules/fused-ssim
find "$LMRS_ROOT" -name .git -prune -exec rm -rf {} +
find "$LMRS_ROOT" -name .git -type f -delete
PYTHONPATH=/app:/app/app/preview/vendor/edgs:/app/app/preview/vendor/edgs/gaussian_splatting:/app/app/preview/vendor/edgs/roma:$LMRS_ROOT:/app/app/preview/vendor/litevggt \
  python - <<'PY'
import torch
import torchvision
import romatch
import pycolmap
import transformer_engine.pytorch as te
import diff_gaussian_rasterization
import simple_knn
import fused_ssim
from transformer_engine.common.recipe import DelayedScaling, Format
from source.trainer import EDGSTrainer
from romatch.utils.local_correlation import local_correlation
from vggt.models.vggt import VGGT
from scene import GaussianModel

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torchvision.__version__.startswith("0.22.1"), torchvision.__version__
ext = diff_gaussian_rasterization._C
for name in ("get_JTv", "get_Diag", "get_JTJv"):
    assert hasattr(ext, name), name
assert getattr(diff_gaussian_rasterization, "MOBILEGS_COMPACT_BOX", False)
x = torch.randn(1, 2, 2, 2)
local_correlation(x, x, 1, torch.zeros(1, 2, 2, 2), use_custom_corr=False)
import app.fine.runner
from app.fine.deblur_mlp import GTnet
print("unified worker runtime import ok")
PY
EOF

CMD ["python", "-m", "app.worker_bootstrap"]
