# syntax=docker/dockerfile:1.7

ARG WORKER_BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${WORKER_BASE_IMAGE} AS worker-common

WORKDIR /app

ARG APT_MIRROR_HOST=https://mirrors.tuna.tsinghua.edu.cn
ARG NODEJS_DIST_BASE_URL=https://npmmirror.com/mirrors/node
ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG HF_MIRROR_BASE_URL=https://hf-mirror.com

ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV HF_ENDPOINT=${HF_MIRROR_BASE_URL}
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

RUN mkdir -p /model-cache/amb3r /model-cache/lingbot /model-cache/speed3r_pi3 /model-cache/mast3r /model-cache/roma

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
    git ca-certificates curl gnupg xz-utils build-essential cmake ninja-build libeigen3-dev \
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
ARG ARTDECO_REPO_URL=https://github.com/InternRobotics/ARTDECO.git
ARG ARTDECO_REPO_COMMIT=bb654395826e50ac9e4671682d901377115a24ce
ARG SPEED3R_REPO_URL=https://github.com/Visual-AI/speed3r.git
ARG SPEED3R_REPO_COMMIT=5460f7309c87e5daac36385ff6611627de7d7267
ARG LINGBOT_MAP_REPO_URL=https://github.com/Robbyant/lingbot-map.git
ARG LINGBOT_MAP_REPO_COMMIT=4cd986009b9adeded8a4e740919221940dedeffe

ENV FINE_RECONSTRUCTION_ENABLED=true
ENV LMRS_ROOT=/opt/lm-rs
ENV ARTDECO_ROOT=/opt/artdeco-runtime
ENV SPEED3R_ROOT=/opt/speed3r-runtime
ENV MAX_JOBS=2
ENV CUDA_PATH=/usr/local/cuda
ENV NVTE_FRAMEWORK=pytorch

COPY backend/requirements.txt /tmp/backend/requirements.txt
COPY worker/requirements.txt /tmp/worker/requirements.txt
COPY worker/patches/lmrs-fastgs-compact-box.patch /tmp/lmrs-fastgs-compact-box.patch
COPY worker/patches/fastgs-cuda-metric-accumulation.patch /tmp/fastgs-cuda-metric-accumulation.patch
RUN <<'EOF'
set -eu
command -v nvcc
python - <<'PY'
from pathlib import Path
import torch
import torchvision

assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torchvision.__version__.startswith("0.23.0"), torchvision.__version__
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

RUN --mount=type=cache,id=three-dgs-worker-flashinfer-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --index-url https://pypi.org/simple flashinfer-python; then
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "FlashInfer install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

RUN --mount=type=cache,id=three-dgs-worker-amb3r-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --no-build-isolation --constraint /tmp/worker/constraints.txt \
      'spconv-cu118==2.3.8' \
      'torch-scatter==2.1.2' \
      -f https://data.pyg.org/whl/torch-2.8.0+cu128.html; then
    python - <<'PY'
import spconv.pytorch as spconv
import torch_scatter
print("AMB3R sparse dependencies import ok")
PY
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "AMB3R sparse dependency install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

RUN --mount=type=cache,id=three-dgs-worker-weight-download-cache,target=/root/.cache/three-dgs-downloads,sharing=locked <<'EOF'
set -eu
download_weight() {
  url="$1"
  target="$2"
  cached="/root/.cache/three-dgs-downloads/$(basename "$target")"
  if [ -s "$target" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$target")" "$(dirname "$cached")"
  if [ -s "$cached" ]; then
    cp "$cached" "$target"
    return 0
  fi
  tmp="${cached}.part"
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if curl -fL -C - --retry 5 --retry-delay 10 "$url" -o "$tmp" || curl -fL --retry 5 --retry-delay 10 "$url" -o "$tmp"; then
      mv "$tmp" "$cached"
      cp "$cached" "$target"
      return 0
    fi
    if [ "$attempt" -eq 5 ]; then
      return 1
    fi
    echo "RoMA weight download failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
    sleep $((attempt * 20))
    attempt=$((attempt + 1))
  done
}
download_weight "${HF_ENDPOINT}/xmanifold/roma/resolve/main/roma_outdoor.pth" /model-cache/roma/roma_outdoor.pth
download_weight "${HF_ENDPOINT}/xmanifold/roma/resolve/main/roma_indoor.pth" /model-cache/roma/roma_indoor.pth
download_weight "${HF_ENDPOINT}/Dominoc/dinov2_vitl14_pretrain/resolve/main/dinov2_vitl14_pretrain.pth" /model-cache/roma/dinov2_vitl14_pretrain.pth
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
    --mount=type=cache,id=three-dgs-worker-extension-wheel-cache,target=/root/.cache/three-dgs-wheels,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-lmrs-git-cache,target=/repo-cache/lm-rs,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-artdeco-git-cache,target=/repo-cache/artdeco,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-lingbot-map-git-cache,target=/repo-cache/lingbot-map,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-speed3r-git-cache,target=/repo-cache/speed3r,sharing=locked <<'EOF'
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
retry_git() {
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 "$@"; then
      return 0
    fi
    if [ "$attempt" -eq 5 ]; then
      return 1
    fi
    echo "git $1 failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
    sleep $((attempt * 20))
    attempt=$((attempt + 1))
  done
}
fingerprint_source() {
  source_dir="$1"
  (
    cd "$source_dir"
    find . \( -path './.git' -o -name build -o -name dist -o -name '*.egg-info' -o -name __pycache__ \) -prune -o -type f -print0 \
      | sort -z \
      | xargs -0 -r sha256sum
  ) | sha256sum | awk '{print $1}'
}
cached_wheel_install() {
  cache_name="$1"
  source_dir="$2"
  shift 2
  fingerprint="$(
    {
      fingerprint_source "$source_dir"
      printf '%s\n' "$*"
      sha256sum /tmp/worker/constraints.txt
      python - <<'PY'
import sys
import torch

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)
PY
      nvcc --version
      printf '%s\n' "${TORCH_CUDA_ARCH_LIST:-}"
    } | sha256sum | awk '{print $1}'
  )"
  wheel_dir="/root/.cache/three-dgs-wheels/${cache_name}/${fingerprint}"
  wheel="$(find "$wheel_dir" -maxdepth 1 -name '*.whl' -print -quit 2>/dev/null || true)"
  if [ -z "$wheel" ]; then
    mkdir -p "$wheel_dir"
    python -m pip wheel --constraint /tmp/worker/constraints.txt --wheel-dir "$wheel_dir" "$@" "$source_dir"
    wheel="$(find "$wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)"
  fi
  retry_pip --no-deps "$wheel"
}
ensure_git_checkout() {
  repo_url="$1"
  repo_path="$2"
  repo_commit="$3"
  if [ ! -d "$repo_path/.git" ]; then
    rm -rf "$repo_path"
    retry_git clone "$repo_url" "$repo_path"
  fi
  retry_git -C "$repo_path" remote set-url origin "$repo_url"
  if ! git -C "$repo_path" cat-file -e "$repo_commit^{commit}" >/dev/null 2>&1; then
    retry_git -C "$repo_path" fetch --all --tags --prune
  fi
  retry_git -C "$repo_path" checkout "$repo_commit"
}
CACHED_LMRS=/repo-cache/lm-rs/source
ensure_git_checkout "$LMRS_REPO_URL" "$CACHED_LMRS" "$LMRS_REPO_COMMIT"
retry_git -C "$CACHED_LMRS" submodule update --init --recursive submodules/simple-knn submodules/diff-gaussian-rasterization
retry_git -C "$CACHED_LMRS/submodules/diff-gaussian-rasterization" checkout "$LMRS_RASTERIZER_COMMIT"
retry_git -C "$CACHED_LMRS/submodules/diff-gaussian-rasterization" submodule update --init --recursive
LMRS_GLM="$CACHED_LMRS/submodules/diff-gaussian-rasterization/third_party/glm"
VENDORED_GLM=/app/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/diff-gaussian-rasterization/third_party/glm
if [ ! -f "$LMRS_GLM/glm/glm.hpp" ]; then
  rm -rf "$LMRS_GLM"
  mkdir -p "$(dirname "$LMRS_GLM")"
  cp -a "$VENDORED_GLM" "$LMRS_GLM"
fi
test -f "$LMRS_GLM/glm/glm.hpp"
rm -rf "$LMRS_ROOT"
mkdir -p "$(dirname "$LMRS_ROOT")"
cp -a "$CACHED_LMRS" "$LMRS_ROOT"
apply_rasterizer_patch() {
  patch_file="$1"
  if ! git -C "$LMRS_ROOT/submodules/diff-gaussian-rasterization" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    git -C "$LMRS_ROOT/submodules/diff-gaussian-rasterization" apply "$patch_file"
  fi
}
apply_rasterizer_patch /tmp/lmrs-fastgs-compact-box.patch
apply_rasterizer_patch /tmp/fastgs-cuda-metric-accumulation.patch
cached_wheel_install diff-gaussian-rasterization "$LMRS_ROOT/submodules/diff-gaussian-rasterization" --no-deps --no-build-isolation
cached_wheel_install simple-knn "$LMRS_ROOT/submodules/simple-knn" --no-deps --no-build-isolation
cached_wheel_install fused-ssim /app/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/fused-ssim --no-deps --no-build-isolation
CACHED_LINGBOT_MAP=/repo-cache/lingbot-map/source
ensure_git_checkout "$LINGBOT_MAP_REPO_URL" "$CACHED_LINGBOT_MAP" "$LINGBOT_MAP_REPO_COMMIT"
cached_wheel_install lingbot-map "$CACHED_LINGBOT_MAP" --no-deps
CACHED_ARTDECO=/repo-cache/artdeco/source
export GIT_LFS_SKIP_SMUDGE=1
ensure_git_checkout "$ARTDECO_REPO_URL" "$CACHED_ARTDECO" "$ARTDECO_REPO_COMMIT"
test -f /usr/include/eigen3/Eigen/Sparse
rm -rf "$ARTDECO_ROOT"
mkdir -p "$(dirname "$ARTDECO_ROOT")"
cp -a "$CACHED_ARTDECO" "$ARTDECO_ROOT"
sed -i 's/if not (torch.cuda.is_available() and CUDA_HOME):/if not CUDA_HOME:/' "$ARTDECO_ROOT/VSLAM/setup.py"
sed -i 's|str(ROOT / "thirdparty/eigen")|"/usr/include/eigen3"|' "$ARTDECO_ROOT/VSLAM/setup.py"
sed -i 's/D11.type()/D11.scalar_type()/' "$ARTDECO_ROOT/VSLAM/backend/src/matching_kernels.cu"
sed -i 's|torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {})|dx.pow(2).sum().sqrt()|g' "$ARTDECO_ROOT/VSLAM/backend/src/gn_kernels.cu"
cached_wheel_install artdeco-vslam "$ARTDECO_ROOT/VSLAM" --no-deps --no-build-isolation
CACHED_SPEED3R=/repo-cache/speed3r/source
ensure_git_checkout "$SPEED3R_REPO_URL" "$CACHED_SPEED3R" "$SPEED3R_REPO_COMMIT"
rm -rf "$SPEED3R_ROOT"
mkdir -p "$(dirname "$SPEED3R_ROOT")"
cp -a "$CACHED_SPEED3R" "$SPEED3R_ROOT"
find "$LMRS_ROOT" -name .git -prune -exec rm -rf {} +
find "$LMRS_ROOT" -name .git -type f -delete
find "$ARTDECO_ROOT" -name .git -prune -exec rm -rf {} +
find "$ARTDECO_ROOT" -name .git -type f -delete
find "$SPEED3R_ROOT" -name .git -prune -exec rm -rf {} +
find "$SPEED3R_ROOT" -name .git -type f -delete
EOF

CMD ["python", "-m", "app.worker_bootstrap"]
