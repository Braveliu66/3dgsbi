# syntax=docker/dockerfile:1.7

ARG WORKER_BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${WORKER_BASE_IMAGE} AS worker-common

WORKDIR /app

ARG APT_MIRROR_HOST=https://mirrors.tuna.tsinghua.edu.cn
ARG NODEJS_DIST_BASE_URL=https://npmmirror.com/mirrors/node
ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG HF_MIRROR_BASE_URL=https://hf-mirror.com
ARG COLMAP_REPO_URL=https://github.com/colmap/colmap.git
ARG COLMAP_REPO_COMMIT=4.0.4
ARG COLMAP_SOURCE_INSTALL=true
ARG COLMAP_CMAKE_CUDA_ARCHITECTURES="80;86;89;90"

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
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions
ENV GSPLAT_PRECOMPILED_MARKER=/opt/torch_extensions/.gsplat_precompiled

RUN mkdir -p /model-cache /repo-cache

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
    git ca-certificates curl gnupg xz-utils build-essential cmake ninja-build pkg-config \
    ccache \
    libglm-dev \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
    libopenimageio-dev openimageio-tools libopenexr-dev libopencv-dev libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev \
    libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libsuitesparse-dev \
    libcurl4-openssl-dev libssl-dev libopenblas-openmp-dev ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    if ! find /usr/local/cuda/include /usr/include /usr -path '*/cudnn.h' -print -quit 2>/dev/null | grep -q .; then \
      apt-get update && apt-get install -y --no-install-recommends libcudnn9-dev-cuda-12; \
    fi && \
    find /usr/local/cuda/include /usr/include /usr -path '*/cudnn.h' -print -quit 2>/dev/null | grep -q .

RUN --mount=type=cache,id=three-dgs-worker-colmap-git-cache,target=/repo-cache/colmap,sharing=locked \
  --mount=type=cache,id=three-dgs-worker-colmap-build-cache,target=/repo-cache/colmap/build,sharing=locked \
  --mount=type=cache,id=three-dgs-worker-ccache,target=/root/.cache/ccache,sharing=locked <<'EOF'
set -eu
export CCACHE_DIR=/root/.cache/ccache
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
  retry_git -C "$repo_path" submodule update --init --recursive
}
if [ "${COLMAP_SOURCE_INSTALL:-true}" = "true" ]; then
  colmap_src=/repo-cache/colmap/source
  colmap_build=/repo-cache/colmap/build
  ensure_git_checkout "$COLMAP_REPO_URL" "$colmap_src" "$COLMAP_REPO_COMMIT"
  cmake -S "$colmap_src" -B "$colmap_build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCUDA_ENABLED=ON \
    -DGUI_ENABLED=OFF \
    -DTESTS_ENABLED=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-DBOOST_BIND_GLOBAL_PLACEHOLDERS" \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_ARCHITECTURES="${COLMAP_CMAKE_CUDA_ARCHITECTURES}"
  cmake --build "$colmap_build" --target install
  ldconfig
fi
help_text="$(colmap help 2>&1 || true)"
printf '%s\n' "$help_text"
for command in feature_extractor exhaustive_matcher mapper image_undistorter point_triangulator global_mapper hierarchical_mapper model_analyzer model_clusterer model_splitter; do
  if ! printf '%s\n' "$help_text" | grep -Eq "(^|[[:space:]])${command}([[:space:]]|$)"; then
    echo "COLMAP source build missing required command: ${command}" >&2
    exit 1
  fi
done
EOF

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

ARG TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV SPARK_SPZ_CLI=/opt/preview-tools/spark_transcode_spz.mjs

FROM worker-common AS worker

ARG DASH_DEBLUR_GROUP_INSTALL_EXTENSIONS=true

ENV FINE_RECONSTRUCTION_ENABLED=true
ENV DASH_DEBLUR_GROUP_REPO=/opt/dash_deblur_group_gs
ENV MAX_JOBS=2
ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV NVTE_FRAMEWORK=pytorch

COPY backend/requirements.txt /tmp/backend/requirements.txt
COPY worker/requirements.txt /tmp/worker/requirements.txt
COPY worker/trainer/dash_deblur_group_gs /opt/dash_deblur_group_gs
RUN <<'EOF'
set -eu
torch_lib="$(
python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
test -f "$torch_lib/libc10.so"
printf '%s\n' "$torch_lib" > /etc/ld.so.conf.d/pytorch.conf
ldconfig
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

RUN --mount=type=cache,id=three-dgs-worker-gsplat-torch-extensions,target=/root/.cache/gsplat_torch_extensions,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-ccache,target=/root/.cache/ccache,sharing=locked <<'EOF'
set -eu
export CCACHE_DIR=/root/.cache/ccache
export TORCH_EXTENSIONS_DIR=/root/.cache/gsplat_torch_extensions
python - <<'PY'
from gsplat.cuda._backend import _C

assert _C is not None, "gsplat CUDA extension did not compile"
print("gsplat CUDA kernels precompiled")
PY
mkdir -p /opt/torch_extensions
cp -a /root/.cache/gsplat_torch_extensions/. /opt/torch_extensions/
printf 'TORCH_CUDA_ARCH_LIST=%s\n' "${TORCH_CUDA_ARCH_LIST:-}" > /opt/torch_extensions/.gsplat_precompiled
EOF

RUN --mount=type=cache,id=three-dgs-worker-runtime-extension-pip-cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-extension-wheel-cache,target=/root/.cache/three-dgs-wheels,sharing=locked \
    --mount=type=cache,id=three-dgs-worker-ccache,target=/root/.cache/ccache,sharing=locked <<'EOF'
set -eu
export CCACHE_DIR=/root/.cache/ccache
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
    echo "extension wheel cache miss: ${cache_name} ${fingerprint}"
    mkdir -p "$wheel_dir"
    python -m pip wheel --constraint /tmp/worker/constraints.txt --wheel-dir "$wheel_dir" "$@" "$source_dir"
    wheel="$(find "$wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)"
  else
    echo "extension wheel cache hit: ${cache_name} ${fingerprint}"
  fi
  retry_pip --force-reinstall --no-deps "$wheel"
}
install_dash_deblur_group_extension() {
  package_name="$1"
  extension_dir="$2"
  if [ ! -d "$extension_dir" ]; then
    echo "DashDeblurGroupGS extension source missing: $extension_dir" >&2
    exit 1
  fi
  if [ "$package_name" = "diff-gaussian-rasterization" ] && [ ! -f "$extension_dir/third_party/glm/glm/glm.hpp" ]; then
    if [ ! -f /usr/include/glm/glm.hpp ]; then
      echo "GLM headers missing; run git submodule update --init --recursive or install libglm-dev" >&2
      exit 1
    fi
    mkdir -p "$extension_dir/third_party/glm"
    ln -sfn /usr/include/glm "$extension_dir/third_party/glm/glm"
  fi
  if [ "$package_name" = "simple-knn" ] && [ -f "$extension_dir/simple_knn.cu" ] && ! grep -q "#include <float.h>" "$extension_dir/simple_knn.cu"; then
    sed -i '1i #include <float.h>' "$extension_dir/simple_knn.cu"
  fi
  cached_wheel_install "$package_name" "$extension_dir" --no-build-isolation
}
test -f "$DASH_DEBLUR_GROUP_REPO/train.py"
if [ "${DASH_DEBLUR_GROUP_INSTALL_EXTENSIONS:-true}" = "true" ]; then
  install_dash_deblur_group_extension diff-gaussian-rasterization "$DASH_DEBLUR_GROUP_REPO/submodules/diff-gaussian-rasterization"
  install_dash_deblur_group_extension simple-knn "$DASH_DEBLUR_GROUP_REPO/submodules/simple-knn"
  python - <<'PY'
import importlib

for module_name in ("diff_gaussian_rasterization", "simple_knn._C"):
    importlib.import_module(module_name)
print("DashDeblurGroupGS CUDA extension imports ok")
PY
fi
EOF

COPY backend/app ./app
COPY backend/blur_detection_no_percent_adaptive_canny.py ./blur_detection_no_percent_adaptive_canny.py

CMD ["python", "-m", "app.worker_bootstrap"]
