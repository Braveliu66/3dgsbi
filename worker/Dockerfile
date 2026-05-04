# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS worker-base

WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG INSTALL_PREVIEW_ALGO_DEPS=true

ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV REPO_CACHE_DIR=/repo-cache
ENV MODEL_CACHE_DIR=/model-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
    git git-lfs ca-certificates curl wget gnupg build-essential cmake ninja-build \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && ln -sf /usr/bin/pip3 /usr/local/bin/pip

COPY backend/app/preview/tools/package*.json /opt/preview-tools/
COPY backend/app/preview/tools/spark_transcode_spz.mjs /opt/preview-tools/spark_transcode_spz.mjs
RUN --mount=type=cache,id=three-dgs-worker-npm-cache,target=/root/.npm,sharing=locked \
    cd /opt/preview-tools && npm ci --omit=dev

COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY worker/requirements-preview.txt /tmp/requirements-preview.txt
RUN --mount=type=cache,id=three-dgs-worker-pip-cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip && \
    pip install --index-url ${TORCH_INDEX_URL} torch torchvision && \
    pip install -r /tmp/backend-requirements.txt && \
    if [ "${INSTALL_PREVIEW_ALGO_DEPS}" = "true" ]; then pip install -r /tmp/requirements-preview.txt; fi

COPY backend/app ./app

RUN --mount=type=cache,id=three-dgs-worker-pip-cache,target=/root/.cache/pip,sharing=locked \
    if [ "${INSTALL_PREVIEW_ALGO_DEPS}" = "true" ]; then \
      pip install -e /app/app/preview/vendor/edgs/gaussian_splatting/submodules/diff-gaussian-rasterization && \
      pip install -e /app/app/preview/vendor/edgs/gaussian_splatting/submodules/simple-knn && \
      pip install -e /app/app/preview/vendor/edgs/gaussian_splatting/submodules/fused-ssim; \
    fi

ENV SPARK_SPZ_CLI=/opt/preview-tools/spark_transcode_spz.mjs

FROM worker-base AS worker-preview
CMD ["python", "-m", "app.worker"]

FROM worker-base AS worker-fine
ENV FINE_RECONSTRUCTION_ENABLED=false
CMD ["python", "-m", "app.worker"]
