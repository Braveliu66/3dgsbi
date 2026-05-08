# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

ARG APT_MIRROR_HOST=https://mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,id=three-dgs-backend-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=three-dgs-backend-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
    -e "s|http://deb.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|https://deb.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|http://security.debian.org|${APT_MIRROR_HOST}|g" \
    -e "s|https://security.debian.org|${APT_MIRROR_HOST}|g" \
    {} + && \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    mkdir -p /var/cache/apt/archives/partial /var/lib/apt/lists/partial && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 \
    && curl --version >/dev/null

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,id=three-dgs-backend-pip-cache,target=/root/.cache/pip,sharing=locked <<'EOF'
set -eu
attempt=1
while [ "$attempt" -le 5 ]; do
  if python -m pip install --upgrade pip && python -m pip install -r requirements.txt; then
    exit 0
  fi
  if [ "$attempt" -eq 5 ]; then
    exit 1
  fi
  echo "backend pip install failed, retrying in $((attempt * 20)) seconds ($attempt/5)"
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
EOF

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
