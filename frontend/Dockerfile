# syntax=docker/dockerfile:1.7
FROM node:24-bookworm-slim

WORKDIR /app

ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG NEXT_PUBLIC_API_BASE_URL=
ARG NEXT_PUBLIC_UPLOAD_API_BASE_URL=http://127.0.0.1:8000
ARG INTERNAL_API_BASE_URL=http://backend:8000
ARG VIEWER_TARGET_FPS=90
ARG VIEWER_QUALITY_UP_FPS=105
ARG VIEWER_QUALITY_DOWN_FPS=90
ARG VIEWER_ADAPTIVE_QUALITY=true
ARG NEXT_BUILD=true

ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ENV NEXT_PUBLIC_UPLOAD_API_BASE_URL=${NEXT_PUBLIC_UPLOAD_API_BASE_URL}
ENV INTERNAL_API_BASE_URL=${INTERNAL_API_BASE_URL}
ENV VIEWER_TARGET_FPS=${VIEWER_TARGET_FPS}
ENV VIEWER_QUALITY_UP_FPS=${VIEWER_QUALITY_UP_FPS}
ENV VIEWER_QUALITY_DOWN_FPS=${VIEWER_QUALITY_DOWN_FPS}
ENV VIEWER_ADAPTIVE_QUALITY=${VIEWER_ADAPTIVE_QUALITY}

COPY package*.json ./
RUN --mount=type=cache,id=three-dgs-frontend-npm-cache,target=/root/.npm,sharing=locked \
    npm config set registry ${NPM_CONFIG_REGISTRY} && \
    npm config set replace-registry-host always && \
    npm ci --cache /root/.npm --prefer-offline
COPY . .
RUN if [ "${NEXT_BUILD}" = "true" ]; then npm run build; fi

EXPOSE 3000
CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
