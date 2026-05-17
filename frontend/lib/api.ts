import type {
  AdminProjectUsage,
  AdminUserUsage,
  AlgorithmEntry,
  Artifact,
  AuthResponse,
  FeedbackEntry,
  MediaAsset,
  PipelineParameterDefaultsResponse,
  PipelineParameterSchema,
  PipelineSceneType,
  Project,
  ProjectShareResponse,
  RuntimePreflight,
  SharedProject,
  Task,
  User,
  ViewerConfig
} from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
const TOKEN_KEY = "three_dgs_token";
const TRANSFER_API_BASE = process.env.NEXT_PUBLIC_UPLOAD_API_BASE_URL || API_BASE;
const UPLOAD_CHUNK_SIZE = readPositiveEnvInt(process.env.NEXT_PUBLIC_UPLOAD_CHUNK_SIZE, 16 * 1024 * 1024);
const UPLOAD_CONCURRENCY = readPositiveEnvInt(process.env.NEXT_PUBLIC_UPLOAD_CONCURRENCY, 6);
const RANGE_DOWNLOAD_CHUNK_SIZE = readPositiveEnvInt(process.env.NEXT_PUBLIC_RANGE_DOWNLOAD_CHUNK_SIZE, 8 * 1024 * 1024);
const RANGE_DOWNLOAD_CONCURRENCY = readPositiveEnvInt(process.env.NEXT_PUBLIC_RANGE_DOWNLOAD_CONCURRENCY, 6);
const CHUNK_RETRIES = 3;
const HASH_READ_SIZE = 4 * 1024 * 1024;

export type TransferPhase = "hashing" | "checking" | "uploading" | "completing" | "downloading" | "complete";

export interface TransferProgress {
  fileName: string;
  phase: TransferPhase;
  loadedBytes: number;
  totalBytes: number;
  percent: number;
}

export type TransferProgressCallback = (progress: TransferProgress) => void;

export class ApiError extends Error {
  constructor(message: string, public status: number, public statusText: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function isAuthError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

export function isPublicPath(pathname: string): boolean {
  return pathname === "/login" || pathname === "/about" || pathname.startsWith("/share/");
}

type ApiRequestInit = RequestInit & { auth?: boolean };

async function request<T>(path: string, init?: ApiRequestInit, base = API_BASE): Promise<T> {
  const { auth = true, ...fetchInit } = init ?? {};
  const token = auth ? getToken() : null;
  const response = await fetch(`${base}${path}`, {
    ...fetchInit,
    headers: {
      ...(fetchInit.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...fetchInit.headers
    },
    cache: "no-store"
  });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(readErrorMessage(text, `${response.status} ${response.statusText}`), response.status, response.statusText);
  }
  return (await response.json()) as T;
}

async function requestText(path: string, init?: ApiRequestInit, base = API_BASE): Promise<string> {
  const { auth = true, ...fetchInit } = init ?? {};
  const token = auth ? getToken() : null;
  const response = await fetch(`${base}${path}`, {
    ...fetchInit,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...fetchInit.headers
    },
    cache: "no-store"
  });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(readErrorMessage(text, `${response.status} ${response.statusText}`), response.status, response.statusText);
  }
  return response.text();
}

function readErrorMessage(text: string, fallback: string): string {
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    return text;
  }
  return text;
}

interface UploadCheckResponse {
  upload_id: string;
  uploaded_chunks: number[];
  completed: boolean;
  media?: MediaAsset;
}

interface ChunkUploadResponse {
  chunk_index: number;
  uploaded_chunks: number[];
}

interface UploadCompleteResponse {
  media: MediaAsset;
}

async function uploadMediaInChunks(projectId: string, file: File, clientOrder: number, onProgress?: TransferProgressCallback): Promise<MediaAsset> {
  const uploadBase = getUploadApiBase();
  const totalChunks = Math.ceil(file.size / UPLOAD_CHUNK_SIZE);
  emitTransferProgress(onProgress, file.name, "checking", 0, file.size);
  const check = await request<UploadCheckResponse>(
    `/api/projects/${projectId}/uploads/check`,
    {
      method: "POST",
      body: JSON.stringify({
        file_name: file.name,
        file_size: file.size,
        chunk_size: UPLOAD_CHUNK_SIZE,
        total_chunks: totalChunks,
        client_order: clientOrder,
        file_signature: fileSignature(file),
        content_type: file.type || null
      })
    },
    uploadBase
  );
  if (check.completed && check.media) {
    emitTransferProgress(onProgress, file.name, "complete", file.size, file.size);
    return check.media;
  }

  const uploaded = new Set(check.uploaded_chunks);
  const missing = Array.from({ length: totalChunks }, (_, index) => index).filter((index) => !uploaded.has(index));
  let uploadedBytes = check.uploaded_chunks.reduce((sum, chunkIndex) => sum + chunkByteSize(file, chunkIndex), 0);
  const activeChunks = new Map<number, number>();
  const reportUpload = () => {
    const activeBytes = Array.from(activeChunks.values()).reduce((sum, value) => sum + value, 0);
    emitTransferProgress(onProgress, file.name, "uploading", Math.min(file.size, uploadedBytes + activeBytes), file.size);
  };
  reportUpload();
  await runPool(missing, UPLOAD_CONCURRENCY, async (chunkIndex) => {
    await uploadChunk(check.upload_id, file, chunkIndex, (loaded) => {
      activeChunks.set(chunkIndex, loaded);
      reportUpload();
    });
    uploadedBytes += chunkByteSize(file, chunkIndex);
    activeChunks.delete(chunkIndex);
    reportUpload();
  });

  emitTransferProgress(onProgress, file.name, "completing", file.size, file.size);
  const complete = await request<UploadCompleteResponse>(
    `/api/uploads/${check.upload_id}/complete`,
    { method: "POST" },
    uploadBase
  );
  emitTransferProgress(onProgress, file.name, "complete", file.size, file.size);
  return complete.media;
}

async function uploadChunk(
  uploadId: string,
  file: File,
  chunkIndex: number,
  onProgress?: (loadedBytes: number, totalBytes: number) => void
): Promise<ChunkUploadResponse> {
  const uploadBase = getUploadApiBase();
  const start = chunkIndex * UPLOAD_CHUNK_SIZE;
  const end = Math.min(file.size, start + UPLOAD_CHUNK_SIZE);
  const body = file.slice(start, end);
  return withRetries(
    () =>
      xhrRequest<ChunkUploadResponse>(
        `/api/uploads/${uploadId}/chunks/${chunkIndex}/raw`,
        { method: "PUT", body, headers: { "Content-Type": "application/octet-stream" } },
        uploadBase,
        onProgress
      ),
    CHUNK_RETRIES
  );
}

function getUploadApiBase(): string {
  return transferApiBase();
}

function transferApiBase(): string {
  return TRANSFER_API_BASE.replace(/\/$/, "");
}

function readPositiveEnvInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function chunkByteSize(file: File, chunkIndex: number): number {
  const start = chunkIndex * UPLOAD_CHUNK_SIZE;
  return Math.max(0, Math.min(file.size, start + UPLOAD_CHUNK_SIZE) - start);
}

function fileSignature(file: File): string {
  return [file.name, file.size, file.lastModified, file.type || "application/octet-stream"].join("|");
}

function emitTransferProgress(
  onProgress: TransferProgressCallback | undefined,
  fileName: string,
  phase: TransferPhase,
  loadedBytes: number,
  totalBytes: number
): void {
  const total = Math.max(0, totalBytes);
  const loaded = Math.max(0, Math.min(loadedBytes, total || loadedBytes));
  onProgress?.({
    fileName,
    phase,
    loadedBytes: loaded,
    totalBytes: total,
    percent: total > 0 ? Math.max(0, Math.min(100, Math.round((loaded / total) * 100))) : 0
  });
}

function xhrRequest<T>(
  path: string,
  init: { method: string; body?: XMLHttpRequestBodyInit; auth?: boolean; headers?: Record<string, string> },
  base = API_BASE,
  onUploadProgress?: (loadedBytes: number, totalBytes: number) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(init.method, `${base}${path}`);
    const token = (init.auth ?? true) ? getToken() : null;
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    const hasContentType = Object.keys(init.headers ?? {}).some((key) => key.toLowerCase() === "content-type");
    if (!(init.body instanceof FormData) && !(init.body instanceof Blob) && !hasContentType) {
      xhr.setRequestHeader("Content-Type", "application/json");
    }
    for (const [key, value] of Object.entries(init.headers ?? {})) {
      xhr.setRequestHeader(key, value);
    }
    xhr.upload.onprogress = (event) => {
      onUploadProgress?.(event.loaded, event.lengthComputable ? event.total : 0);
    };
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new ApiError(readErrorMessage(xhr.responseText, `${xhr.status} ${xhr.statusText}`), xhr.status, xhr.statusText));
        return;
      }
      try {
        resolve(JSON.parse(xhr.responseText || "{}") as T);
      } catch (err) {
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("Network request failed"));
    xhr.send(init.body ?? null);
  });
}

async function runPool<T>(items: T[], concurrency: number, worker: (item: T) => Promise<unknown>): Promise<void> {
  let next = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) {
      const item = items[next];
      next += 1;
      await worker(item);
    }
  });
  await Promise.all(runners);
}

async function withRetries<T>(fn: () => Promise<T>, retries: number): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < retries) await new Promise((resolve) => setTimeout(resolve, 300 * (attempt + 1)));
    }
  }
  throw lastError;
}

function hashFile(file: File, onProgress?: (loadedBytes: number, totalBytes: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const workerUrl = URL.createObjectURL(
      new Blob(
        [
          `
            self.onmessage = async (event) => {
              try {
                const file = event.data.file;
                const readSize = event.data.readSize;
                const hasher = createSha256();
                let loaded = 0;
                for (let offset = 0; offset < file.size; offset += readSize) {
                  const buffer = await file.slice(offset, offset + readSize).arrayBuffer();
                  hasher.update(new Uint8Array(buffer));
                  loaded += buffer.byteLength;
                  self.postMessage({ loaded, total: file.size });
                }
                self.postMessage({ hash: hasher.digest() });
              } catch (error) {
                self.postMessage({ error: error instanceof Error ? error.message : "Failed to hash file" });
              }
            };

            function createSha256() {
              const K = [
                0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
                0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
                0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
                0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
                0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
                0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
                0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
                0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
              ];
              const h = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
              const w = new Uint32Array(64);
              const buffer = new Uint8Array(64);
              let bufferLength = 0;
              let bytesHashed = 0n;

              function rotr(value, shift) {
                return (value >>> shift) | (value << (32 - shift));
              }

              function processBlock(chunk) {
                for (let i = 0; i < 16; i += 1) {
                  const j = i * 4;
                  w[i] = ((chunk[j] << 24) | (chunk[j + 1] << 16) | (chunk[j + 2] << 8) | chunk[j + 3]) >>> 0;
                }
                for (let i = 16; i < 64; i += 1) {
                  const s0 = (rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3)) >>> 0;
                  const s1 = (rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10)) >>> 0;
                  w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
                }
                let a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
                for (let i = 0; i < 64; i += 1) {
                  const s1 = (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) >>> 0;
                  const ch = ((e & f) ^ (~e & g)) >>> 0;
                  const temp1 = (hh + s1 + ch + K[i] + w[i]) >>> 0;
                  const s0 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) >>> 0;
                  const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
                  const temp2 = (s0 + maj) >>> 0;
                  hh = g;
                  g = f;
                  f = e;
                  e = (d + temp1) >>> 0;
                  d = c;
                  c = b;
                  b = a;
                  a = (temp1 + temp2) >>> 0;
                }
                h[0] = (h[0] + a) >>> 0;
                h[1] = (h[1] + b) >>> 0;
                h[2] = (h[2] + c) >>> 0;
                h[3] = (h[3] + d) >>> 0;
                h[4] = (h[4] + e) >>> 0;
                h[5] = (h[5] + f) >>> 0;
                h[6] = (h[6] + g) >>> 0;
                h[7] = (h[7] + hh) >>> 0;
              }

              return {
                update(data) {
                  bytesHashed += BigInt(data.length);
                  let offset = 0;
                  if (bufferLength > 0) {
                    const available = Math.min(64 - bufferLength, data.length);
                    buffer.set(data.subarray(0, available), bufferLength);
                    bufferLength += available;
                    offset += available;
                    if (bufferLength === 64) {
                      processBlock(buffer);
                      bufferLength = 0;
                    }
                  }
                  while (offset + 64 <= data.length) {
                    processBlock(data.subarray(offset, offset + 64));
                    offset += 64;
                  }
                  if (offset < data.length) {
                    buffer.set(data.subarray(offset), 0);
                    bufferLength = data.length - offset;
                  }
                },
                digest() {
                  const bitLength = bytesHashed * 8n;
                  buffer[bufferLength] = 0x80;
                  bufferLength += 1;
                  if (bufferLength > 56) {
                    buffer.fill(0, bufferLength, 64);
                    processBlock(buffer);
                    bufferLength = 0;
                  }
                  buffer.fill(0, bufferLength, 56);
                  let bits = bitLength;
                  for (let i = 63; i >= 56; i -= 1) {
                    buffer[i] = Number(bits & 0xffn);
                    bits >>= 8n;
                  }
                  processBlock(buffer);
                  return h.map((value) => value.toString(16).padStart(8, "0")).join("");
                }
              };
            }
          `
        ],
        { type: "text/javascript" }
      )
    );
    const worker = new Worker(workerUrl);
    const cleanup = () => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
    };
    worker.onmessage = (event: MessageEvent<{ hash?: string; error?: string; loaded?: number; total?: number }>) => {
      if (typeof event.data.loaded === "number") {
        onProgress?.(event.data.loaded, event.data.total || file.size);
        return;
      }
      cleanup();
      if (event.data.hash) resolve(event.data.hash);
      else reject(new Error(event.data.error || "Failed to hash file"));
    };
    worker.onerror = (event) => {
      cleanup();
      reject(new Error(event.message || "Failed to hash file"));
    };
    worker.postMessage({ file, readSize: HASH_READ_SIZE });
  });
}

export async function downloadFileWithProgress(
  url: string,
  fileName: string,
  expectedBytes = 0,
  onProgress?: TransferProgressCallback
): Promise<void> {
  const { bytes, contentType } = await fetchBytesWithProgress(url, fileName, expectedBytes, onProgress);
  saveBlob(new Blob([bytesToArrayBuffer(bytes)], { type: contentType || "application/octet-stream" }), fileName);
}

export async function fetchBytesWithProgress(
  url: string,
  fileName: string,
  expectedBytes = 0,
  onProgress?: TransferProgressCallback,
  signal?: AbortSignal
): Promise<{ bytes: Uint8Array; contentType: string }> {
  const totalBytes = expectedBytes > 0 ? expectedBytes : await probeRangeSize(url, signal).catch(() => 0);
  if (totalBytes > RANGE_DOWNLOAD_CHUNK_SIZE) {
    return fetchRangeBytes(url, fileName, totalBytes, onProgress, signal);
  }
  return fetchStreamBytes(url, fileName, totalBytes, onProgress, signal);
}

async function fetchStreamBytes(
  url: string,
  fileName: string,
  expectedBytes = 0,
  onProgress?: TransferProgressCallback,
  signal?: AbortSignal
): Promise<{ bytes: Uint8Array; contentType: string }> {
  emitTransferProgress(onProgress, fileName, "downloading", 0, expectedBytes);
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(readErrorMessage(text, `${response.status} ${response.statusText}`));
  }

  const contentLength = Number(response.headers.get("Content-Length") || 0);
  const totalBytes = contentLength > 0 ? contentLength : expectedBytes;
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    emitTransferProgress(onProgress, fileName, "complete", totalBytes || bytes.byteLength, totalBytes || bytes.byteLength);
    return { bytes, contentType: response.headers.get("Content-Type") || "application/octet-stream" };
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    const chunk = new Uint8Array(value.byteLength);
    chunk.set(value);
    chunks.push(chunk);
    loadedBytes += value.byteLength;
    emitTransferProgress(onProgress, fileName, "downloading", loadedBytes, totalBytes);
  }
  emitTransferProgress(onProgress, fileName, "complete", totalBytes || loadedBytes, totalBytes || loadedBytes);
  return { bytes: concatBytes(chunks, loadedBytes), contentType: response.headers.get("Content-Type") || "application/octet-stream" };
}

async function fetchRangeBytes(
  url: string,
  fileName: string,
  totalBytes: number,
  onProgress?: TransferProgressCallback,
  signal?: AbortSignal
): Promise<{ bytes: Uint8Array; contentType: string }> {
  emitTransferProgress(onProgress, fileName, "downloading", 0, totalBytes);
  const ranges = Array.from({ length: Math.ceil(totalBytes / RANGE_DOWNLOAD_CHUNK_SIZE) }, (_, index) => {
    const start = index * RANGE_DOWNLOAD_CHUNK_SIZE;
    return { index, start, end: Math.min(totalBytes - 1, start + RANGE_DOWNLOAD_CHUNK_SIZE - 1) };
  });
  const result = new Uint8Array(totalBytes);
  const active = new Map<number, number>();
  let completedBytes = 0;
  let contentType = "application/octet-stream";
  const report = () => {
    const activeBytes = Array.from(active.values()).reduce((sum, value) => sum + value, 0);
    emitTransferProgress(onProgress, fileName, "downloading", Math.min(totalBytes, completedBytes + activeBytes), totalBytes);
  };

  await runPool(ranges, RANGE_DOWNLOAD_CONCURRENCY, async ({ index, start, end }) => {
    const response = await fetch(url, {
      cache: "no-store",
      headers: { Range: `bytes=${start}-${end}` },
      signal
    });
    if (response.status !== 206) {
      const text = await response.text();
      throw new Error(readErrorMessage(text, `${response.status} ${response.statusText}`));
    }
    contentType = response.headers.get("Content-Type") || contentType;
    const expected = end - start + 1;
    let loaded = 0;
    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== expected) throw new Error("Downloaded chunk size does not match expected size");
      result.set(bytes, start);
      loaded = bytes.byteLength;
    } else {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;
        result.set(value, start + loaded);
        loaded += value.byteLength;
        active.set(index, loaded);
        report();
      }
      if (loaded !== expected) throw new Error("Downloaded chunk size does not match expected size");
    }
    active.delete(index);
    completedBytes += loaded;
    report();
  });

  emitTransferProgress(onProgress, fileName, "complete", totalBytes, totalBytes);
  return { bytes: result, contentType };
}

async function probeRangeSize(url: string, signal?: AbortSignal): Promise<number> {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { Range: "bytes=0-0" },
    signal
  });
  if (response.status !== 206) return 0;
  const contentRange = response.headers.get("Content-Range") || "";
  const match = contentRange.match(/\/(\d+)$/);
  return match ? Number(match[1]) : 0;
}

function concatBytes(chunks: Uint8Array[], totalBytes: number): Uint8Array {
  const result = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function bytesToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

function saveBlob(blob: Blob, fileName: string): void {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

export const api = {
  register: (payload: { username: string; password: string; email?: string }) =>
    request<AuthResponse>("/api/auth/register", { method: "POST", body: JSON.stringify(payload), auth: false }),
  login: (payload: { username: string; password: string }) =>
    request<AuthResponse>("/api/auth/login", { method: "POST", body: JSON.stringify(payload), auth: false }),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  me: () => request<User>("/api/me"),
  resources: () => request<{ cpu: Record<string, unknown>; memory: Record<string, unknown>; gpu: Record<string, unknown>; workers?: Record<string, unknown> }>("/api/system/resources"),
  algorithms: () => request<{ algorithms: AlgorithmEntry[] }>("/api/algorithms", { auth: false }),
  adminAlgorithms: () => request<{ algorithms: AlgorithmEntry[] }>("/api/admin/algorithms"),
  runtimePreflight: () => request<RuntimePreflight>("/api/admin/runtime/preflight"),
  adminTasks: () => request<{ tasks: Task[] }>("/api/admin/tasks"),
  adminProjects: () => request<{ projects: AdminProjectUsage[] }>("/api/admin/projects"),
  adminUsers: () => request<{ users: AdminUserUsage[] }>("/api/admin/users"),
  adminFeedback: () => request<{ feedback: FeedbackEntry[] }>("/api/admin/feedback"),
  workers: () => request<{ workers: unknown[]; message?: string }>("/api/admin/workers"),
  pipelineParameterSchema: () => request<PipelineParameterSchema>("/api/pipeline-parameters/schema", { auth: false }),
  pipelineParameterDefaults: () => request<PipelineParameterDefaultsResponse>("/api/admin/pipeline-parameter-defaults"),
  savePipelineParameterDefaults: (pipeline: string, sceneType: PipelineSceneType, options: Record<string, unknown>) =>
    request<{ pipeline: string; scene_type: PipelineSceneType; options: Record<string, unknown>; updated_at: string }>(
      `/api/admin/pipeline-parameter-defaults/${encodeURIComponent(pipeline)}/${sceneType}`,
      { method: "PUT", body: JSON.stringify({ options }) }
    ),
  projectSummary: () => request<Record<string, number>>("/api/projects/summary"),
  projects: () => request<{ projects: Project[] }>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  deleteProject: (id: string) => request<{ deleted: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),
  deleteProjects: (projectIds: string[]) =>
    request<{ deleted: number; project_ids: string[] }>("/api/projects/bulk-delete", { method: "POST", body: JSON.stringify({ project_ids: projectIds }) }),
  createProject: (payload: { name: string; input_type: Project["input_type"]; tags: string[] }) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify(payload) }),
  uploadMedia: (projectId: string, file: File, clientOrder: number, onProgress?: TransferProgressCallback) => uploadMediaInChunks(projectId, file, clientOrder, onProgress),
  deleteMedia: (projectId: string, mediaId: string) =>
    request<{ deleted: boolean; source_version?: number }>(`/api/projects/${projectId}/media/${mediaId}`, { method: "DELETE" }),
  media: (projectId: string) => request<{ media: MediaAsset[] }>(`/api/projects/${projectId}/media`),
  mediaStats: (projectId: string) => request<Record<string, unknown>>(`/api/projects/${projectId}/media/stats`),
  startPreview: (projectId: string, options: Record<string, unknown> = {}) =>
    request<Task>(`/api/projects/${projectId}/tasks/preview`, { method: "POST", body: JSON.stringify({ options }) }),
  startFine: (projectId: string, options: Record<string, unknown> = {}) =>
    request<Task>(`/api/projects/${projectId}/tasks/fine`, { method: "POST", body: JSON.stringify({ options }) }),
  task: (id: string) => request<Task>(`/api/tasks/${id}`),
  taskLog: (id: string) => requestText(`/api/tasks/${id}/log`),
  cancelTask: (id: string) => request<Task>(`/api/tasks/${id}/cancel`, { method: "POST" }),
  artifacts: (projectId: string) => request<{ artifacts: Artifact[] }>(`/api/projects/${projectId}/artifacts`),
  artifactDownloadUrl: (artifactId: string) => request<{ url: string; expires_in_seconds: number }>(`/api/artifacts/${artifactId}/download-url`),
  artifactOriginalPlyDownloadUrl: (artifactId: string) =>
    request<{ url: string; expires_in_seconds: number }>(`/api/artifacts/${artifactId}/original-ply/download-url`),
  viewerConfig: (projectId: string) => request<ViewerConfig>(`/api/projects/${projectId}/viewer-config`),
  createProjectShare: (projectId: string) => request<ProjectShareResponse>(`/api/projects/${projectId}/share`, { method: "POST" }),
  deleteProjectShare: (projectId: string) => request<{ deleted: boolean }>(`/api/projects/${projectId}/share`, { method: "DELETE" }),
  sharedProject: (shareToken: string) => request<SharedProject>(`/api/shared-projects/${shareToken}`, { auth: false }),
  feedback: (payload: { title: string; content: string; project_id?: string }) =>
    request<Record<string, unknown>>("/api/feedback", { method: "POST", body: JSON.stringify(payload) })
};

export function projectEventsUrl(projectId: string): string {
  const token = getToken();
  const suffix = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${API_BASE}/api/projects/${projectId}/events${suffix}`;
}

export function artifactUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${transferApiBase()}${path}`;
}

export function mediaThumbnailUrl(media: MediaAsset): string | null {
  if (!media.thumbnail_uri) return null;
  return authenticatedAssetPath(`/api/media/${media.id}/thumbnail`);
}

export function mediaFileUrl(media: MediaAsset): string {
  return authenticatedAssetPath(`/api/media/${media.id}/file`);
}

function authenticatedAssetPath(path: string): string {
  const token = getToken();
  const suffix = token ? `${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}` : "";
  return `${transferApiBase()}${path}${suffix}`;
}

export function formatBytes(value: number | undefined | null): string {
  const bytes = value ?? 0;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}
