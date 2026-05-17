import type { Project, ProjectStatus, Task } from "@/lib/types";

const projectStatusLabels: Record<ProjectStatus, string> = {
  CREATED: "已创建",
  UPLOADING: "上传中",
  PREPROCESSING: "预处理中",
  PREVIEW_RUNNING: "预览生成中",
  PREVIEW_READY: "预览就绪",
  FINE_QUEUED: "精细重建排队",
  GLOBAL_OPTIMIZING: "全局优化中",
  FINE_RUNNING: "精细重建中",
  COMPLETED: "已完成",
  FAILED: "失败",
  CANCELED: "已取消"
};

const taskStatusLabels: Record<Task["status"], string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消"
};

const taskTypeLabels: Record<Task["type"], string> = {
  preview: "极速预览",
  fine: "精细重建",
  lod: "LOD 生成",
  mesh_export: "网格导出"
};

export function projectStatusLabel(status?: ProjectStatus | null): string {
  return status ? projectStatusLabels[status] ?? status : "-";
}

export function taskStatusLabel(status?: Task["status"] | null): string {
  return status ? taskStatusLabels[status] ?? status : "-";
}

export function taskTypeLabel(type?: Task["type"] | null): string {
  return type ? taskTypeLabels[type] ?? type : "-";
}

export function inputTypeLabel(type?: Project["input_type"] | null): string {
  if (type === "images") return "图片序列";
  if (type === "video") return "视频";
  return "-";
}

export function isActiveTask(task?: Task | null): boolean {
  return task?.status === "queued" || task?.status === "running";
}

export function formatTaskEta(task?: Task | null): string {
  if (!task || !isActiveTask(task)) return formatEta(0);
  const estimated = estimateTaskEtaSeconds(task);
  return formatEta(estimated ?? task.eta_seconds);
}

export function estimateTaskEtaSeconds(task: Pick<Task, "progress" | "eta_seconds" | "logs" | "created_at" | "started_at">): number | null {
  const progress = Math.max(0, Math.min(99, effectiveTaskProgress(task)));
  if (progress <= 0) return task.eta_seconds ?? null;
  const logEstimate = estimateEtaFromLogs(task, progress);
  if (logEstimate !== null) return logEstimate;

  const startedAt = Date.parse(task.started_at || task.created_at);
  if (Number.isFinite(startedAt)) {
    const elapsedSeconds = Math.max(1, (Date.now() - startedAt) / 1000);
    return Math.max(0, Math.round((elapsedSeconds * (100 - progress)) / progress));
  }
  return task.eta_seconds ?? null;
}

export function effectiveTaskProgress(task?: (Pick<Task, "progress" | "logs"> & Partial<Pick<Task, "status">>) | null): number {
  if (!task) return 0;
  if ("status" in task && task.status === "succeeded") return 100;
  const stored = Math.max(0, Math.min(100, Number(task.progress || 0)));
  const logged = latestLogProgress(task.logs ?? []);
  if (logged === null) return Math.round(stored);
  return Math.round(Math.max(stored, logged));
}

export function formatDateTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

export function formatEta(seconds?: number | null): string {
  if (seconds === undefined || seconds === null) return "估算中";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} 秒`;
  const minutes = Math.floor(seconds / 60);
  const remain = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes} 分 ${remain} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes % 60} 分`;
}

function estimateEtaFromLogs(task: Pick<Task, "logs" | "created_at" | "started_at">, currentProgress: number): number | null {
  const directEta = latestLogEta(task.logs ?? []);
  if (directEta !== null) return directEta;
  const points = (task.logs ?? []).map(parseTimedProgressLog).filter((item): item is { time: number; progress: number; etaSeconds: number | null } => Boolean(item));
  const latestEta = [...points].reverse().find((point) => point.etaSeconds !== null)?.etaSeconds;
  if (latestEta !== undefined) return latestEta;
  if (points.length >= 2) {
    const window = points.slice(-8);
    const first = window.find((point) => point.progress < window[window.length - 1].progress) ?? window[0];
    const last = window[window.length - 1];
    const progressDelta = last.progress - first.progress;
    const secondsDelta = (last.time - first.time) / 1000;
    if (progressDelta > 0 && secondsDelta > 0) {
      return Math.max(0, Math.round((secondsDelta / progressDelta) * (100 - currentProgress)));
    }
  }

  const timedLogs = (task.logs ?? []).map(parseLogTime).filter((value): value is number => value !== null);
  const startedAt = Date.parse(task.started_at || task.created_at);
  if (timedLogs.length >= 2 && Number.isFinite(startedAt)) {
    const latestLogTime = Math.max(...timedLogs);
    const elapsedSeconds = Math.max(1, (latestLogTime - startedAt) / 1000);
    if (elapsedSeconds > 0) {
      return Math.max(0, Math.round((elapsedSeconds * (100 - currentProgress)) / currentProgress));
    }
  }
  return null;
}

function latestLogProgress(logs: string[]): number | null {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const point = parseProgressLog(logs[index]);
    if (point !== null) return point.progress;
  }
  return null;
}

function latestLogEta(logs: string[]): number | null {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const eta = parseEtaSeconds(logs[index]);
    if (eta !== null) return eta;
  }
  return null;
}

function parseTimedProgressLog(line: string): { time: number; progress: number; etaSeconds: number | null } | null {
  const time = parseLogTime(line);
  if (time === null) return null;
  const progressPoint = parseProgressLog(line);
  if (!progressPoint) return null;
  return { time, progress: progressPoint.progress, etaSeconds: parseEtaSeconds(line) };
}

function parseProgressLog(line: string): { progress: number } | null {
  const progressMatch = line.match(/(?:progress\s*[=:]\s*(\d{1,3})(?:\.\d+)?|(\d{1,3})(?:\.\d+)?\s*%)/i);
  const progressValue = progressMatch?.[1] ?? progressMatch?.[2];
  const progress = progressValue ? Number(progressValue) : NaN;
  if (!Number.isFinite(progress) || progress < 0 || progress > 100) return null;
  return { progress };
}

function parseEtaSeconds(line: string): number | null {
  const match = line.match(/\beta_seconds\s*=\s*(\d+)/i);
  if (match) {
    const value = Number(match[1]);
    return Number.isFinite(value) ? value : null;
  }
  const tqdm = line.match(/\[[^\]<]*<([^,\]]+)/);
  return tqdm ? parseDurationSeconds(tqdm[1].trim()) : null;
}

function parseDurationSeconds(value: string): number | null {
  const parts = value.split(":");
  if (parts.length < 1 || parts.length > 3) return null;
  const numbers = parts.map((part) => Number(part));
  if (numbers.some((part) => !Number.isFinite(part))) return null;
  return numbers.reduce((total, part) => total * 60 + part, 0);
}

function parseLogTime(line: string): number | null {
  const iso = line.match(/\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/);
  if (iso) {
    const normalized = iso[0].includes("T") ? iso[0] : iso[0].replace(" ", "T");
    const parsed = Date.parse(normalized);
    if (Number.isFinite(parsed)) return parsed;
  }
  const bracketed = line.match(/\[(\d{2}:\d{2}:\d{2})\]/);
  if (!bracketed) return null;
  const base = new Date();
  const [hours, minutes, seconds] = bracketed[1].split(":").map(Number);
  base.setHours(hours, minutes, seconds, 0);
  return base.getTime();
}
