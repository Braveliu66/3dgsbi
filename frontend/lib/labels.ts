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
  if (type === "camera") return "实时相机";
  return "-";
}

export function isActiveTask(task?: Task | null): boolean {
  return task?.status === "queued" || task?.status === "running";
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
