import type { Task } from "@/lib/types";
import { formatEta, taskStatusLabel, taskTypeLabel } from "@/lib/labels";

export function TaskProgress({ task }: { task?: Task | null }) {
  if (!task) return <div className="empty-state">暂无运行任务</div>;

  const progress = Math.max(0, Math.min(100, task.progress || 0));
  const lingbotDetail = lingbotProgressLabel(task.metrics);
  return (
    <div className={`task-progress ${task.status}`}>
      <div className="row between">
        <strong className="truncate" title={task.current_stage || task.type}>
          {task.current_stage || taskTypeLabel(task.type)}
        </strong>
        <span className={`status-pill ${task.status}`}>{taskStatusLabel(task.status)}</span>
      </div>
      <div className="progress-track" aria-label="任务进度">
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="row between muted small">
        <span>{progress}%</span>
        <span>{formatEta(task.eta_seconds)}</span>
      </div>
      {lingbotDetail ? <div className="muted small">{lingbotDetail}</div> : null}
      {task.error_message ? <div className="error-box">{task.error_message}</div> : null}
    </div>
  );
}

function lingbotProgressLabel(metrics?: Record<string, unknown>): string | null {
  if (!metrics) return null;
  const currentWindow = readNumber(metrics.lingbot_current_window);
  const totalWindows = readNumber(metrics.lingbot_total_windows);
  if (currentWindow && totalWindows) {
    return `${currentWindow}/${totalWindows} windows`;
  }
  const currentFrame = readNumber(metrics.lingbot_current_frame);
  const totalFrames = readNumber(metrics.lingbot_total_frames);
  if (currentFrame && totalFrames) {
    return `${currentFrame}/${totalFrames} frames`;
  }
  return null;
}

function readNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : null;
}
