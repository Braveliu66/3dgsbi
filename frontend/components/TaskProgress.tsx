import type { Task } from "@/lib/types";
import { formatEta, taskStatusLabel, taskTypeLabel } from "@/lib/labels";

export function TaskProgress({ task }: { task?: Task | null }) {
  if (!task) return <div className="empty-state">暂无运行任务</div>;

  const progress = Math.max(0, Math.min(100, task.progress || 0));
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
      {task.error_message ? <div className="error-box">{task.error_message}</div> : null}
    </div>
  );
}
