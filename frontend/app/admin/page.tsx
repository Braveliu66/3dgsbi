"use client";

import { Cpu, Gauge, HardDrive, MemoryStick, PauseCircle, X, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import { formatDateTime, formatEta, projectStatusLabel, taskStatusLabel, taskTypeLabel } from "@/lib/labels";
import type { AdminProjectUsage, AdminUserUsage, FeedbackEntry, ResourceSnapshotPoint, User } from "@/lib/types";

const RESOURCE_LIMIT = 60;

type ResourceTone = "cpu" | "memory" | "gpu" | "vram";

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [projects, setProjects] = useState<AdminProjectUsage[]>([]);
  const [users, setUsers] = useState<AdminUserUsage[]>([]);
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([]);
  const [selectedFeedback, setSelectedFeedback] = useState<FeedbackEntry | null>(null);
  const [resourceHistory, setResourceHistory] = useState<ResourceSnapshotPoint[]>([]);
  const [busyTaskId, setBusyTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.me().then(setUser).catch(() => setError("需要登录"));
  }, []);

  useEffect(() => {
    if (!user) return;
    if (user.role !== "admin") {
      setError("需要管理员权限。");
      return;
    }

    let cancelled = false;
    const loadAdminData = () => {
      void Promise.all([api.adminProjects(), api.adminUsers(), api.adminFeedback()])
        .then(([projectData, userData, feedbackData]) => {
          if (cancelled) return;
          setProjects(projectData.projects);
          setUsers(userData.users);
          setFeedback(feedbackData.feedback);
        })
        .catch((err) => {
          if (!cancelled) setError(err instanceof Error ? err.message : "读取管理接口失败");
        });
    };
    const sampleResources = () => {
      void api.resources()
        .then((resources) => {
          if (cancelled) return;
          setResourceHistory((items) => [...items, resourcePoint(resources)].slice(-RESOURCE_LIMIT));
        })
        .catch(() => undefined);
    };

    loadAdminData();
    sampleResources();
    const dataTimer = window.setInterval(loadAdminData, 5000);
    const resourceTimer = window.setInterval(sampleResources, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(dataTimer);
      window.clearInterval(resourceTimer);
    };
  }, [user]);

  const runningProjects = useMemo(() => projects.filter((project) => isAdminTaskActive(project.latest_task)).length, [projects]);
  const totalStorage = useMemo(() => users.reduce((sum, item) => sum + item.total_size_bytes, 0), [users]);

  async function stopTask(taskId: string) {
    setBusyTaskId(taskId);
    setError(null);
    try {
      await api.cancelTask(taskId);
      const projectData = await api.adminProjects();
      setProjects(projectData.projects);
    } catch (err) {
      setError(err instanceof Error ? err.message : "停止任务失败");
    } finally {
      setBusyTaskId(null);
    }
  }

  if (error && user?.role !== "admin") {
    return (
      <div className="workspace-page">
        <header className="page-header compact">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>管理</h1>
          </div>
        </header>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  return (
    <div className="workspace-page admin-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>管理</h1>
        </div>
        <div className="actions muted small">
          <span>{projects.length} 个项目</span>
          <span>{runningProjects} 个运行中</span>
          <span>{formatBytes(totalStorage)}</span>
        </div>
      </header>

      <section className="admin-grid">
        <div className="panel fill">
          <div className="panel-head">
            <h2>资源</h2>
            <span className="muted small">最近 {Math.min(resourceHistory.length, RESOURCE_LIMIT)} 秒</span>
          </div>
          <div className="panel-body admin-chart-grid">
            <ResourceChart tone="cpu" icon={Cpu} label="CPU" value={latest(resourceHistory, "cpu")} points={resourceHistory.map((item) => item.cpu)} />
            <ResourceChart tone="memory" icon={MemoryStick} label="RAM" value={latest(resourceHistory, "memory")} points={resourceHistory.map((item) => item.memory)} />
            <ResourceChart tone="gpu" icon={Gauge} label="GPU" value={latest(resourceHistory, "gpu")} points={resourceHistory.map((item) => item.gpu)} />
            <ResourceChart tone="vram" icon={HardDrive} label="VRAM" value={latest(resourceHistory, "vram")} points={resourceHistory.map((item) => item.vram)} />
          </div>
        </div>

        <div className="panel fill">
          <div className="panel-head"><h2>训练项目</h2></div>
          <div className="panel-body scrollable">
            <div className="admin-list">
              <div className="admin-row header project">
                <span>项目</span><span>用户</span><span>任务</span><span>资源</span><span>占用</span><span />
              </div>
              {projects.map((project) => (
                <div className="admin-row project" key={project.id}>
                  <span className="truncate project-main" title={project.name}>
                    <strong>{project.name}</strong>
                    <small>{projectStatusLabel(project.status)} · {formatDateTime(project.updated_at)}</small>
                  </span>
                  <span className="truncate muted small">{project.owner_username ?? project.owner_id}</span>
                  <span>
                    {project.latest_task ? (
                      <>
                        <span className={`status-pill ${project.latest_task.status}`}>{taskStatusLabel(project.latest_task.status)}</span>
                        <small>{taskTypeLabel(project.latest_task.type)} · {project.latest_task.progress}% · {formatEta(project.latest_task.eta_seconds)}</small>
                      </>
                    ) : <span className="muted small">无任务</span>}
                  </span>
                  <ProjectResourceBubbles project={project} />
                  <span className="muted small">{formatBytes(project.total_size_bytes)}</span>
                  <span>
                    {project.latest_task && isAdminTaskActive(project.latest_task) ? (
                      <button className="danger-button compact-action" type="button" disabled={busyTaskId === project.latest_task.id} onClick={() => void stopTask(project.latest_task!.id)}>
                        <PauseCircle size={15} />停止
                      </button>
                    ) : null}
                  </span>
                </div>
              ))}
              {!projects.length ? <div className="empty-state">暂无项目</div> : null}
            </div>
          </div>
        </div>

        <div className="panel fill">
          <div className="panel-head"><h2>用户</h2></div>
          <div className="panel-body scrollable">
            <div className="admin-list">
              <div className="admin-row header user"><span>用户</span><span>角色</span><span>项目</span><span>占用</span><span>反馈</span></div>
              {users.map((item) => (
                <div className="admin-row user" key={item.id}>
                  <span className="truncate"><strong>{item.username}</strong><small>{formatDateTime(item.created_at)}</small></span>
                  <span className={`status-pill ${item.role === "admin" ? "ready" : ""}`}>{item.role}</span>
                  <span>{item.project_count}</span>
                  <span>{formatBytes(item.total_size_bytes)}</span>
                  <span>{item.feedback_count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="panel fill">
          <div className="panel-head"><h2>反馈</h2></div>
          <div className="panel-body scrollable">
            <div className="admin-list">
              {feedback.map((item) => (
                <button className="feedback-row" type="button" key={item.id} onClick={() => setSelectedFeedback(item)}>
                  <div className="row between">
                    <strong className="truncate" title={item.title}>{item.title}</strong>
                    <span className="status-pill">{item.status}</span>
                  </div>
                  <p className="muted small">{item.username ?? item.user_id} · {item.project_name ?? "未关联项目"} · {formatDateTime(item.created_at)}</p>
                  <p>{item.content}</p>
                </button>
              ))}
              {!feedback.length ? <div className="empty-state">暂无反馈</div> : null}
            </div>
          </div>
        </div>
      </section>
      {selectedFeedback ? <FeedbackDetail feedback={selectedFeedback} onClose={() => setSelectedFeedback(null)} /> : null}
      {error ? <div className="error-box admin-floating-error">{error}</div> : null}
    </div>
  );
}

function ResourceChart({ label, value, points, icon: Icon, tone }: { label: string; value: number; points: number[]; icon: LucideIcon; tone: ResourceTone }) {
  return (
    <div className={`resource-chart tone-${tone}`}>
      <div className="resource-chart-side">
        <span className="resource-chart-icon"><Icon size={16} /></span>
        <strong>{Math.round(value)}%</strong>
      </div>
      <div className="resource-chart-main">
        <span>{label}</span>
        <svg viewBox="0 0 120 48" preserveAspectRatio="none" aria-label={`${label} usage`}>
          <polyline points={chartPoints(points)} />
        </svg>
      </div>
    </div>
  );
}

function ProjectResourceBubbles({ project }: { project: AdminProjectUsage }) {
  const cpu = project.worker?.cpu_utilization;
  const gpu = project.worker?.gpu_utilization;
  const used = project.worker?.gpu_memory_used ?? 0;
  const total = project.worker?.gpu_memory_total ?? 0;
  const vram = total > 0 ? `${(used / 1024).toFixed(1)}G/${(total / 1024).toFixed(0)}G` : "--";
  return (
    <span className="resource-bubbles">
      <span className="resource-bubble cpu">CPU {cpu == null ? "--" : `${Math.round(cpu)}%`}</span>
      <span className="resource-bubble gpu">GPU {project.worker?.gpu_index ?? "-"} · {gpu == null ? "--" : `${Math.round(gpu)}%`}</span>
      <span className="resource-bubble vram">VRAM {vram}</span>
    </span>
  );
}

function FeedbackDetail({ feedback, onClose }: { feedback: FeedbackEntry; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div className="modal-panel feedback-detail" role="dialog" aria-modal="true" aria-label="反馈详情" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <p className="eyebrow">Feedback</p>
            <h2>{feedback.title}</h2>
          </div>
          <button className="icon-button" type="button" aria-label="关闭" onClick={onClose}><X size={18} /></button>
        </div>
        <div className="feedback-detail-meta">
          <span>{feedback.username ?? feedback.user_id}</span>
          <span>{feedback.project_name ?? "未关联项目"}</span>
          <span>{feedback.status}</span>
          <span>{formatDateTime(feedback.created_at)}</span>
        </div>
        <p className="feedback-detail-content">{feedback.content}</p>
      </div>
    </div>
  );
}

function chartPoints(values: number[]): string {
  const source = values.length ? values : [0];
  const maxIndex = Math.max(1, source.length - 1);
  return source.map((value, index) => {
    const x = (index / maxIndex) * 120;
    const y = 48 - (Math.max(0, Math.min(100, value)) / 100) * 44 - 2;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}

function latest(history: ResourceSnapshotPoint[], key: keyof Omit<ResourceSnapshotPoint, "time">): number {
  return history.at(-1)?.[key] ?? 0;
}

function resourcePoint(resources: { cpu?: Record<string, unknown>; memory?: Record<string, unknown>; gpu?: Record<string, unknown> }): ResourceSnapshotPoint {
  const gpu = resources.gpu ?? {};
  const memoryUsed = numeric(gpu.memory_used);
  const memoryTotal = numeric(gpu.memory_total);
  return {
    time: Date.now(),
    cpu: numeric(resources.cpu?.usage_percent),
    memory: numeric(resources.memory?.usage_percent),
    gpu: numeric(gpu.usage_percent),
    vram: numeric(gpu.memory_usage_percent) || (memoryTotal > 0 ? (memoryUsed / memoryTotal) * 100 : 0)
  };
}

function isAdminTaskActive(task?: AdminProjectUsage["latest_task"]): boolean {
  return task?.status === "queued" || task?.status === "running";
}

function numeric(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}
