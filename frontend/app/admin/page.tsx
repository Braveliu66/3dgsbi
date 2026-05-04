"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { formatDateTime, taskStatusLabel } from "@/lib/labels";
import type { AlgorithmEntry, RuntimePreflight, Task, User } from "@/lib/types";

export default function AdminPage() {
  const [user, setUser] = useState<User | null>(null);
  const [algorithms, setAlgorithms] = useState<AlgorithmEntry[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [resources, setResources] = useState<Record<string, unknown>>({});
  const [workers, setWorkers] = useState<unknown[]>([]);
  const [preflight, setPreflight] = useState<RuntimePreflight | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.me().then(setUser).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!user) return;
    if (user.role !== "admin") {
      setError("需要管理员权限。");
      return;
    }
    void Promise.all([api.adminAlgorithms(), api.adminTasks(), api.resources(), api.workers(), api.runtimePreflight()])
      .then(([algorithmData, taskData, resourceData, workerData, preflightData]) => {
        setAlgorithms(algorithmData.algorithms);
        setTasks(taskData.tasks);
        setResources(resourceData as unknown as Record<string, unknown>);
        setWorkers(workerData.workers);
        setPreflight(preflightData);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "读取管理接口失败"));
  }, [user]);

  if (error) {
    return (
      <div className="workspace-page">
        <header className="page-header compact">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>管理面板</h1>
          </div>
        </header>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>管理面板</h1>
        </div>
      </header>

      <section className="detail-grid">
        <div className="panel fill">
          <div className="panel-head">
            <h2>运行预检</h2>
            <span className={`status-pill ${preflight?.errors.length ? "failed" : "ready"}`}>{preflight?.errors.length ? "需处理" : "就绪"}</span>
          </div>
          <div className="panel-body scrollable stack">
            <section className="grid three">
              <div className="panel stat flat"><span className="muted small">Python</span><strong>{String(preflight?.python.version ?? "-")}</strong></div>
              <div className="panel stat flat"><span className="muted small">CUDA</span><strong>{String(preflight?.torch.cuda_available ?? false)}</strong></div>
              <div className="panel stat flat"><span className="muted small">GPU</span><strong>{String(preflight?.gpu.available ?? false)}</strong></div>
            </section>
            {preflight?.errors.length ? <div className="error-box">{preflight.errors.join("\n")}</div> : null}
            <div className="table">
              <div className="table-row header"><span>算法</span><span>许可证</span><span>Commit</span><span>状态</span></div>
              {(preflight?.algorithms ?? []).map((item) => (
                <div className="table-row" key={item.name}>
                  <span className="truncate" title={item.name}>{item.name}</span>
                  <span>{item.license ?? "-"}</span>
                  <span className="small muted truncate" title={item.commit_hash ?? "-"}>{item.commit_hash ?? "-"}</span>
                  <span className={`status-pill ${item.ready ? "ready" : "failed"}`}>{item.ready ? "ready" : item.enabled ? "check" : "disabled"}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <aside className="detail-side">
          <section className="grid three">
            <div className="panel stat"><span className="muted small">任务</span><strong>{tasks.length}</strong></div>
            <div className="panel stat"><span className="muted small">Worker</span><strong>{workers.length}</strong></div>
            <div className="panel stat"><span className="muted small">启用算法</span><strong>{algorithms.filter((item) => item.enabled).length}</strong></div>
          </section>

          <div className="panel fill">
            <div className="panel-head"><h2>任务与资源</h2></div>
            <div className="panel-body scrollable stack">
              <section className="stack">
                <h3>近期任务</h3>
                {tasks.length ? (
                  <div className="task-list">
                    {tasks.slice(0, 12).map((task) => (
                      <div className="list-row" style={{ gridTemplateColumns: "minmax(0, 1fr) 92px 82px" }} key={task.id}>
                        <span className="truncate" title={task.current_stage}>{task.type} · {task.current_stage || "-"}</span>
                        <span className={`status-pill ${task.status}`}>{taskStatusLabel(task.status)}</span>
                        <span className="muted small">{formatDateTime(task.created_at)}</span>
                      </div>
                    ))}
                  </div>
                ) : <div className="empty-state">暂无任务</div>}
              </section>

              <section className="stack">
                <h3>资源 JSON</h3>
                <pre className="empty-state">{JSON.stringify(resources, null, 2)}</pre>
              </section>
            </div>
          </div>
        </aside>
      </section>
    </div>
  );
}
