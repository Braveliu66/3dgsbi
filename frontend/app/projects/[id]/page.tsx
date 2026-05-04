"use client";

import Link from "next/link";
import { Download, FileArchive, Film, Images, PauseCircle, PlayCircle, ScrollText, Trash2 } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { api, artifactUrl, formatBytes } from "@/lib/api";
import { formatDateTime, inputTypeLabel, isActiveTask, projectStatusLabel, taskStatusLabel, taskTypeLabel } from "@/lib/labels";
import type { Artifact, Project, Task, ViewerConfig } from "@/lib/types";
import { SplatViewer } from "@/components/SplatViewer";
import { TaskProgress } from "@/components/TaskProgress";

const MEDIA_LIST_THRESHOLD = 18;
const LOG_LIST_THRESHOLD = 12;

export default function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [project, setProject] = useState<Project | null>(null);
  const [viewer, setViewer] = useState<ViewerConfig | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [logTask, setLogTask] = useState<Task | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.id) return;
    void loadProject(params.id);
  }, [params.id]);

  async function loadProject(projectId: string) {
    setError(null);
    try {
      const [projectData, viewerData, artifactData] = await Promise.all([
        api.project(projectId),
        api.viewerConfig(projectId),
        api.artifacts(projectId)
      ]);
      setProject(projectData);
      setViewer(viewerData);
      setArtifacts(artifactData.artifacts);
    } catch (err) {
      setError(err instanceof Error ? err.message : "项目加载失败");
    }
  }

  const latestTask = project?.tasks?.[0];
  const media = project?.media ?? [];
  const tasks = project?.tasks ?? [];
  const logs = latestTask?.logs ?? [];
  const showMediaList = media.length > MEDIA_LIST_THRESHOLD;
  const showCompactLogs = logs.length > LOG_LIST_THRESHOLD;
  const canStartFine = Boolean(project && !isActiveTask(latestTask) && (project.status === "PREVIEW_READY" || project.status === "COMPLETED"));

  const storageStats = useMemo(() => {
    const artifactBytes = artifacts.reduce((sum, item) => sum + item.file_size, 0);
    return { artifactBytes, mediaBytes: project?.total_size_bytes ?? 0 };
  }, [artifacts, project?.total_size_bytes]);

  async function cancelTask(task: Task) {
    setBusy(true);
    setError(null);
    try {
      await api.cancelTask(task.id);
      await loadProject(task.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "取消任务失败");
    } finally {
      setBusy(false);
    }
  }

  async function startFine() {
    if (!project) return;
    setBusy(true);
    setError(null);
    try {
      await api.startFine(project.id);
      await loadProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "精细重建入队失败");
    } finally {
      setBusy(false);
    }
  }

  async function deleteProject() {
    if (!project) return;
    if (!window.confirm(`确定删除项目「${project.name}」吗？此操作无法撤销。`)) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteProject(project.id);
      router.push("/projects");
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除项目失败");
    } finally {
      setBusy(false);
    }
  }

  async function downloadArtifact(artifact: Artifact) {
    setError(null);
    try {
      const result = await api.artifactDownloadUrl(artifact.id);
      window.open(artifactUrl(result.url), "_blank", "noopener,noreferrer");
    } catch (err) {
      setError(err instanceof Error ? err.message : "获取下载链接失败");
    }
  }

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Project Detail</p>
          <h1 className="truncate" title={project?.name}>{project?.name ?? "加载项目中"}</h1>
        </div>
        <div className="actions">
          <Link className="ghost-button" href="/projects">返回项目</Link>
          <button className="ghost-button" type="button" onClick={() => setLogTask(latestTask ?? null)} disabled={!latestTask}>
            <ScrollText size={17} />查看日志
          </button>
          <button className="ghost-button" type="button" onClick={() => void startFine()} disabled={!canStartFine || busy}>
            <PlayCircle size={17} />精细重建
          </button>
          {latestTask && isActiveTask(latestTask) ? (
            <button className="danger-button" type="button" onClick={() => void cancelTask(latestTask)} disabled={busy}>
              <PauseCircle size={17} />取消任务
            </button>
          ) : null}
          <button className="danger-button" type="button" onClick={() => void deleteProject()} disabled={!project || busy}>
            <Trash2 size={17} />删除
          </button>
        </div>
      </header>

      <section className="detail-grid">
        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>3D 查看器</h2>
              <p className="muted small">
                {viewer?.status === "ready"
                  ? viewer.source === "final"
                    ? "已加载精细重建 final_web.spz"
                    : "已加载极速预览 SPZ"
                  : viewer?.message ?? "等待真实预览产物"}
              </p>
            </div>
            {project ? <span className={`status-pill ${project.status}`}>{projectStatusLabel(project.status)}</span> : null}
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            <SplatViewer
              modelUrl={viewer?.status === "ready" ? viewer.model_url : null}
              segments={viewer?.status === "ready" ? viewer.segments : undefined}
            />
          </div>
        </div>

        <aside className="detail-side">
          <section className="grid three">
            <div className="panel stat"><span className="muted small">输入</span><strong>{inputTypeLabel(project?.input_type)}</strong></div>
            <div className="panel stat"><span className="muted small">素材</span><strong>{media.length}</strong></div>
            <div className="panel stat"><span className="muted small">占用</span><strong>{formatBytes(storageStats.mediaBytes + storageStats.artifactBytes)}</strong></div>
          </section>

          <div className="panel fill">
            <div className="panel-head">
              <h2>任务与数据</h2>
              {latestTask ? <span className={`status-pill ${latestTask.status}`}>{taskStatusLabel(latestTask.status)}</span> : null}
            </div>
            <div className="panel-body scrollable stack">
              {error ? <div className="error-box">{error}</div> : null}
              {project?.error_message ? <div className="error-box">{project.error_message}</div> : null}
              {viewer?.status === "unavailable" ? <div className="notice-box">{viewer.message}</div> : null}

              <section className="stack">
                <h3>最新任务</h3>
                <TaskProgress task={latestTask} />
              </section>

              <section className="stack">
                <h3>产物</h3>
                {artifacts.length ? (
                  <div className="artifact-list">
                    {artifacts.map((artifact) => (
                      <div className="list-row" style={{ gridTemplateColumns: "minmax(0, 1fr) 92px 44px" }} key={artifact.id}>
                        <span className="truncate" title={artifact.file_name}><FileArchive size={15} /> {artifact.file_name}</span>
                        <span className="muted small">{formatBytes(artifact.file_size)}</span>
                        <button className="icon-button" type="button" onClick={() => void downloadArtifact(artifact)} aria-label="下载产物">
                          <Download size={16} />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : <div className="empty-state">暂无真实产物</div>}
              </section>

              {viewer?.lods?.length ? (
                <section className="stack">
                  <h3>LOD</h3>
                  <div className="artifact-list">
                    {viewer.lods.map((lod) => (
                      <div className="list-row" style={{ gridTemplateColumns: "64px minmax(0, 1fr) 92px" }} key={lod.artifact_id}>
                        <span>LOD{lod.lod}</span>
                        <span className="muted small truncate">
                          target {Number(lod.target_gaussians ?? 0).toLocaleString()} / actual {lod.actual_gaussians?.toLocaleString() ?? "-"}
                        </span>
                        <span className="muted small">{formatBytes(lod.file_size)}</span>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className="stack">
                <h3>媒体数据</h3>
                {showMediaList ? (
                  <div className="media-list">
                    <div className="list-row header" style={{ gridTemplateColumns: "minmax(0, 1fr) 76px 92px" }}>
                      <span>文件名</span><span>类型</span><span>大小</span>
                    </div>
                    {media.map((item) => (
                      <div className="list-row" style={{ gridTemplateColumns: "minmax(0, 1fr) 76px 92px" }} key={item.id}>
                        <span className="truncate" title={item.file_name}>{item.file_name}</span>
                        <span>{item.kind === "image" ? "图片" : "视频"}</span>
                        <span className="muted small">{formatBytes(item.file_size)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="media-grid">
                    {media.map((item) => (
                      <div className="media-tile" title={`${item.file_name} · ${formatBytes(item.file_size)}`} key={item.id}>
                        {item.kind === "image" ? <Images size={22} /> : <Film size={22} />}
                      </div>
                    ))}
                    {!media.length ? <div className="empty-state" style={{ gridColumn: "1 / -1" }}>暂无媒体</div> : null}
                  </div>
                )}
              </section>

              <section className="stack">
                <h3>任务历史</h3>
                {tasks.length ? (
                  <div className="task-list">
                    {tasks.map((task) => (
                      <div className="list-row" style={{ gridTemplateColumns: "minmax(0, 1fr) 94px 80px 44px" }} key={task.id}>
                        <span className="truncate" title={task.current_stage}>{taskTypeLabel(task.type)} · {task.current_stage || "-"}</span>
                        <span className={`status-pill ${task.status}`}>{taskStatusLabel(task.status)}</span>
                        <span className="muted small">{formatDateTime(task.created_at)}</span>
                        <button className="icon-button" type="button" onClick={() => setLogTask(task)} aria-label="查看日志">
                          <ScrollText size={15} />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : <div className="empty-state">暂无任务</div>}
              </section>

              {logs.length ? (
                <section className="stack">
                  <h3>任务日志</h3>
                  <div className="log-console" style={{ maxHeight: showCompactLogs ? 220 : 180 }}>
                    {logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>)}
                  </div>
                </section>
              ) : null}
            </div>
          </div>
        </aside>
      </section>
      {logTask ? (
        <div className="modal-backdrop" onClick={() => setLogTask(null)}>
          <section className="modal-panel" onClick={(event) => event.stopPropagation()}>
            <div className="panel-head">
              <div>
                <h2>任务日志</h2>
                <p className="muted small">{taskTypeLabel(logTask.type)} · {logTask.current_stage || taskStatusLabel(logTask.status)}</p>
              </div>
              <button className="ghost-button" type="button" onClick={() => setLogTask(null)}>关闭</button>
            </div>
            <div className="panel-body scrollable stack">
              {logTask.error_message ? <div className="error-box">{logTask.error_message}</div> : null}
              <pre className="code-view">{formatTaskLog(logTask)}</pre>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function formatTaskLog(task: Task): string {
  const lines = [...(task.logs ?? [])];
  if (task.metrics && Object.keys(task.metrics).length > 0) {
    lines.push("metrics:");
    lines.push(JSON.stringify(task.metrics, null, 2));
  }
  if (!lines.length) return "暂无日志。";
  return lines.join("\n\n");
}
