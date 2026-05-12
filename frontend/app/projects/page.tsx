"use client";

import Link from "next/link";
import { FilePlus2, Image, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes, mediaThumbnailUrl } from "@/lib/api";
import { formatDateTime, formatEta, inputTypeLabel, projectStatusLabel, taskTypeLabel } from "@/lib/labels";
import type { Project, ProjectStatus, Task } from "@/lib/types";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<ProjectStatus | "all">("all");
  const [selectedProjectIds, setSelectedProjectIds] = useState<Set<string>>(() => new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [brokenCoverIds, setBrokenCoverIds] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    void api.projects()
      .then((data) => {
        setProjects(data.projects);
        setBrokenCoverIds(new Set());
      })
      .catch((err) => setError(err instanceof Error ? err.message : "读取项目失败"));
  }, []);

  useEffect(() => {
    const projectIds = new Set(projects.map((project) => project.id));
    setSelectedProjectIds((ids) => new Set([...ids].filter((id) => projectIds.has(id))));
  }, [projects]);

  const visible = useMemo(() => {
    return projects.filter((project) => {
      const matchesQuery = projectMatchesQuery(project, query);
      const matchesStatus = status === "all" || project.status === status;
      return matchesQuery && matchesStatus;
    });
  }, [projects, query, status]);

  const visibleProjectIds = useMemo(() => visible.map((project) => project.id), [visible]);
  const selectedCount = selectedProjectIds.size;
  const allVisibleSelected = visibleProjectIds.length > 0 && visibleProjectIds.every((id) => selectedProjectIds.has(id));

  function toggleProject(projectId: string, selected: boolean) {
    setSelectedProjectIds((ids) => {
      const next = new Set(ids);
      if (selected) next.add(projectId);
      else next.delete(projectId);
      return next;
    });
  }

  function toggleVisibleProjects(selected: boolean) {
    setSelectedProjectIds((ids) => {
      const next = new Set(ids);
      for (const id of visibleProjectIds) {
        if (selected) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  }

  async function deleteSelectedProjects() {
    if (!selectedCount) return;
    if (!window.confirm(`确定删除选中的 ${selectedCount} 个项目吗？此操作无法撤销。`)) return;
    setBusy(true);
    setError(null);
    try {
      const projectIds = [...selectedProjectIds];
      const result = await api.deleteProjects(projectIds);
      const deletedIds = new Set(result.project_ids);
      setProjects((items) => items.filter((project) => !deletedIds.has(project.id)));
      setSelectedProjectIds(new Set());
      setBrokenCoverIds(new Set());
    } catch (err) {
      setError(err instanceof Error ? err.message : "批量删除项目失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Projects</p>
          <h1>项目控制台</h1>
        </div>
        <Link className="button" href="/upload"><FilePlus2 size={18} />新建项目</Link>
      </header>

      <section className="panel fill">
        <div className="panel-head project-toolbar">
          <div className="row project-search">
            <Search size={18} className="muted" />
            <input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索项目名称或标签" />
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={allVisibleSelected}
              disabled={busy || visibleProjectIds.length === 0}
              onChange={(event) => toggleVisibleProjects(event.target.checked)}
            />
            <span>选择当前结果</span>
          </label>
          <div className="field" style={{ minWidth: 180 }}>
            <select className="select" value={status} onChange={(event) => setStatus(event.target.value as ProjectStatus | "all")}>
              <option value="all">全部状态</option>
              <option value="CREATED">已创建</option>
              <option value="UPLOADING">上传中</option>
              <option value="PREVIEW_RUNNING">预览生成中</option>
              <option value="PREVIEW_READY">预览就绪</option>
              <option value="FAILED">失败</option>
              <option value="CANCELED">已取消</option>
            </select>
          </div>
          <button className="danger-button" type="button" onClick={() => void deleteSelectedProjects()} disabled={busy || selectedCount === 0}>
            <Trash2 size={16} />删除选中{selectedCount ? ` (${selectedCount})` : ""}
          </button>
        </div>

        <div className="panel-body scrollable">
          {error ? <div className="error-box">{error}</div> : null}
          {visible.length === 0 && !error ? <div className="empty-state">暂无匹配项目</div> : null}

          {visible.length > 0 ? (
            <div className="project-grid">
              {visible.map((project) => {
                const cover = projectCover(project, brokenCoverIds);
                const selected = selectedProjectIds.has(project.id);
                const activeTask = projectActiveTask(project);
                return (
                  <article className={`panel project-card${selected ? " selected" : ""}`} key={project.id}>
                    <label className="project-card-select" onClick={(event) => event.stopPropagation()}>
                      <input
                        className="project-checkbox"
                        type="checkbox"
                        aria-label={`选择项目 ${project.name}`}
                        checked={selected}
                        disabled={busy}
                        onChange={(event) => toggleProject(project.id, event.target.checked)}
                      />
                    </label>
                    <Link className="project-card-link" href={`/projects/${project.id}`}>
                      <div className={`preview-tile${cover ? " has-cover" : ""}`}>
                        {cover ? (
                          <img
                            className="project-cover-image"
                            src={cover.url}
                            alt={project.name}
                            onError={() => setBrokenCoverIds((ids) => new Set(ids).add(cover.mediaId))}
                          />
                        ) : (
                          <div className="stack" style={{ placeItems: "center", textAlign: "center" }}>
                            <Image size={30} />
                            <strong>{project.status === "PREVIEW_READY" ? "SPZ READY" : inputTypeLabel(project.input_type)}</strong>
                          </div>
                        )}
                      </div>
                      <div className="row between">
                        <h3 className="truncate" title={project.name}>{project.name}</h3>
                        <span className={`status-pill ${project.status}`}>{projectStatusLabel(project.status)}</span>
                      </div>
                      <p className="project-card-tags muted small truncate" title={projectTagsLabel(project)}>
                        {projectTagsLabel(project)}
                      </p>
                      <div className="project-card-meta small muted">
                        <span>{formatDateTime(project.updated_at)}</span>
                        <span>{inputTypeLabel(project.input_type)}</span>
                        <span>{formatBytes(project.total_size_bytes)}</span>
                      </div>
                      <ProjectTrainingProgress task={activeTask} />
                    </Link>
                  </article>
                );
              })}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

function projectMatchesQuery(project: Project, query: string): boolean {
  const terms = query.toLowerCase().split(/[\s,]+/).filter(Boolean);
  if (!terms.length) return true;
  const haystack = [project.name, ...(project.tags ?? [])].join(" ").toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

function projectTagsLabel(project: Project): string {
  return project.tags.join(" / ") || "无标签";
}

function projectActiveTask(project: Project): Task | null {
  return project.tasks?.find((task) => task.status === "queued" || task.status === "running") ?? null;
}

function projectCover(project: Project, brokenCoverIds: Set<string>): { mediaId: string; url: string } | null {
  const media = project.media?.find((item) => item.kind === "image" && item.thumbnail_uri && !brokenCoverIds.has(item.id))
    ?? project.media?.find((item) => item.thumbnail_uri && !brokenCoverIds.has(item.id));
  const url = media ? mediaThumbnailUrl(media) : null;
  return media && url ? { mediaId: media.id, url } : null;
}

function ProjectTrainingProgress({ task }: { task: Task | null }) {
  if (!task) return null;
  const progress = Math.max(0, Math.min(100, Math.round(task.progress || 0)));
  return (
    <div className="project-training">
      <div className="row between small">
        <span className="truncate" title={task.current_stage || task.type}>
          {taskTypeLabel(task.type)} · {task.current_stage || "等待调度"}
        </span>
        <span>{progress}%</span>
      </div>
      <div className="progress-track" aria-label="训练进度">
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="row between small muted">
        <span>{task.status === "queued" ? "排队中" : "训练中"}</span>
        <span>剩余 {formatEta(task.eta_seconds)}</span>
      </div>
    </div>
  );
}
