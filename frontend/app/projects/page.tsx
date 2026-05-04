"use client";

import Link from "next/link";
import { FilePlus2, Image, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import { formatDateTime, inputTypeLabel, projectStatusLabel } from "@/lib/labels";
import type { Project, ProjectStatus } from "@/lib/types";

const PROJECT_LIST_THRESHOLD = 9;

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<ProjectStatus | "all">("all");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.projects()
      .then((data) => setProjects(data.projects))
      .catch((err) => setError(err instanceof Error ? err.message : "读取项目失败"));
  }, []);

  const visible = useMemo(() => {
    return projects.filter((project) => {
      const matchesQuery = project.name.toLowerCase().includes(query.toLowerCase());
      const matchesStatus = status === "all" || project.status === status;
      return matchesQuery && matchesStatus;
    });
  }, [projects, query, status]);

  const listMode = visible.length > PROJECT_LIST_THRESHOLD;

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
        <div className="panel-head">
          <div className="row" style={{ flex: 1 }}>
            <Search size={18} className="muted" />
            <input className="input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索项目名称" />
          </div>
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
        </div>

        <div className="panel-body scrollable">
          {error ? <div className="error-box">{error}</div> : null}
          {visible.length === 0 && !error ? <div className="empty-state">暂无匹配项目</div> : null}

          {visible.length > 0 && listMode ? (
            <div className="project-list">
              <div className="list-row header" style={{ gridTemplateColumns: "minmax(0, 1.5fr) 150px 140px 110px 112px" }}>
                <span>项目</span><span>状态</span><span>输入</span><span>占用</span><span>更新</span>
              </div>
              {visible.map((project) => (
                <Link className="list-row" style={{ gridTemplateColumns: "minmax(0, 1.5fr) 150px 140px 110px 112px" }} href={`/projects/${project.id}`} key={project.id}>
                  <span className="truncate" title={project.name}><strong>{project.name}</strong></span>
                  <span className={`status-pill ${project.status}`}>{projectStatusLabel(project.status)}</span>
                  <span>{inputTypeLabel(project.input_type)}</span>
                  <span className="muted small">{formatBytes(project.total_size_bytes)}</span>
                  <span className="muted small">{formatDateTime(project.updated_at)}</span>
                </Link>
              ))}
            </div>
          ) : null}

          {visible.length > 0 && !listMode ? (
            <div className="project-grid">
              {visible.map((project) => (
                <Link className="panel project-card" href={`/projects/${project.id}`} key={project.id}>
                  <div className="preview-tile">
                    <div className="stack" style={{ placeItems: "center", textAlign: "center" }}>
                      <Image size={30} />
                      <strong>{project.status === "PREVIEW_READY" ? "SPZ READY" : inputTypeLabel(project.input_type)}</strong>
                    </div>
                  </div>
                  <div className="row between">
                    <h3 className="truncate" title={project.name}>{project.name}</h3>
                    <span className={`status-pill ${project.status}`}>{projectStatusLabel(project.status)}</span>
                  </div>
                  <p className="muted small truncate" title={project.tags.join(" / ")}>
                    {project.tags.join(" / ") || "无标签"}
                  </p>
                  <div className="row between small muted">
                    <span>{formatBytes(project.total_size_bytes)}</span>
                    <span>{formatDateTime(project.updated_at)}</span>
                  </div>
                </Link>
              ))}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
