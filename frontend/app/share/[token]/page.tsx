"use client";

import { Download, FileArchive, Tag } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { SplatViewer } from "@/components/SplatViewer";
import { api, artifactUrl, downloadFileWithProgress, formatBytes } from "@/lib/api";
import type { TransferProgress } from "@/lib/api";
import { formatDateTime } from "@/lib/labels";
import type { SharedProject } from "@/lib/types";

export default function SharedProjectPage() {
  const params = useParams<{ token: string }>();
  const [project, setProject] = useState<SharedProject | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<TransferProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.token) return;
    void loadSharedProject(params.token);
  }, [params.token]);

  async function loadSharedProject(token: string) {
    setError(null);
    try {
      setProject(await api.sharedProject(token));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load shared project");
    }
  }

  async function downloadShared(url: string | null | undefined, fileName: string) {
    if (!url) return;
    setError(null);
    setDownloadProgress(null);
    try {
      await downloadFileWithProgress(artifactUrl(url), fileName, 0, setDownloadProgress);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    } finally {
      window.setTimeout(() => setDownloadProgress(null), 800);
    }
  }

  const viewer = project?.viewer;
  const displaySize = viewer?.file_size ?? project?.total_size_bytes ?? 0;

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Shared Project</p>
          <h1 className="truncate" title={project?.name}>{project?.name ?? "Loading shared project"}</h1>
          {project ? <p className="muted small">Created {formatDateTime(project.created_at)} / Updated {formatDateTime(project.updated_at)}</p> : null}
        </div>
      </header>

      <section className="detail-grid">
        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>3D Viewer</h2>
              <p className="muted small">{viewer?.status === "ready" ? viewerSourceLabel(viewer.source) : viewer?.message ?? "Waiting for model"}</p>
            </div>
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            <SplatViewer
              modelUrl={viewer?.status === "ready" ? viewer.model_url : null}
              format={viewer?.format}
              viewerMetaUrl={viewer?.status === "ready" ? viewer.viewer_meta_url ?? viewer.preview_meta_url : null}
              gaussianPlyUrl={viewer?.status === "ready" ? viewer.gaussian_ply_url : null}
              debugPointsUrl={viewer?.status === "ready" ? viewer.debug_points_ply_url : null}
              cameraPathUrl={viewer?.status === "ready" ? viewer.camera_path_url : null}
              defaultViewMode={viewer?.status === "ready" && viewer.point_source ? "points" : "splats"}
            />
          </div>
        </div>

        <aside className="detail-side">
          <section className="grid three">
            <div className="panel stat"><span className="muted small">Name</span><strong className="truncate" title={project?.name}>{project?.name ?? "-"}</strong></div>
            <div className="panel stat"><span className="muted small">Updated</span><strong>{project ? formatShortDate(project.updated_at) : "-"}</strong></div>
            <div className="panel stat"><span className="muted small">Size</span><strong>{formatBytes(displaySize)}</strong></div>
          </section>

          <div className="panel fill">
            <div className="panel-head">
              <div>
                <h2>Project Info</h2>
                <p className="muted small">{viewer?.status === "ready" ? viewerSourceLabel(viewer.source) : "Shared model"}</p>
              </div>
            </div>
            <div className="panel-body scrollable stack">
              {error ? <div className="error-box">{error}</div> : null}
              <DownloadProgressBar progress={downloadProgress} />

              <div className="share-meta-list">
                <div className="list-row"><span className="muted small">Name</span><strong className="truncate" title={project?.name}>{project?.name ?? "-"}</strong></div>
                <div className="list-row"><span className="muted small">Created</span><span>{project ? formatDateTime(project.created_at) : "-"}</span></div>
                <div className="list-row"><span className="muted small">Updated</span><span>{project ? formatDateTime(project.updated_at) : "-"}</span></div>
                <div className="list-row"><span className="muted small">Source</span><span>{viewerSourceLabel(viewer?.source)}</span></div>
                <div className="list-row"><span className="muted small">Size</span><span>{formatBytes(displaySize)}</span></div>
              </div>

              <section className="stack">
                <h3>Tags</h3>
                {project?.tags?.length ? (
                  <div className="tag-list compact-tag-list">
                    {project.tags.map((tag) => (
                      <span className="tag-pill" key={tag}><Tag size={13} />{tag}</span>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">No tags</div>
                )}
              </section>

              <section className="stack">
                <h3>Downloads</h3>
                <div className="actions">
                  <button className="ghost-button" type="button" disabled={!viewer?.download_spz_url} onClick={() => void downloadShared(viewer?.download_spz_url, "model.spz")}>
                    <Download size={17} />SPZ
                  </button>
                  <button className="ghost-button" type="button" disabled={!viewer?.download_ply_url} onClick={() => void downloadShared(viewer?.download_ply_url, "model.ply")}>
                    <FileArchive size={17} />PLY
                  </button>
                </div>
              </section>
            </div>
          </div>
        </aside>
      </section>
    </div>
  );
}

function viewerSourceLabel(source: SharedProject["viewer"]["source"] | undefined): string {
  if (source === "final") return "Final reconstruction";
  if (source === "preview") return "Fast preview";
  return "Shared model";
}

function formatShortDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleDateString();
}

function DownloadProgressBar({ progress }: { progress: TransferProgress | null }) {
  if (!progress) return null;
  return (
    <div className="transfer-progress">
      <div className="row between">
        <span className="truncate" title={progress.fileName}>Downloading {progress.fileName}</span>
        <span className="muted small">{progress.percent}%</span>
      </div>
      <div className="progress-track" aria-label="Download progress">
        <span style={{ width: `${progress.percent}%` }} />
      </div>
      <div className="muted small">
        {formatBytes(progress.loadedBytes)} / {progress.totalBytes ? formatBytes(progress.totalBytes) : "calculating"}
      </div>
    </div>
  );
}
