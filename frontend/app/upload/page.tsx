"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, Eye, FileUp, Film, FolderOpen, Images, Loader2, Play, RefreshCw, Trash2, Wand2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, formatBytes, mediaFileUrl, mediaThumbnailUrl } from "@/lib/api";
import { formatDateTime, inputTypeLabel, isActiveTask, projectStatusLabel } from "@/lib/labels";
import { rememberTaskId } from "@/lib/taskTracking";
import type { MediaAsset, Project, Task, ViewerConfig } from "@/lib/types";
import { SplatViewer } from "@/components/SplatViewer";
import { TaskProgress } from "@/components/TaskProgress";

const MIN_INPUT_FRAMES = 1;
const MAX_INPUT_FRAMES = 800;
export default function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const thumbsRef = useRef<Record<string, string>>({});
  const [name, setName] = useState("新建重建项目");
  const [inputType, setInputType] = useState<Project["input_type"]>("images");
  const [previewPipeline, setPreviewPipeline] = useState<"litevggt_edgs" | "litevggt_spz">("litevggt_edgs");
  const [tags, setTags] = useState("preview, research");
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [task, setTask] = useState<Task | null>(null);
  const [viewer, setViewer] = useState<ViewerConfig | null>(null);
  const [selectedMedia, setSelectedMedia] = useState<MediaAsset | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalBytes = useMemo(() => media.reduce((sum, item) => sum + item.file_size, 0), [media]);
  const imageCount = media.filter((item) => item.kind === "image").length;
  const canStartPreview = Boolean(project && media.length > 0 && (inputType === "video" || imageCount >= MIN_INPUT_FRAMES) && !isActiveTask(task));
  const canStartFine = Boolean(project && media.length > 0 && !isActiveTask(task));

  useEffect(() => {
    return () => {
      Object.values(thumbsRef.current).forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  useEffect(() => {
    if (!task || !isActiveTask(task)) return;
    const timer = window.setInterval(() => {
      void api.task(task.id)
        .then((next) => {
          setTask(next);
          if (!isActiveTask(next) && project) {
            void refreshProject(project.id);
          }
        })
        .catch(() => undefined);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [project, task]);

  async function ensureProject() {
    if (project) return project;
    const created = await api.createProject({
      name: name.trim() || "新建重建项目",
      input_type: inputType,
      tags: tags.split(",").map((item) => item.trim()).filter(Boolean)
    });
    setProject(created);
    return created;
  }

  async function createProjectOnly() {
    setBusy(true);
    setError(null);
    try {
      await ensureProject();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建项目失败");
    } finally {
      setBusy(false);
    }
  }

  async function onFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      const active = await ensureProject();
      const uploaded: MediaAsset[] = [];
      for (const file of Array.from(files)) {
        const asset = await api.uploadMedia(active.id, file);
        uploaded.push(asset);
        if (asset.kind === "image" && file.type.startsWith("image/")) {
          const url = URL.createObjectURL(file);
          thumbsRef.current[asset.id] = url;
        }
      }
      setThumbs({ ...thumbsRef.current });
      setMedia((items) => [...items, ...uploaded]);
      await refreshProject(active.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function refreshProject(projectId: string) {
    const [projectData, viewerData] = await Promise.all([
      api.project(projectId),
      api.viewerConfig(projectId).catch(() => null)
    ]);
    setProject(projectData);
    setMedia(projectData.media ?? media);
    setViewer(viewerData);
  }

  async function startPreview() {
    if (!project) return;
    setBusy(true);
    setError(null);
    try {
      const options = inputType === "images"
        ? { preview_pipeline: previewPipeline }
        : { preview_pipeline: "lingbot_map_spark" };
      const next = await api.startPreview(project.id, options);
      rememberTaskId(next.id);
      setTask(next);
      setViewer(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "预览任务创建失败");
    } finally {
      setBusy(false);
    }
  }

  async function deleteMedia(item: MediaAsset) {
    if (!project) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteMedia(project.id, item.id);
      if (thumbsRef.current[item.id]) {
        URL.revokeObjectURL(thumbsRef.current[item.id]);
        delete thumbsRef.current[item.id];
        setThumbs({ ...thumbsRef.current });
      }
      if (selectedMedia?.id === item.id) setSelectedMedia(null);
      await refreshProject(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除素材失败");
    } finally {
      setBusy(false);
    }
  }

  async function startFine() {
    if (!project) return;
    setBusy(true);
    setError(null);
    try {
      const next = await api.startFine(project.id);
      rememberTaskId(next.id);
      setTask(next);
      setViewer(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "训练任务创建失败");
    } finally {
      setBusy(false);
    }
  }

  function handleDrop(event: React.DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    void onFiles(event.dataTransfer.files);
  }

  const activeMessage = task && isActiveTask(task)
    ? "后端正在处理真实预览任务，完成后会自动加载 SPZ 产物。"
    : viewer?.status === "ready"
      ? "预览产物已就绪。"
      : "上传满足条件的数据后即可启动真实极速预览。";

  return (
    <div className="workspace-page no-page-title">
      <section className="split-workspace">
        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>源数据集</h2>
              <p className="muted small">{media.length} 个文件 · {formatBytes(totalBytes)}</p>
            </div>
            {project ? <span className={`status-pill ${project.status}`}>{projectStatusLabel(project.status)}</span> : <span className="status-pill">未创建</span>}
          </div>

          <div className="panel-body scrollable stack">
            <div className="grid two">
              <div className="field">
                <label>项目名称</label>
                <input className="input" value={name} onChange={(event) => setName(event.target.value)} disabled={Boolean(project)} />
              </div>
              <div className="field">
                <label>输入类型</label>
                <select className="select" value={inputType} onChange={(event) => setInputType(event.target.value as Project["input_type"])} disabled={Boolean(project)}>
                  <option value="images">图片序列</option>
                  <option value="video">视频</option>
                </select>
              </div>
            </div>
            <div className="field">
              <label>标签</label>
              <input className="input" value={tags} onChange={(event) => setTags(event.target.value)} disabled={Boolean(project)} />
            </div>
            <div className="field">
              <label>预览管线</label>
              {inputType === "images" ? (
                <select
                  className="select"
                  value={previewPipeline}
                  onChange={(event) => setPreviewPipeline(event.target.value as "litevggt_edgs" | "litevggt_spz")}
                  disabled={Boolean(task && isActiveTask(task))}
                >
                  <option value="litevggt_edgs">LiteVGGT + EDGS + Spark-SPZ</option>
                  <option value="litevggt_spz">LiteVGGT 直接出 Spark-SPZ</option>
                </select>
              ) : (
                <input className="input" value="LingBot-Map + Spark-SPZ" disabled />
              )}
            </div>

            <label
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <FileUp size={26} />
              <strong>{inputType === "images" ? "选择或拖入图片序列" : "选择或拖入视频文件"}</strong>
              <span className="muted small">
                图片预览至少需要 {MIN_INPUT_FRAMES} 张，最多采样 {MAX_INPUT_FRAMES} 帧；失败任务不会生成假产物。
              </span>
              <input
                ref={fileInputRef}
                hidden
                type="file"
                multiple={inputType === "images"}
                accept={inputType === "images" ? "image/*" : "video/*"}
                onChange={(event) => void onFiles(event.target.files)}
              />
            </label>

            <div className="grid three">
              <div className="panel stat flat"><span className="muted small">文件</span><strong>{media.length}</strong></div>
              <div className="panel stat flat"><span className="muted small">图片</span><strong>{imageCount}</strong></div>
              <div className="panel stat flat"><span className="muted small">大小</span><strong>{formatBytes(totalBytes)}</strong></div>
            </div>

            <div className="media-grid dense">
              {media.map((item) => {
                const thumb = thumbs[item.id] ?? mediaThumbnailUrl(item);
                return (
                  <div
                    className="media-tile selectable"
                    title={`${item.file_name} · ${formatBytes(item.file_size)}`}
                    key={item.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => item.kind === "image" ? setSelectedMedia(item) : undefined}
                    onKeyDown={(event) => {
                      if ((event.key === "Enter" || event.key === " ") && item.kind === "image") setSelectedMedia(item);
                    }}
                  >
                    {thumb ? <img src={thumb} alt={item.file_name} /> : item.kind === "image" ? <Images size={24} /> : <Film size={24} />}
                    <span className="media-name" title={item.file_name}>{item.file_name}</span>
                    <span className="media-kind">{item.kind === "image" ? "图片" : "视频"}</span>
                    <button
                      className="media-delete"
                      type="button"
                      aria-label="删除素材"
                      disabled={busy}
                      onClick={(event) => {
                        event.stopPropagation();
                        void deleteMedia(item);
                      }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                );
              })}
              {media.length === 0 ? <div className="empty-state" style={{ gridColumn: "1 / -1" }}>等待上传真实素材</div> : null}
            </div>

            {inputType === "images" && imageCount > 0 && imageCount < MIN_INPUT_FRAMES ? (
              <div className="error-box"><AlertTriangle size={16} /> 启动预览前至少上传 {MIN_INPUT_FRAMES} 张图片。</div>
            ) : null}
            {error ? <div className="error-box">{error}</div> : null}
          </div>

          <div className="sticky-actions">
            <button className="ghost-button" type="button" onClick={() => void createProjectOnly()} disabled={Boolean(project) || busy}>
              {busy && !project ? <Loader2 size={17} /> : <CheckCircle2 size={17} />}创建项目
            </button>
            <button className="ghost-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={busy}>
              <FileUp size={17} />继续上传
            </button>
            <button className="button secondary" type="button" onClick={() => void startPreview()} disabled={!canStartPreview || busy}>
              {busy ? <RefreshCw size={17} /> : <Play size={17} />}启动预览
            </button>
            <button className="button" type="button" onClick={() => void startFine()} disabled={!canStartFine || busy}>
              {busy ? <RefreshCw size={17} /> : <Wand2 size={17} />}直接训练
            </button>
          </div>
        </div>

        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>真实 3D 预览</h2>
              <p className="muted small">{activeMessage}</p>
            </div>
            {viewer?.status === "ready" ? <span className="status-pill ready">SPZ</span> : <span className="status-pill">{task ? task.status : "idle"}</span>}
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            {viewer?.stale ? <div className="notice-box preview-stale">{viewer.message}</div> : null}
            {viewer?.status === "ready" ? (
              <SplatViewer modelUrl={viewer.model_url} segments={viewer.segments} />
            ) : (
              <div className="preview-stage">
                <div className="preview-placeholder">
                  <span className="preview-icon">{task && isActiveTask(task) ? <Loader2 size={28} /> : <Eye size={28} />}</span>
                  <h2>{task && isActiveTask(task) ? "预览生成中" : "等待真实产物"}</h2>
                  <p className="muted">{activeMessage}</p>
                  <TaskProgress task={task} />
                </div>
              </div>
            )}
          </div>
          <div className="sticky-actions">
            <div className="muted small">
              {project ? `${project.name} · ${inputTypeLabel(project.input_type)} · ${formatDateTime(project.updated_at)}` : "项目创建后会在这里显示真实状态。"}
            </div>
            {project ? <Link className="ghost-button" href={`/projects/${project.id}`}><FolderOpen size={17} />项目详情</Link> : null}
          </div>
        </div>
      </section>
      {selectedMedia ? (
        <div className="modal-backdrop" onClick={() => setSelectedMedia(null)}>
          <section className="modal-panel image-modal" onClick={(event) => event.stopPropagation()}>
            <div className="panel-head">
              <div>
                <h2 className="truncate" title={selectedMedia.file_name}>{selectedMedia.file_name}</h2>
                <p className="muted small">{formatBytes(selectedMedia.file_size)}</p>
              </div>
              <button className="icon-button" type="button" onClick={() => setSelectedMedia(null)} aria-label="关闭">
                <X size={17} />
              </button>
            </div>
            <div className="image-preview-body">
              <img src={mediaFileUrl(selectedMedia)} alt={selectedMedia.file_name} />
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
