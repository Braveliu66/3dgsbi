"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, Eye, FileUp, Film, FolderOpen, Images, Loader2, Play, RefreshCw, Wand2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, formatBytes } from "@/lib/api";
import { formatDateTime, inputTypeLabel, isActiveTask, projectStatusLabel } from "@/lib/labels";
import { rememberTaskId } from "@/lib/taskTracking";
import type { MediaAsset, Project, Task, ViewerConfig } from "@/lib/types";
import { SplatViewer } from "@/components/SplatViewer";
import { TaskProgress } from "@/components/TaskProgress";

const MIN_INPUT_FRAMES = 1;
const MAX_INPUT_FRAMES = 800;
const MEDIA_LIST_THRESHOLD = 18;

export default function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const thumbsRef = useRef<Record<string, string>>({});
  const [name, setName] = useState("新建重建项目");
  const [inputType, setInputType] = useState<Project["input_type"]>("images");
  const [previewPipeline, setPreviewPipeline] = useState<"edgs" | "litevggt_spark">("edgs");
  const [tags, setTags] = useState("preview, research");
  const [project, setProject] = useState<Project | null>(null);
  const [media, setMedia] = useState<MediaAsset[]>([]);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [task, setTask] = useState<Task | null>(null);
  const [viewer, setViewer] = useState<ViewerConfig | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalBytes = useMemo(() => media.reduce((sum, item) => sum + item.file_size, 0), [media]);
  const imageCount = media.filter((item) => item.kind === "image").length;
  const canStartPreview = Boolean(project && media.length > 0 && (inputType === "video" || imageCount >= MIN_INPUT_FRAMES) && !isActiveTask(task));
  const canStartFine = Boolean(project && media.length > 0 && !isActiveTask(task));
  const showMediaList = media.length > MEDIA_LIST_THRESHOLD;

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
                  onChange={(event) => setPreviewPipeline(event.target.value as "edgs" | "litevggt_spark")}
                  disabled={Boolean(task && isActiveTask(task))}
                >
                  <option value="edgs">LiteVGGT + EDGS + Spark-SPZ</option>
                  <option value="litevggt_spark">LiteVGGT + Spark-SPZ 粗预览</option>
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

            {showMediaList ? (
              <div className="media-list">
                <div className="list-row header" style={{ gridTemplateColumns: "minmax(0, 1.3fr) 90px 96px" }}>
                  <span>文件名</span><span>类型</span><span>大小</span>
                </div>
                {media.map((item) => (
                  <div className="list-row" style={{ gridTemplateColumns: "minmax(0, 1.3fr) 90px 96px" }} key={item.id}>
                    <span className="truncate" title={item.file_name}>{item.file_name}</span>
                    <span>{inputTypeLabel(item.kind === "image" ? "images" : "video")}</span>
                    <span className="muted small">{formatBytes(item.file_size)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="media-grid">
                {media.map((item) => (
                  <div className="media-tile" title={`${item.file_name} · ${formatBytes(item.file_size)}`} key={item.id}>
                    {thumbs[item.id] ? <img src={thumbs[item.id]} alt={item.file_name} /> : item.kind === "image" ? <Images size={24} /> : <Film size={24} />}
                  </div>
                ))}
                {media.length === 0 ? <div className="empty-state" style={{ gridColumn: "1 / -1" }}>等待上传真实素材</div> : null}
              </div>
            )}

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
    </div>
  );
}
