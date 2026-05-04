"use client";

import Link from "next/link";
import { Camera, CircleStop, Loader2, Play, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { SplatViewer } from "@/components/SplatViewer";
import { TaskProgress } from "@/components/TaskProgress";
import { api, projectEventsUrl } from "@/lib/api";
import { rememberTaskId } from "@/lib/taskTracking";
import type { Project, Task, ViewerConfig } from "@/lib/types";

const SEGMENT_MS = 5000;

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const segmentIndexRef = useRef(0);
  const startedAtRef = useRef(0);
  const segmentStartRef = useRef(0);

  const [project, setProject] = useState<Project | null>(null);
  const [viewer, setViewer] = useState<ViewerConfig | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState(0);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      if (recorderRef.current?.state && recorderRef.current.state !== "inactive") recorderRef.current.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startCamera() {
    setBusy(true);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      const session = await api.createCameraSession({ name: `Realtime camera ${new Date().toLocaleString("zh-CN")}` });
      setProject(session);
      connectEvents(session.id);

      segmentIndexRef.current = 0;
      startedAtRef.current = performance.now();
      segmentStartRef.current = 0;
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) void uploadSegment(session.id, event.data);
      };
      recorder.start(SEGMENT_MS);
      recorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法启动摄像头");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    } finally {
      setBusy(false);
    }
  }

  async function stopCamera() {
    setBusy(true);
    setError(null);
    try {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setRecording(false);
      if (project) {
        const finished = await api.finishCameraSession(project.id);
        setProject(finished);
        await refreshViewer(project.id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "结束实时预览失败");
    } finally {
      setBusy(false);
    }
  }

  async function uploadSegment(projectId: string, blob: Blob) {
    const nowSeconds = (performance.now() - startedAtRef.current) / 1000;
    const segmentIndex = segmentIndexRef.current;
    const segmentStart = segmentStartRef.current;
    segmentIndexRef.current += 1;
    segmentStartRef.current = nowSeconds;
    try {
      const result = await api.uploadCameraChunk(projectId, blob, {
        segment_index: segmentIndex,
        segment_start_seconds: segmentStart,
        segment_end_seconds: nowSeconds
      });
      rememberTaskId(result.task.id);
      setTask(result.task);
      setSegments((value) => Math.max(value, segmentIndex + 1));
    } catch (err) {
      setError(err instanceof Error ? err.message : "摄像头分片上传失败");
    }
  }

  function connectEvents(projectId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(projectEventsUrl(projectId));
    source.addEventListener("project_snapshot", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as Project;
        setProject(payload);
        const latestTask = payload.tasks?.[0];
        if (latestTask) setTask(latestTask);
      } catch {
        return;
      }
    });
    source.addEventListener("preview_segment_ready", () => {
      void refreshViewer(projectId);
    });
    source.addEventListener("task_succeeded", () => {
      void refreshViewer(projectId);
    });
    source.addEventListener("task_failed", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as Task;
        setTask(payload);
      } catch {
        return;
      }
    });
    eventSourceRef.current = source;
  }

  async function refreshViewer(projectId: string) {
    const next = await api.viewerConfig(projectId).catch(() => null);
    setViewer(next);
  }

  return (
    <div className="workspace-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Realtime</p>
          <h1>实时视频重建</h1>
        </div>
        <div className="actions">
          <Link className="button" href="/upload"><UploadCloud size={18} />改用离线上传</Link>
        </div>
      </header>

      <section className="split-workspace">
        <div className="panel fill">
          <div className="panel-head">
            <h2>摄像头通道</h2>
            <span className={`status-pill ${recording ? "running" : viewer?.status === "ready" ? "ready" : ""}`}>
              {recording ? "采集中" : viewer?.status === "ready" ? "可预览" : "待启动"}
            </span>
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            <div className="camera-stage">
              <video ref={videoRef} className="camera-video" autoPlay muted playsInline />
              {!recording && !streamRef.current ? (
                <div className="camera-empty">
                  <Camera size={30} />
                  <strong>等待摄像头输入</strong>
                </div>
              ) : null}
            </div>
          </div>
          <div className="sticky-actions">
            <button className="button secondary" type="button" onClick={() => void startCamera()} disabled={recording || busy}>
              {busy && !recording ? <Loader2 size={17} /> : <Play size={17} />}连接设备
            </button>
            <button className="danger-button" type="button" onClick={() => void stopCamera()} disabled={!recording || busy}>
              <CircleStop size={17} />结束采集
            </button>
          </div>
        </div>

        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>LingBot-Map 增量预览</h2>
              <p className="muted small">{segments ? `已提交 ${segments} 个时间窗口` : "每 5 秒提交一个窗口，Worker 完成后增量加载"}</p>
            </div>
            <span className="status-pill">{viewer?.mode ?? "progressive"}</span>
          </div>
          <div className="panel-body scrollable" style={{ padding: 0 }}>
            {viewer?.status === "ready" ? (
              <SplatViewer modelUrl={viewer.model_url} segments={viewer.segments} />
            ) : (
              <div className="preview-stage">
                <div className="preview-placeholder">
                  <span className="preview-icon">{busy || recording ? <Loader2 size={28} /> : <Camera size={28} />}</span>
                  <h2>{recording ? "等待增量片段" : "尚未开始实时预览"}</h2>
                  <p className="muted">实时摄像头预览使用 LingBot-Map streaming 模式生成窗口级 SPZ 片段。</p>
                  <TaskProgress task={task} />
                  {error ? <div className="error-box">{error}</div> : null}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
