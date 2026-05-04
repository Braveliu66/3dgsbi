"use client";

import { Maximize2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Object3D, WebGLRenderer } from "three";
import { artifactUrl } from "@/lib/api";
import type { ViewerSegment } from "@/lib/types";

type ViewerState = "idle" | "loading" | "ready" | "error";

interface SparkRendererLike {
  lodSplatScale?: number;
  maxStdDev?: number;
  maxPixelRadius?: number;
  dispose?: () => void;
}

interface SplatMeshLike {
  rotation: { y: number };
  lodScale?: number;
  numSplats?: number;
  dispose?: () => void;
}

interface QualityLevel {
  label: string;
  pixelRatio: number;
  lodSplatScale: number;
  meshLodScale: number;
  maxStdDev: number;
  maxPixelRadius: number;
}

const QUALITY_LEVELS: QualityLevel[] = [
  { label: "speed", pixelRatio: 0.65, lodSplatScale: 0.45, meshLodScale: 0.75, maxStdDev: Math.sqrt(5), maxPixelRadius: 192 },
  { label: "balanced", pixelRatio: 0.8, lodSplatScale: 0.65, meshLodScale: 0.9, maxStdDev: Math.sqrt(6), maxPixelRadius: 256 },
  { label: "normal", pixelRatio: 1, lodSplatScale: 0.85, meshLodScale: 1, maxStdDev: Math.sqrt(7), maxPixelRadius: 384 },
  { label: "sharp", pixelRatio: 1.15, lodSplatScale: 1, meshLodScale: 1.1, maxStdDev: Math.sqrt(8), maxPixelRadius: 512 },
  { label: "max", pixelRatio: 1.3, lodSplatScale: 1.2, meshLodScale: 1.2, maxStdDev: Math.sqrt(8), maxPixelRadius: 512 }
];

const TARGET_FPS = readNumber(process.env.VIEWER_TARGET_FPS, 90);
const QUALITY_UP_FPS = readNumber(process.env.VIEWER_QUALITY_UP_FPS, 105);
const QUALITY_DOWN_FPS = readNumber(process.env.VIEWER_QUALITY_DOWN_FPS, 90);
const ADAPTIVE_QUALITY = (process.env.VIEWER_ADAPTIVE_QUALITY ?? "true").toLowerCase() !== "false";
const MAX_RENDER_SPLATS = readNumber(process.env.VIEWER_MAX_SPLATS, 5_000_000);

export function SplatViewer({ modelUrl, segments }: { modelUrl?: string | null; segments?: ViewerSegment[] }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<ViewerState>("idle");
  const [message, setMessage] = useState("真实 preview_spz 产物会加载在这里。");
  const [fps, setFps] = useState(0);
  const [qualityIndex, setQualityIndex] = useState(1);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [activeSegment, setActiveSegment] = useState(0);
  const segmentList = [...(segments ?? [])].sort((a, b) => a.segment_index - b.segment_index);
  const segmentKey = segmentList.map((segment) => `${segment.artifact_id}:${segment.segment_index}`).join("|");
  const hasSegments = segmentList.length > 0;

  useEffect(() => {
    if (segmentList.length) setActiveSegment(segmentList.length - 1);
  }, [segmentKey]);

  useEffect(() => {
    const urls = hasSegments
      ? segmentList.slice(0, activeSegment + 1).map((segment) => segment.model_url)
      : modelUrl
        ? [modelUrl]
        : [];
    if (!urls.length || !hostRef.current) {
      setState("idle");
      setMessage("真实 preview_spz 产物会加载在这里。");
      setFps(0);
      setSplatCount(null);
      return;
    }

    let cancelled = false;
    let animationFrame = 0;
    let resizeObserver: ResizeObserver | undefined;
    let cleanup: (() => void) | undefined;
    const qualityRef = { current: qualityIndex };
    const fpsWindow = { startedAt: performance.now(), frames: 0, highStreak: 0, lowStreak: 0 };

    async function mountViewer() {
      try {
        setState("loading");
        setMessage("正在初始化 Spark Viewer");
        const THREE = await import("three");
        const Spark = (await import("@sparkjsdev/spark") as unknown) as {
          SplatMesh?: new (options: { url: string; lod?: boolean | "quality"; lodAbove?: number }) => SplatMeshLike;
          SparkRenderer?: new (options: Record<string, unknown>) => SparkRendererLike;
        };
        const SplatMesh = Spark.SplatMesh;
        if (!SplatMesh) throw new Error("@sparkjsdev/spark did not expose SplatMesh");

        const host = hostRef.current;
        if (!host || cancelled) return;
        host.innerHTML = "";

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(55, host.clientWidth / Math.max(host.clientHeight, 1), 0.01, 1000);
        camera.position.set(0, 0, 3);
        const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
        renderer.setSize(host.clientWidth, host.clientHeight);
        host.appendChild(renderer.domElement);

        const initialQuality = QUALITY_LEVELS[qualityRef.current];
        const sparkRenderer = Spark.SparkRenderer
          ? new Spark.SparkRenderer({
              renderer,
              lodSplatScale: initialQuality.lodSplatScale,
              maxStdDev: initialQuality.maxStdDev,
              maxPixelRadius: initialQuality.maxPixelRadius
            })
          : null;
        if (sparkRenderer) scene.add(sparkRenderer as unknown as Object3D);

        const weakNetwork = isWeakNetwork();
        const splats = urls.map((url, index) => {
          const splat = new SplatMesh({
            url: artifactUrl(url),
            lod: true,
            lodAbove: weakNetwork || index < urls.length - 1 ? 50000 : 100000
          });
          scene.add(splat as unknown as Object3D);
          applyQuality(renderer, sparkRenderer, splat, initialQuality, host);
          return splat;
        });

        resizeObserver = new ResizeObserver(() => {
          if (!host.clientWidth || !host.clientHeight) return;
          camera.aspect = host.clientWidth / host.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(host.clientWidth, host.clientHeight);
          for (const splat of splats) applyQuality(renderer, sparkRenderer, splat, QUALITY_LEVELS[qualityRef.current], host);
        });
        resizeObserver.observe(host);

        const render = (now: number) => {
          animationFrame = requestAnimationFrame(render);
          for (const splat of splats) splat.rotation.y += 0.002;
          renderer.render(scene, camera);
          const totalSplats = splats.reduce((sum, splat) => sum + (typeof splat.numSplats === "number" ? splat.numSplats : 0), 0);
          updateFps(now, fpsWindow, qualityRef, renderer, sparkRenderer, splats, host, setFps, setQualityIndex, totalSplats);
          if (totalSplats > 0) setSplatCount(totalSplats);
        };
        animationFrame = requestAnimationFrame(render);

        cleanup = () => {
          cancelAnimationFrame(animationFrame);
          resizeObserver?.disconnect();
          for (const splat of splats) splat.dispose?.();
          sparkRenderer?.dispose?.();
          renderer.dispose();
          host.innerHTML = "";
        };
        setState("ready");
        setMessage(hasSegments ? `Spark Viewer 已加载 ${urls.length} 个增量 SPZ 片段。` : "Spark Viewer 已加载真实 SPZ 产物。");
      } catch (error) {
        setState("error");
        setMessage(error instanceof Error ? error.message : "Spark Viewer 加载失败");
      }
    }

    void mountViewer();
    return () => {
      cancelled = true;
      cleanup?.();
    };
  }, [modelUrl, segmentKey, activeSegment, hasSegments]);

  const quality = QUALITY_LEVELS[qualityIndex] ?? QUALITY_LEVELS[0];
  return (
    <section className="viewer-shell">
      <div ref={hostRef} className="viewer-canvas" />
      {hasSegments ? (
        <div className="viewer-timeline">
          <input
            type="range"
            min={0}
            max={Math.max(segmentList.length - 1, 0)}
            value={activeSegment}
            onChange={(event) => setActiveSegment(Number(event.target.value))}
            aria-label="预览时间线"
          />
          <span>{Math.round(((activeSegment + 1) / Math.max(segmentList.length, 1)) * 100)}%</span>
        </div>
      ) : null}
      <div className={`viewer-overlay ${state}`}>
        <span>{message}</span>
        <span className="viewer-stats">
          {state === "ready" ? `${Math.round(fps)} FPS / ${quality.label} / target ${TARGET_FPS}` : quality.label}
          {splatCount ? ` / ${splatCount.toLocaleString()} splats` : ""}
        </span>
        <button className="icon-button" type="button" onClick={() => hostRef.current?.requestFullscreen?.()} aria-label="Fullscreen">
          <Maximize2 size={17} />
        </button>
      </div>
    </section>
  );
}

function updateFps(
  now: number,
  fpsWindow: { startedAt: number; frames: number; highStreak: number; lowStreak: number },
  qualityRef: { current: number },
  renderer: WebGLRenderer,
  sparkRenderer: SparkRendererLike | null,
  splats: SplatMeshLike[],
  host: HTMLDivElement,
  setFps: (value: number) => void,
  setQualityIndex: (value: number) => void,
  totalSplats: number
) {
  fpsWindow.frames += 1;
  const elapsed = now - fpsWindow.startedAt;
  if (elapsed < 1000) return;
  const currentFps = (fpsWindow.frames * 1000) / elapsed;
  fpsWindow.frames = 0;
  fpsWindow.startedAt = now;
  setFps(currentFps);
  if (!ADAPTIVE_QUALITY) return;

  const overBudget = totalSplats > MAX_RENDER_SPLATS;
  if ((currentFps < QUALITY_DOWN_FPS || overBudget) && qualityRef.current > 0) {
    fpsWindow.lowStreak += 1;
    fpsWindow.highStreak = 0;
    if (fpsWindow.lowStreak >= 1) {
      qualityRef.current -= 1;
      fpsWindow.lowStreak = 0;
      for (const splat of splats) applyQuality(renderer, sparkRenderer, splat, QUALITY_LEVELS[qualityRef.current], host);
      setQualityIndex(qualityRef.current);
    }
  } else if (!overBudget && currentFps > QUALITY_UP_FPS && qualityRef.current < QUALITY_LEVELS.length - 1) {
    fpsWindow.highStreak += 1;
    fpsWindow.lowStreak = 0;
    if (fpsWindow.highStreak >= 3) {
      qualityRef.current += 1;
      fpsWindow.highStreak = 0;
      for (const splat of splats) applyQuality(renderer, sparkRenderer, splat, QUALITY_LEVELS[qualityRef.current], host);
      setQualityIndex(qualityRef.current);
    }
  } else {
    fpsWindow.lowStreak = 0;
    fpsWindow.highStreak = 0;
  }
}

function applyQuality(
  renderer: WebGLRenderer,
  sparkRenderer: SparkRendererLike | null,
  splat: SplatMeshLike,
  quality: QualityLevel,
  host: HTMLDivElement
) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, quality.pixelRatio));
  renderer.setSize(host.clientWidth, host.clientHeight);
  if (sparkRenderer) {
    sparkRenderer.lodSplatScale = quality.lodSplatScale;
    sparkRenderer.maxStdDev = quality.maxStdDev;
    sparkRenderer.maxPixelRadius = quality.maxPixelRadius;
  }
  splat.lodScale = quality.meshLodScale;
}

function readNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function isWeakNetwork(): boolean {
  const connection = (navigator as Navigator & { connection?: { effectiveType?: string; saveData?: boolean } }).connection;
  if (!connection) return false;
  return Boolean(connection.saveData || connection.effectiveType === "slow-2g" || connection.effectiveType === "2g" || connection.effectiveType === "3g");
}
