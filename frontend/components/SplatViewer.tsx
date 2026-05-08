"use client";

import { Focus, Maximize2, RotateCcw, RotateCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Box3, Object3D, PerspectiveCamera, Vector3, WebGLRenderer } from "three";
import type { OrbitControls as OrbitControlsType } from "three/examples/jsm/controls/OrbitControls.js";
import { artifactUrl } from "@/lib/api";
import type { ViewerSegment } from "@/lib/types";

type ViewerState = "idle" | "loading" | "ready" | "error";
type AxisView = "x-positive" | "x-negative" | "y-positive" | "y-negative" | "z-positive" | "z-negative";

interface ViewerControlApi {
  resetCamera: () => void;
  rotateModel: (radians: number) => void;
  setAxisView: (view: AxisView) => void;
}

interface SparkRendererLike extends Object3D {
  lodSplatScale?: number;
  maxStdDev?: number;
  maxPixelRadius?: number;
  dispose?: () => void;
}

interface SplatMeshLike extends Object3D {
  initialized?: Promise<SplatMeshLike>;
  lodScale?: number;
  numSplats?: number;
  splats?: { getNumSplats?: () => number };
  dispose?: () => void;
  getBoundingBox?: (centersOnly?: boolean) => Box3;
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
const DEFAULT_FIT_RADIUS = 1;
const FIT_PADDING = 1.35;

export function SplatViewer({ modelUrl, segments }: { modelUrl?: string | null; segments?: ViewerSegment[] }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerApiRef = useRef<ViewerControlApi | null>(null);
  const [state, setState] = useState<ViewerState>("idle");
  const [viewerReady, setViewerReady] = useState(false);
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
      viewerApiRef.current = null;
      setViewerReady(false);
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
    let controls: OrbitControlsType | undefined;
    const qualityRef = { current: qualityIndex };
    const fpsWindow = { startedAt: performance.now(), frames: 0, highStreak: 0, lowStreak: 0 };
    viewerApiRef.current = null;
    setViewerReady(false);

    async function mountViewer() {
      try {
        setState("loading");
        setMessage("正在初始化 Spark Viewer");
        const THREE = await import("three");
        const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
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
        scene.background = new THREE.Color(0x0f1512);
        const camera = new THREE.PerspectiveCamera(55, host.clientWidth / Math.max(host.clientHeight, 1), 0.01, 1000);
        camera.position.set(0, 0, 3);
        const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.setClearColor(0x0f1512, 1);
        renderer.setSize(host.clientWidth, host.clientHeight);
        host.appendChild(renderer.domElement);
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = false;
        controls.screenSpacePanning = true;
        controls.rotateSpeed = 1;
        controls.zoomSpeed = 0.9;
        controls.panSpeed = 0.7;
        controls.autoRotate = false;
        controls.enableRotate = true;
        controls.enablePan = true;
        controls.enableZoom = true;
        controls.minPolarAngle = 0;
        controls.maxPolarAngle = Math.PI;
        controls.mouseButtons = {
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN
        };
        controls.touches = {
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN
        };

        const initialQuality = QUALITY_LEVELS[qualityRef.current];
        const sparkRenderer = Spark.SparkRenderer
          ? new Spark.SparkRenderer({
              renderer,
              lodSplatScale: initialQuality.lodSplatScale,
              maxStdDev: initialQuality.maxStdDev,
              maxPixelRadius: initialQuality.maxPixelRadius
            })
          : null;
        if (sparkRenderer) scene.add(sparkRenderer);

        const modelPivot = new THREE.Group();
        scene.add(modelPivot);
        const splatGroup = new THREE.Group();
        splatGroup.rotation.x = Math.PI;
        modelPivot.add(splatGroup);

        const weakNetwork = isWeakNetwork();
        const splats = urls.map((url, index) => {
          const splat = new SplatMesh({
            url: artifactUrl(url),
            lod: true,
            lodAbove: weakNetwork || index < urls.length - 1 ? 50000 : 100000
          });
          splatGroup.add(splat);
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

        cleanup = () => {
          cancelAnimationFrame(animationFrame);
          resizeObserver?.disconnect();
          controls?.dispose();
          for (const splat of splats) splat.dispose?.();
          sparkRenderer?.dispose?.();
          renderer.dispose();
          host.innerHTML = "";
        };

        await Promise.all(splats.map((splat) => splat.initialized ?? Promise.resolve(splat)));
        if (cancelled) return;
        scene.updateMatrixWorld(true);
        const fit = fitCameraToSplats(THREE, camera, splats);
        if (controls) {
          modelPivot.position.copy(fit.center);
          splatGroup.position.copy(fit.center.clone().multiplyScalar(-1));
          scene.updateMatrixWorld(true);
          controls.target.copy(fit.center);
          controls.minDistance = Math.max(0.01, fit.radius * 0.03);
          controls.maxDistance = Math.max(10, fit.distance * 10);
          controls.update();
          controls.saveState();
          const home = {
            position: camera.position.clone(),
            target: controls.target.clone(),
            up: camera.up.clone(),
            zoom: camera.zoom
          };
          viewerApiRef.current = {
            resetCamera: () => {
              modelPivot.rotation.set(0, 0, 0);
              camera.position.copy(home.position);
              camera.up.copy(home.up);
              camera.zoom = home.zoom;
              controls?.target.copy(home.target);
              camera.lookAt(home.target);
              camera.updateProjectionMatrix();
              controls?.update();
            },
            rotateModel: (radians: number) => {
              modelPivot.rotation.y += radians;
              modelPivot.updateMatrixWorld(true);
              controls?.update();
            },
            setAxisView: (view: AxisView) => {
              const axis = axisViewVector(THREE, view);
              const up = axisViewUp(THREE, view);
              camera.position.copy(fit.center).addScaledVector(axis, fit.distance);
              camera.up.copy(up);
              camera.zoom = 1;
              controls?.target.copy(fit.center);
              camera.lookAt(fit.center);
              camera.updateProjectionMatrix();
              controls?.update();
            }
          };
          setViewerReady(true);
        }

        const render = (now: number) => {
          animationFrame = requestAnimationFrame(render);
          controls?.update();
          renderer.render(scene, camera);
          const totalSplats = splats.reduce((sum, splat) => sum + readSplatCount(splat), 0);
          updateFps(now, fpsWindow, qualityRef, renderer, sparkRenderer, splats, host, setFps, setQualityIndex, totalSplats);
          if (totalSplats > 0) setSplatCount(totalSplats);
        };
        animationFrame = requestAnimationFrame(render);

        setState("ready");
        setMessage(hasSegments ? `Spark Viewer 已加载 ${urls.length} 个增量 SPZ 片段。` : "Spark Viewer 已加载真实 SPZ 产物。");
      } catch (error) {
        cleanup?.();
        if (cancelled) return;
        setState("error");
        setMessage(error instanceof Error ? error.message : "Spark Viewer 加载失败");
      }
    }

    void mountViewer();
    return () => {
      cancelled = true;
      viewerApiRef.current = null;
      setViewerReady(false);
      cleanup?.();
    };
  }, [modelUrl, segmentKey, activeSegment, hasSegments]);

  const quality = QUALITY_LEVELS[qualityIndex] ?? QUALITY_LEVELS[0];
  return (
    <section className="viewer-shell">
      <div ref={hostRef} className="viewer-canvas" />
      <div className="viewer-axis-panel" aria-label="视角控制">
        <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.resetCamera()} disabled={!viewerReady} title="回到初始视角" aria-label="回到初始视角">
          <Focus size={16} />
        </button>
        <div className="axis-grid">
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("x-negative")} disabled={!viewerReady} title="X- 左侧视角" aria-label="X- 左侧视角">X-</button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("y-positive")} disabled={!viewerReady} title="Y+ 顶部视角" aria-label="Y+ 顶部视角">Y+</button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("x-positive")} disabled={!viewerReady} title="X+ 右侧视角" aria-label="X+ 右侧视角">X+</button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("z-negative")} disabled={!viewerReady} title="Z- 背面视角" aria-label="Z- 背面视角">Z-</button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("y-negative")} disabled={!viewerReady} title="Y- 底部视角" aria-label="Y- 底部视角">Y-</button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.setAxisView("z-positive")} disabled={!viewerReady} title="Z+ 正面视角" aria-label="Z+ 正面视角">Z+</button>
        </div>
        <div className="axis-rotate">
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.rotateModel(Math.PI / 2)} disabled={!viewerReady} title="物体左转 90 度" aria-label="物体左转 90 度">
            <RotateCcw size={15} />
          </button>
          <button className="axis-button" type="button" onClick={() => viewerApiRef.current?.rotateModel(-Math.PI / 2)} disabled={!viewerReady} title="物体右转 90 度" aria-label="物体右转 90 度">
            <RotateCw size={15} />
          </button>
        </div>
      </div>
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

function fitCameraToSplats(THREE: typeof import("three"), camera: PerspectiveCamera, splats: SplatMeshLike[]): { center: Vector3; radius: number; distance: number } {
  const bounds = new THREE.Box3();
  let hasBounds = false;
  for (const splat of splats) {
    const box = splat.getBoundingBox?.(true);
    if (!box || box.isEmpty()) continue;
    splat.updateWorldMatrix(true, false);
    const worldBox = box.clone().applyMatrix4(splat.matrixWorld);
    if (worldBox.isEmpty()) continue;
    if (hasBounds) {
      bounds.union(worldBox);
    } else {
      bounds.copy(worldBox);
      hasBounds = true;
    }
  }

  const center = hasBounds ? bounds.getCenter(new THREE.Vector3()) : new THREE.Vector3(0, 0, 0);
  const size = hasBounds ? bounds.getSize(new THREE.Vector3()) : new THREE.Vector3(DEFAULT_FIT_RADIUS * 2, DEFAULT_FIT_RADIUS * 2, DEFAULT_FIT_RADIUS * 2);
  const radius = Math.max(Math.max(size.x, size.y, size.z) / 2, 0.05);
  const fov = camera.fov * Math.PI / 180;
  const distance = Math.max(0.35, (radius / Math.tan(fov / 2)) * FIT_PADDING);
  const direction = new THREE.Vector3(0.18, -0.12, 1).normalize();

  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(0.001, distance / 1000);
  camera.far = Math.max(1000, distance + radius * 8);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  return { center, radius, distance };
}

function axisViewVector(THREE: typeof import("three"), view: AxisView): Vector3 {
  switch (view) {
    case "x-positive":
      return new THREE.Vector3(1, 0, 0);
    case "x-negative":
      return new THREE.Vector3(-1, 0, 0);
    case "y-positive":
      return new THREE.Vector3(0, 1, 0);
    case "y-negative":
      return new THREE.Vector3(0, -1, 0);
    case "z-negative":
      return new THREE.Vector3(0, 0, -1);
    case "z-positive":
    default:
      return new THREE.Vector3(0, 0, 1);
  }
}

function axisViewUp(THREE: typeof import("three"), view: AxisView): Vector3 {
  if (view === "y-positive") return new THREE.Vector3(0, 0, -1);
  if (view === "y-negative") return new THREE.Vector3(0, 0, 1);
  return new THREE.Vector3(0, 1, 0);
}

function readSplatCount(splat: SplatMeshLike): number {
  if (typeof splat.numSplats === "number") return splat.numSplats;
  const count = splat.splats?.getNumSplats?.();
  return typeof count === "number" && Number.isFinite(count) ? count : 0;
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
