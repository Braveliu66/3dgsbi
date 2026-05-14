"use client";

import { Focus, Maximize2, Move, Rotate3D } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Box3, BufferGeometry, Object3D, PerspectiveCamera, Points, Vector3, WebGLRenderer } from "three";
import type { OrbitControls as OrbitControlsType } from "three/examples/jsm/controls/OrbitControls.js";
import { artifactUrl, fetchBytesWithProgress, formatBytes } from "@/lib/api";

type ViewerState = "idle" | "loading" | "ready" | "error";
type ModelFormat = "spz" | "rad" | "ply";
type ViewMode = "splats" | "ply" | "points";
type CameraMode = "orbit" | "fly";
type ViewerMeta = {
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  center: [number, number, number];
  radius: number;
  recommended_view?: {
    position?: [number, number, number];
    target?: [number, number, number];
    up?: [number, number, number];
    fov_y_degrees?: number;
  };
};

interface ViewerControlApi {
  resetCamera: () => void;
  setCameraMode: (mode: CameraMode) => void;
  setSensitivity: (panSpeed: number, zoomSpeed: number) => void;
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

interface SplatMeshOptionsLike {
  url?: string;
  fileBytes?: Uint8Array | ArrayBuffer;
  fileName?: string;
  lod?: boolean | "quality";
  lodAbove?: number;
}

interface QualityLevel {
  label: string;
  pixelRatio: number;
  lodSplatScale: number;
  meshLodScale: number;
  maxStdDev: number;
  maxPixelRadius: number;
}

interface ModelLoadProgress {
  loadedBytes: number;
  totalBytes: number;
  percent: number;
}

interface LoadedModel {
  fileBytes: Uint8Array;
  fileName: string;
}

const QUALITY_LEVELS: QualityLevel[] = [
  { label: "speed", pixelRatio: 0.65, lodSplatScale: 0.28, meshLodScale: 0.55, maxStdDev: Math.sqrt(3), maxPixelRadius: 96 },
  { label: "balanced", pixelRatio: 0.8, lodSplatScale: 0.4, meshLodScale: 0.65, maxStdDev: Math.sqrt(4), maxPixelRadius: 128 },
  { label: "normal", pixelRatio: 1, lodSplatScale: 0.5, meshLodScale: 0.75, maxStdDev: Math.sqrt(5), maxPixelRadius: 160 },
  { label: "sharp", pixelRatio: 1.15, lodSplatScale: 0.6, meshLodScale: 0.85, maxStdDev: Math.sqrt(5), maxPixelRadius: 192 },
  { label: "max", pixelRatio: 1.3, lodSplatScale: 0.7, meshLodScale: 0.95, maxStdDev: Math.sqrt(6), maxPixelRadius: 224 }
];

const TARGET_FPS = readNumber(process.env.VIEWER_TARGET_FPS, 60);
const QUALITY_UP_FPS = readNumber(process.env.VIEWER_QUALITY_UP_FPS, 72);
const QUALITY_DOWN_FPS = readNumber(process.env.VIEWER_QUALITY_DOWN_FPS, 45);
const ADAPTIVE_QUALITY = (process.env.VIEWER_ADAPTIVE_QUALITY ?? "true").toLowerCase() !== "false";
const MAX_RENDER_SPLATS = readNumber(process.env.VIEWER_MAX_SPLATS, 5_000_000);
const DEFAULT_FIT_RADIUS = 1;
const FIT_PADDING = 1.35;

export function SplatViewer({
  modelUrl,
  format = "spz",
  viewerMetaUrl,
  gaussianPlyUrl,
  debugPointsUrl,
  defaultViewMode = "splats"
}: {
  modelUrl?: string | null;
  format?: ModelFormat | null;
  viewerMetaUrl?: string | null;
  gaussianPlyUrl?: string | null;
  debugPointsUrl?: string | null;
  defaultViewMode?: ViewMode;
}) {
  const shellRef = useRef<HTMLElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewerApiRef = useRef<ViewerControlApi | null>(null);
  const applyPointCloudControlsRef = useRef<(() => void) | null>(null);
  const debugControlsRef = useRef({ pointSizeScale: 1, confidenceThreshold: 0, downsampleFactor: 10 });
  const [state, setState] = useState<ViewerState>("idle");
  const [viewerReady, setViewerReady] = useState(false);
  const [message, setMessage] = useState("Waiting for a 3D asset.");
  const [fps, setFps] = useState(0);
  const [qualityIndex, setQualityIndex] = useState(3);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [modelLoadProgress, setModelLoadProgress] = useState<ModelLoadProgress | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>(defaultViewMode);
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [panSensitivity, setPanSensitivity] = useState(0.7);
  const [zoomSensitivity, setZoomSensitivity] = useState(0.9);
  const [debugPointSizeScale, setDebugPointSizeScale] = useState(1);
  const [debugConfidenceThreshold, setDebugConfidenceThreshold] = useState(0);
  const [debugDownsampleFactor, setDebugDownsampleFactor] = useState(10);
  const [pointStats, setPointStats] = useState<{ shown: number; total: number; hasConfidence: boolean } | null>(null);

  debugControlsRef.current = {
    pointSizeScale: debugPointSizeScale,
    confidenceThreshold: debugConfidenceThreshold,
    downsampleFactor: debugDownsampleFactor
  };

  const cameraSettingsRef = useRef({ mode: cameraMode, panSensitivity, zoomSensitivity });
  cameraSettingsRef.current = { mode: cameraMode, panSensitivity, zoomSensitivity };

  useEffect(() => {
    setViewMode(defaultViewMode);
  }, [defaultViewMode, modelUrl, gaussianPlyUrl, debugPointsUrl]);

  useEffect(() => {
    applyPointCloudControlsRef.current?.();
  }, [debugPointSizeScale, debugConfidenceThreshold, debugDownsampleFactor]);

  useEffect(() => {
    const activeModelUrl = viewMode === "ply" ? gaussianPlyUrl : modelUrl;
    const activeFormat: ModelFormat = viewMode === "ply" ? "ply" : format ?? "spz";
    const urls = activeModelUrl ? [activeModelUrl] : [];
    const debugMode = viewMode === "points" && Boolean(debugPointsUrl);
    if ((!urls.length && !debugMode) || !hostRef.current) {
      viewerApiRef.current = null;
      setViewerReady(false);
      setState("idle");
      setMessage("Waiting for a 3D asset.");
      setFps(0);
      setSplatCount(null);
      setModelLoadProgress(null);
      return;
    }

    let cancelled = false;
    const abortController = new AbortController();
    let animationFrame = 0;
    let resizeObserver: ResizeObserver | undefined;
    let cleanup: (() => void) | undefined;
    let controls: OrbitControlsType | undefined;
    let rawPointGeometry: BufferGeometry | null = null;
    const modelFormat = activeFormat;
    const qualityRef = { current: qualityIndex };
    const fpsWindow = { startedAt: performance.now(), frames: 0, highStreak: 0, lowStreak: 0 };
    viewerApiRef.current = null;
    applyPointCloudControlsRef.current = null;
    setViewerReady(false);
    setModelLoadProgress(null);
    setPointStats(null);

    async function mountViewer() {
      try {
        setState("loading");
        setMessage(debugMode ? "Loading raw point cloud" : `Loading ${modelFormat.toUpperCase()} model`);
        const viewerMeta = viewerMetaUrl ? await fetchViewerMeta(viewerMetaUrl, abortController.signal).catch(() => null) : null;
        const loadedModels: LoadedModel[] = [];
        if (!debugMode) {
          for (const url of urls) {
            const loadedModel = await fetchModelBytes(url, modelFormat, abortController.signal, (progress) => {
              if (!cancelled) setModelLoadProgress(progress);
            });
            loadedModels.push(loadedModel);
          }
        }
        if (cancelled) return;
        setMessage(debugMode ? "Initializing point cloud viewer" : "Initializing Spark viewer");
        const THREE = await import("three");
        const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
        const Spark = debugMode
          ? null
          : ((await import("@sparkjsdev/spark") as unknown) as {
              SplatMesh?: new (options: SplatMeshOptionsLike) => SplatMeshLike;
              SparkRenderer?: new (options: Record<string, unknown>) => SparkRendererLike;
            });
        const SplatMesh = Spark?.SplatMesh;
        if (!debugMode && !SplatMesh) throw new Error("@sparkjsdev/spark did not expose SplatMesh");

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
        controls.zoomSpeed = cameraSettingsRef.current.zoomSensitivity;
        controls.panSpeed = cameraSettingsRef.current.panSensitivity;
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
        const sparkRenderer = Spark?.SparkRenderer
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
        const splats = SplatMesh ? loadedModels.map((model, index) => {
          const splat = new SplatMesh({
            fileBytes: model.fileBytes,
            fileName: model.fileName,
            lod: true,
            lodAbove: weakNetwork || index < urls.length - 1 ? 100000 : 250000
          });
          splatGroup.add(splat);
          applyQuality(renderer, sparkRenderer, splat, initialQuality, host);
          return splat;
        }) : [];
        let pointCloud: Points | null = null;
        if (debugMode && debugPointsUrl) {
          const { PLYLoader } = await import("three/examples/jsm/loaders/PLYLoader.js");
          const loader = new PLYLoader() as InstanceType<typeof PLYLoader> & {
            setCustomPropertyNameMapping: (mapping: Record<string, string[]>) => void;
          };
          loader.setCustomPropertyNameMapping({ confidence: ["confidence"] });
          rawPointGeometry = await loader.loadAsync(artifactUrl(debugPointsUrl));
          rawPointGeometry.computeBoundingBox();
          const radius = viewerMeta?.radius ?? 1;
          const basePointSize = Math.max(radius * 0.00016, 0.000012);
          const pointMaterial = new THREE.PointsMaterial({
            size: basePointSize,
            vertexColors: Boolean(rawPointGeometry.getAttribute("color")),
            sizeAttenuation: true
          });
          pointCloud = new THREE.Points(new THREE.BufferGeometry(), pointMaterial);
          applyPointCloudControlsRef.current = () => {
            if (!pointCloud || !rawPointGeometry || cancelled) return;
            const controlsValue = debugControlsRef.current;
            const nextGeometry = buildDebugPointGeometry(
              THREE,
              rawPointGeometry,
              controlsValue.confidenceThreshold,
              controlsValue.downsampleFactor
            );
            const oldGeometry = pointCloud.geometry;
            pointCloud.geometry = nextGeometry.geometry;
            oldGeometry.dispose();
            pointMaterial.size = basePointSize * controlsValue.pointSizeScale;
            setPointStats(nextGeometry.stats);
          };
          applyPointCloudControlsRef.current();
          splatGroup.add(pointCloud);
        }

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
          pointCloud?.geometry.dispose();
          rawPointGeometry?.dispose();
          const pointMaterial = pointCloud?.material;
          if (Array.isArray(pointMaterial)) {
            pointMaterial.forEach((material) => material.dispose());
          } else {
            pointMaterial?.dispose();
          }
          sparkRenderer?.dispose?.();
          renderer.dispose();
          host.innerHTML = "";
          applyPointCloudControlsRef.current = null;
        };

        await Promise.all(splats.map((splat) => splat.initialized ?? Promise.resolve(splat)));
        if (cancelled) return;
        scene.updateMatrixWorld(true);
        const fit = viewerMeta
          ? fitCameraToViewerMeta(THREE, camera, viewerMeta)
          : pointCloud
            ? fitCameraToObject(THREE, camera, pointCloud)
            : fitCameraToSplats(THREE, camera, splats);
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
          const applyCameraMode = (mode: CameraMode) => {
            if (!controls) return;
            controls.enableRotate = mode === "orbit";
            controls.enablePan = true;
            controls.mouseButtons = {
              LEFT: mode === "orbit" ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN,
              MIDDLE: THREE.MOUSE.DOLLY,
              RIGHT: THREE.MOUSE.PAN
            };
            controls.touches = {
              ONE: mode === "orbit" ? THREE.TOUCH.ROTATE : THREE.TOUCH.PAN,
              TWO: THREE.TOUCH.DOLLY_PAN
            };
            controls.update();
          };
          const applySensitivity = (panSpeed: number, zoomSpeed: number) => {
            if (!controls) return;
            controls.panSpeed = panSpeed;
            controls.zoomSpeed = zoomSpeed;
            controls.update();
          };
          applyCameraMode(cameraSettingsRef.current.mode);
          applySensitivity(cameraSettingsRef.current.panSensitivity, cameraSettingsRef.current.zoomSensitivity);
          viewerApiRef.current = {
            resetCamera: () => {
              camera.position.copy(home.position);
              camera.up.copy(home.up);
              camera.zoom = home.zoom;
              controls?.target.copy(home.target);
              camera.lookAt(home.target);
              camera.updateProjectionMatrix();
              controls?.update();
            },
            setCameraMode: applyCameraMode,
            setSensitivity: applySensitivity
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
        setMessage(debugMode ? "Point cloud loaded." : `${modelFormat.toUpperCase()} model loaded.`);
        setModelLoadProgress(null);
      } catch (error) {
        cleanup?.();
        if (cancelled) return;
        setState("error");
        setMessage(error instanceof Error ? error.message : "Viewer failed to load");
        setModelLoadProgress(null);
      }
    }

    void mountViewer();
    return () => {
      cancelled = true;
      abortController.abort();
      viewerApiRef.current = null;
      setViewerReady(false);
      cleanup?.();
    };
  }, [modelUrl, format, viewerMetaUrl, gaussianPlyUrl, debugPointsUrl, viewMode]);

  const quality = QUALITY_LEVELS[qualityIndex] ?? QUALITY_LEVELS[0];
  return (
    <section className="viewer-shell" ref={shellRef}>
      <div ref={hostRef} className="viewer-canvas" />
      <div className="viewer-camera-panel" aria-label="Camera controls">
        <div className="viewer-fps-badge">{Math.round(fps)} FPS</div>
        <div className="viewer-camera-actions">
          <button className="camera-button" type="button" onClick={() => viewerApiRef.current?.resetCamera()} disabled={!viewerReady} title="Reset view" aria-label="Reset view">
            <Focus size={14} />
          </button>
          <button
            className={`camera-button ${cameraMode === "fly" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setCameraMode("fly");
              viewerApiRef.current?.setCameraMode("fly");
            }}
            disabled={!viewerReady}
            title="Fly camera"
            aria-label="Fly camera"
          >
            <Move size={14} />
          </button>
          <button
            className={`camera-button ${cameraMode === "orbit" ? "active" : ""}`}
            type="button"
            onClick={() => {
              setCameraMode("orbit");
              viewerApiRef.current?.setCameraMode("orbit");
            }}
            disabled={!viewerReady}
            title="Orbit camera"
            aria-label="Orbit camera"
          >
            <Rotate3D size={14} />
          </button>
        </div>
        <div className="viewer-sensitivity">
          <label>
            <span>Pan</span>
            <input
              type="range"
              min="0.15"
              max="2"
              step="0.05"
              value={panSensitivity}
              onChange={(event) => {
                const value = Number(event.target.value);
                setPanSensitivity(value);
                viewerApiRef.current?.setSensitivity(value, zoomSensitivity);
              }}
            />
            <strong>{panSensitivity.toFixed(2)}x</strong>
          </label>
          <label>
            <span>Zoom</span>
            <input
              type="range"
              min="0.15"
              max="2"
              step="0.05"
              value={zoomSensitivity}
              onChange={(event) => {
                const value = Number(event.target.value);
                setZoomSensitivity(value);
                viewerApiRef.current?.setSensitivity(panSensitivity, value);
              }}
            />
            <strong>{zoomSensitivity.toFixed(2)}x</strong>
          </label>
        </div>
      </div>
      <div className={`viewer-overlay ${state}`}>
        <span>{message}</span>
        <span className="viewer-stats">
          {state === "ready" ? `${Math.round(fps)} FPS / ${quality.label} / target ${TARGET_FPS}` : quality.label}
          {viewMode === "points" && pointStats ? ` / ${pointStats.shown.toLocaleString()} points` : splatCount ? ` / ${splatCount.toLocaleString()} splats` : ""}
        </span>
        <button className="icon-button" type="button" onClick={() => shellRef.current?.requestFullscreen?.()} aria-label="Fullscreen" title="Fullscreen">
          <Maximize2 size={17} />
        </button>
        {gaussianPlyUrl || debugPointsUrl ? (
          <div className="viewer-mode-buttons" aria-label="Viewer mode">
            <button className={`axis-button ${viewMode === "splats" ? "active" : ""}`} type="button" onClick={() => setViewMode("splats")} disabled={!modelUrl}>
              {format === "rad" ? "RAD" : "SPZ"}
            </button>
            {gaussianPlyUrl ? (
              <button className={`axis-button ${viewMode === "ply" ? "active" : ""}`} type="button" onClick={() => setViewMode("ply")}>
                PLY
              </button>
            ) : null}
            {debugPointsUrl ? (
              <button className={`axis-button ${viewMode === "points" ? "active" : ""}`} type="button" onClick={() => setViewMode("points")}>
                POINTS
              </button>
            ) : null}
          </div>
        ) : null}
        {modelLoadProgress ? <ViewerLoadProgress progress={modelLoadProgress} format={viewMode === "ply" ? "ply" : format ?? "spz"} /> : null}
      </div>
      {debugPointsUrl && viewMode === "points" ? (
        <div className="viewer-point-controls">
          <label>
            <span>Size</span>
            <input type="range" min="0.25" max="4" step="0.05" value={debugPointSizeScale} onChange={(event) => setDebugPointSizeScale(Number(event.target.value))} />
            <strong>{debugPointSizeScale.toFixed(2)}x</strong>
          </label>
          <label>
            <span>Conf</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={debugConfidenceThreshold}
              disabled={pointStats?.hasConfidence === false}
              onChange={(event) => setDebugConfidenceThreshold(Number(event.target.value))}
            />
            <strong>{debugConfidenceThreshold.toFixed(2)}</strong>
          </label>
          <label>
            <span>Sample</span>
            <input type="range" min="1" max="30" step="1" value={debugDownsampleFactor} onChange={(event) => setDebugDownsampleFactor(Number(event.target.value))} />
            <strong>{debugDownsampleFactor}x</strong>
          </label>
        </div>
      ) : null}
    </section>
  );
}

function buildDebugPointGeometry(
  THREE: typeof import("three"),
  source: BufferGeometry,
  confidenceThreshold: number,
  downsampleFactor: number
): { geometry: BufferGeometry; stats: { shown: number; total: number; hasConfidence: boolean } } {
  const position = source.getAttribute("position");
  const color = source.getAttribute("color");
  const confidence = source.getAttribute("confidence");
  const total = position?.count ?? 0;
  if (!position) {
    return { geometry: new THREE.BufferGeometry(), stats: { shown: 0, total: 0, hasConfidence: false } };
  }
  const step = Math.max(1, Math.round(downsampleFactor));
  const hasConfidence = Boolean(confidence && confidence.count === total);
  const maxOutput = Math.ceil(total / step);
  const positions = new Float32Array(maxOutput * 3);
  const colors = color ? new Float32Array(maxOutput * 3) : null;
  const confidences = hasConfidence ? new Float32Array(maxOutput) : null;
  let written = 0;

  for (let index = 0; index < total; index += step) {
    const conf = hasConfidence && confidence ? confidence.getX(index) : 1;
    if (hasConfidence && conf < confidenceThreshold) continue;
    const output = written * 3;
    positions[output] = position.getX(index);
    positions[output + 1] = position.getY(index);
    positions[output + 2] = position.getZ(index);
    if (colors && color) {
      colors[output] = color.getX(index);
      colors[output + 1] = color.getY(index);
      colors[output + 2] = color.getZ(index);
    }
    if (confidences) confidences[written] = conf;
    written += 1;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(0, written * 3), 3));
  if (colors) geometry.setAttribute("color", new THREE.BufferAttribute(colors.slice(0, written * 3), 3));
  if (confidences) geometry.setAttribute("confidence", new THREE.BufferAttribute(confidences.slice(0, written), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return { geometry, stats: { shown: written, total, hasConfidence } };
}

function ViewerLoadProgress({ progress, format }: { progress: ModelLoadProgress; format: ModelFormat }) {
  return (
    <div className="viewer-progress">
      <div className="row between small">
        <span>{format.toUpperCase()} model transfer</span>
        <span>{progress.percent}%</span>
      </div>
      <div className="progress-track" aria-label="SPZ model loading progress">
        <span style={{ width: `${progress.percent}%` }} />
      </div>
      <div className="muted small">
        {formatBytes(progress.loadedBytes)} / {progress.totalBytes ? formatBytes(progress.totalBytes) : "calculating"}
      </div>
    </div>
  );
}

async function fetchModelBytes(url: string, format: ModelFormat, signal: AbortSignal, onProgress: (progress: ModelLoadProgress) => void): Promise<LoadedModel> {
  const fileName = modelFileName(format);
  const result = await fetchBytesWithProgress(
    artifactUrl(url),
    fileName,
    0,
    (progress) => onProgress(modelProgress(progress.loadedBytes, progress.totalBytes)),
    signal
  );
  return { fileBytes: result.bytes, fileName };
}

async function fetchViewerMeta(url: string, signal: AbortSignal): Promise<ViewerMeta> {
  const response = await fetch(artifactUrl(url), { cache: "no-store", signal });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  const meta = await response.json() as Partial<ViewerMeta>;
  if (!isVec3(meta.center) || !isVec3(meta.bbox_min) || !isVec3(meta.bbox_max) || !Number.isFinite(meta.radius)) {
    throw new Error("Invalid viewer meta");
  }
  const recommended = meta.recommended_view;
  return {
    bbox_min: meta.bbox_min,
    bbox_max: meta.bbox_max,
    center: meta.center,
    radius: Math.max(Number(meta.radius), 0.05),
    recommended_view: recommended && isVec3(recommended.position) && isVec3(recommended.target)
      ? {
          position: recommended.position,
          target: recommended.target,
          up: isVec3(recommended.up) ? recommended.up : undefined,
          fov_y_degrees: Number.isFinite(recommended.fov_y_degrees) ? Number(recommended.fov_y_degrees) : undefined
        }
      : undefined
  };
}

function isVec3(value: unknown): value is [number, number, number] {
  return Array.isArray(value) && value.length === 3 && value.every((item) => Number.isFinite(item));
}

function modelProgress(loadedBytes: number, totalBytes: number): ModelLoadProgress {
  const total = Math.max(0, totalBytes);
  const loaded = Math.max(0, Math.min(loadedBytes, total || loadedBytes));
  return {
    loadedBytes: loaded,
    totalBytes: total,
    percent: total > 0 ? Math.max(0, Math.min(100, Math.round((loaded / total) * 100))) : 0
  };
}

function modelFileName(format: ModelFormat): string {
  return `model.${format}`;
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

function fitCameraToViewerMeta(THREE: typeof import("three"), camera: PerspectiveCamera, meta: ViewerMeta): { center: Vector3; radius: number; distance: number } {
  const recommended = meta.recommended_view;
  if (recommended && isVec3(recommended.position) && isVec3(recommended.target)) {
    const target = viewerMetaVector(THREE, recommended.target);
    const position = viewerMetaVector(THREE, recommended.position);
    const distance = Math.max(position.distanceTo(target), 0.35);
    if (Number.isFinite(recommended.fov_y_degrees) && recommended.fov_y_degrees) {
      camera.fov = Math.max(15, Math.min(100, recommended.fov_y_degrees));
    }
    camera.position.copy(position);
    camera.up.copy(isVec3(recommended.up) ? viewerMetaVector(THREE, recommended.up).normalize() : new THREE.Vector3(0, 1, 0));
    camera.near = Math.max(0.001, distance / 1000);
    camera.far = Math.max(100, distance + meta.radius * 100);
    camera.lookAt(target);
    camera.updateProjectionMatrix();
    return { center: target, radius: Math.max(meta.radius, 0.05), distance };
  }
  const center = viewerMetaVector(THREE, meta.center);
  return fitCameraToCenter(THREE, camera, center, Math.max(meta.radius, 0.05));
}

function viewerMetaVector(THREE: typeof import("three"), value: [number, number, number]): Vector3 {
  return new THREE.Vector3(value[0], -value[1], -value[2]);
}

function fitCameraToObject(THREE: typeof import("three"), camera: PerspectiveCamera, object: Object3D): { center: Vector3; radius: number; distance: number } {
  const bounds = new THREE.Box3().setFromObject(object);
  if (bounds.isEmpty()) {
    return fitCameraToCenter(THREE, camera, new THREE.Vector3(0, 0, 0), DEFAULT_FIT_RADIUS);
  }
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(Math.max(size.x, size.y, size.z) / 2, 0.05);
  return fitCameraToCenter(THREE, camera, center, radius);
}

function fitCameraToCenter(THREE: typeof import("three"), camera: PerspectiveCamera, center: Vector3, radius: number): { center: Vector3; radius: number; distance: number } {
  const fov = camera.fov * Math.PI / 180;
  const distance = Math.max(0.35, (radius / Math.tan(fov / 2)) * FIT_PADDING);
  const direction = new THREE.Vector3(0.18, -0.12, 1).normalize();

  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(0.001, radius / 1000);
  camera.far = Math.max(100, distance + radius * 100);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  return { center, radius, distance };
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
