"use client";

import { Focus } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { BufferGeometry, PerspectiveCamera, Points, Vector3, WebGLRenderer } from "three";
import type { OrbitControls as OrbitControlsType } from "three/examples/jsm/controls/OrbitControls.js";
import { artifactUrl, fetchBytesWithProgress, formatBytes } from "@/lib/api";
import type { Artifact } from "@/lib/types";

type ViewerState = "idle" | "loading" | "ready" | "error";

type PointCloudPreviewViewerProps = {
  artifact: Artifact;
  pointSize?: number;
  downsampleFactor?: number;
  confidenceThreshold?: number;
};

type PointCloudData = {
  positions: Float32Array;
  colors: Float32Array;
  total: number;
  shown: number;
  hasConfidence: boolean;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
};

type PlyProperty = {
  name: string;
  type: string;
  offset: number;
  size: number;
};

const DEFAULT_POINT_SIZE = 0.00001;

export function PointCloudPreviewViewer({
  artifact,
  pointSize = DEFAULT_POINT_SIZE,
  downsampleFactor = 10,
  confidenceThreshold = 1.5
}: PointCloudPreviewViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const resetRef = useRef<(() => void) | null>(null);
  const [state, setState] = useState<ViewerState>("idle");
  const [message, setMessage] = useState("Waiting for point cloud.");
  const [progress, setProgress] = useState<{ loadedBytes: number; totalBytes: number; percent: number } | null>(null);
  const [stats, setStats] = useState<{ shown: number; total: number; hasConfidence: boolean } | null>(null);

  useEffect(() => {
    if (!hostRef.current || !artifact.object_uri) {
      resetRef.current = null;
      setState("idle");
      setMessage("Waiting for point cloud.");
      setStats(null);
      return;
    }

    let cancelled = false;
    let animationFrame = 0;
    let resizeObserver: ResizeObserver | undefined;
    let cleanup: (() => void) | undefined;
    let controls: OrbitControlsType | undefined;
    let renderer: WebGLRenderer | undefined;
    let camera: PerspectiveCamera | undefined;
    let cloud: Points | undefined;
    let geometry: BufferGeometry | undefined;
    const abortController = new AbortController();

    async function mountViewer() {
      try {
        setState("loading");
        setMessage("Loading point cloud");
        setProgress(null);
        setStats(null);

        const loaded = await fetchBytesWithProgress(
          artifactUrl(artifact.object_uri),
          artifact.file_name || "preview.ply",
          artifact.file_size || 0,
          (next) => {
            if (!cancelled) {
              setProgress({
                loadedBytes: next.loadedBytes,
                totalBytes: next.totalBytes,
                percent: next.percent
              });
            }
          },
          abortController.signal
        );
        if (cancelled) return;

        const parsed = parsePointCloudPly(
          loaded.bytes,
          Math.max(1, Math.round(downsampleFactor)),
          Number.isFinite(confidenceThreshold) ? confidenceThreshold : 0
        );
        const THREE = await import("three");
        const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
        const host = hostRef.current;
        if (!host || cancelled) return;

        host.innerHTML = "";
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f1512);
        camera = new THREE.PerspectiveCamera(55, host.clientWidth / Math.max(host.clientHeight, 1), 0.001, 1000);
        renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.setClearColor(0x0f1512, 1);
        renderer.setSize(host.clientWidth, host.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
        host.appendChild(renderer.domElement);

        geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(parsed.positions, 3));
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(parsed.colors, 3));
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true
        });
        cloud = new THREE.Points(geometry, material);
        scene.add(cloud);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = false;
        controls.screenSpacePanning = true;
        controls.enableRotate = true;
        controls.enablePan = true;
        controls.enableZoom = true;

        const home = fitCameraToBounds(THREE, camera, controls, parsed.bounds);
        resetRef.current = () => {
          if (!camera || !controls) return;
          camera.position.copy(home.position);
          camera.up.copy(home.up);
          controls.target.copy(home.target);
          camera.lookAt(home.target);
          camera.updateProjectionMatrix();
          controls.update();
        };

        resizeObserver = new ResizeObserver(() => {
          if (!host.clientWidth || !host.clientHeight || !camera || !renderer) return;
          camera.aspect = host.clientWidth / Math.max(host.clientHeight, 1);
          camera.updateProjectionMatrix();
          renderer.setSize(host.clientWidth, host.clientHeight);
        });
        resizeObserver.observe(host);

        const render = () => {
          animationFrame = requestAnimationFrame(render);
          controls?.update();
          if (renderer && camera) renderer.render(scene, camera);
        };
        animationFrame = requestAnimationFrame(render);

        cleanup = () => {
          cancelAnimationFrame(animationFrame);
          resizeObserver?.disconnect();
          controls?.dispose();
          cloud?.geometry.dispose();
          const material = cloud?.material;
          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose());
          } else {
            material?.dispose();
          }
          renderer?.dispose();
          host.innerHTML = "";
          resetRef.current = null;
        };

        setStats({ shown: parsed.shown, total: parsed.total, hasConfidence: parsed.hasConfidence });
        setProgress(null);
        setState("ready");
        setMessage("Point cloud loaded.");
      } catch (error) {
        cleanup?.();
        if (cancelled) return;
        setState("error");
        setMessage(error instanceof Error ? error.message : "Point cloud viewer failed to load");
        setProgress(null);
      }
    }

    void mountViewer();

    return () => {
      cancelled = true;
      abortController.abort();
      cleanup?.();
    };
  }, [artifact.id, artifact.object_uri, artifact.file_name, artifact.file_size, pointSize, downsampleFactor, confidenceThreshold]);

  return (
    <section className="viewer-shell">
      <div ref={hostRef} className="viewer-canvas" />
      <div className="viewer-camera-panel" aria-label="Point cloud controls">
        <div className="viewer-fps-badge">{state === "ready" ? "PLY" : state.toUpperCase()}</div>
        <div className="viewer-camera-actions">
          <button className="camera-button" type="button" onClick={() => resetRef.current?.()} disabled={state !== "ready"} title="Reset view" aria-label="Reset view">
            <Focus size={14} />
          </button>
        </div>
      </div>
      {state !== "ready" ? (
        <div className="viewer-overlay">
          <div>
            <strong>{message}</strong>
            {progress ? (
              <p className="muted small">
                {progress.percent}% · {formatBytes(progress.loadedBytes)} / {progress.totalBytes ? formatBytes(progress.totalBytes) : "calculating"}
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
      {stats ? (
        <div className="viewer-point-controls">
          <label>
            <span>Points</span>
            <strong>{stats.shown.toLocaleString()} / {stats.total.toLocaleString()}</strong>
          </label>
          <label>
            <span>Confidence</span>
            <strong>{stats.hasConfidence ? confidenceThreshold.toString() : "none"}</strong>
          </label>
        </div>
      ) : null}
    </section>
  );
}

function parsePointCloudPly(bytes: Uint8Array, downsampleFactor: number, confidenceThreshold: number): PointCloudData {
  const headerEnd = findHeaderEnd(bytes);
  if (headerEnd <= 0) throw new Error("PLY header terminator missing");
  const header = new TextDecoder("ascii").decode(bytes.slice(0, headerEnd));
  const lines = header.split(/\r?\n/).filter(Boolean);
  if (lines[0] !== "ply") throw new Error("Invalid PLY file");
  if (lines[1] !== "format binary_little_endian 1.0") throw new Error("Only binary little-endian PLY is supported");

  const vertexCount = readVertexCount(lines);
  const properties = readVertexProperties(lines);
  const rowSize = properties.reduce((sum, property) => sum + property.size, 0);
  const bodyOffset = headerEnd;
  if (bytes.byteLength < bodyOffset + vertexCount * rowSize) throw new Error("PLY body is truncated");

  const required = ["x", "y", "z", "red", "green", "blue"];
  for (const name of required) {
    if (!properties.some((property) => property.name === name)) throw new Error(`PLY property missing: ${name}`);
  }
  const hasConfidence = properties.some((property) => property.name === "confidence");
  const step = Math.max(1, downsampleFactor);
  const maxOutput = Math.ceil(vertexCount / step);
  const positions = new Float32Array(maxOutput * 3);
  const colors = new Float32Array(maxOutput * 3);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let written = 0;

  for (let index = 0; index < vertexCount; index += step) {
    const offset = bodyOffset + index * rowSize;
    const confidence = hasConfidence ? readProperty(view, offset, properties, "confidence") : 1;
    if (confidence < confidenceThreshold) continue;
    const x = readProperty(view, offset, properties, "x");
    const y = readProperty(view, offset, properties, "y");
    const z = readProperty(view, offset, properties, "z");
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const base = written * 3;
    positions[base] = x;
    positions[base + 1] = -y;
    positions[base + 2] = -z;
    colors[base] = readProperty(view, offset, properties, "red") / 255;
    colors[base + 1] = readProperty(view, offset, properties, "green") / 255;
    colors[base + 2] = readProperty(view, offset, properties, "blue") / 255;

    min[0] = Math.min(min[0], positions[base]);
    min[1] = Math.min(min[1], positions[base + 1]);
    min[2] = Math.min(min[2], positions[base + 2]);
    max[0] = Math.max(max[0], positions[base]);
    max[1] = Math.max(max[1], positions[base + 1]);
    max[2] = Math.max(max[2], positions[base + 2]);
    written += 1;
  }

  if (written <= 0) throw new Error("No points passed the preview filters");
  return {
    positions: positions.slice(0, written * 3),
    colors: colors.slice(0, written * 3),
    total: vertexCount,
    shown: written,
    hasConfidence,
    bounds: { min, max }
  };
}

function findHeaderEnd(bytes: Uint8Array): number {
  const marker = new TextEncoder().encode("end_header\n");
  for (let index = 0; index <= bytes.length - marker.length; index += 1) {
    let matched = true;
    for (let offset = 0; offset < marker.length; offset += 1) {
      if (bytes[index + offset] !== marker[offset]) {
        matched = false;
        break;
      }
    }
    if (matched) return index + marker.length;
  }
  return -1;
}

function readVertexCount(lines: string[]): number {
  const line = lines.find((item) => item.startsWith("element vertex "));
  const count = Number(line?.split(/\s+/).at(-1));
  if (!Number.isFinite(count) || count <= 0) throw new Error("PLY vertex count is invalid");
  return Math.floor(count);
}

function readVertexProperties(lines: string[]): PlyProperty[] {
  const properties: PlyProperty[] = [];
  let inVertex = false;
  let offset = 0;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      inVertex = true;
      continue;
    }
    if (inVertex && line.startsWith("element ")) break;
    if (!inVertex || !line.startsWith("property ")) continue;
    const [, type, name] = line.split(/\s+/);
    const size = propertySize(type);
    properties.push({ name, type, offset, size });
    offset += size;
  }
  return properties;
}

function propertySize(type: string): number {
  if (type === "float" || type === "float32") return 4;
  if (type === "uchar" || type === "uint8" || type === "char" || type === "int8") return 1;
  throw new Error(`Unsupported PLY property type: ${type}`);
}

function readProperty(view: DataView, rowOffset: number, properties: PlyProperty[], name: string): number {
  const property = properties.find((item) => item.name === name);
  if (!property) return 0;
  const offset = rowOffset + property.offset;
  if (property.type === "float" || property.type === "float32") return view.getFloat32(offset, true);
  if (property.type === "char" || property.type === "int8") return view.getInt8(offset);
  return view.getUint8(offset);
}

function fitCameraToBounds(
  THREE: typeof import("three"),
  camera: PerspectiveCamera,
  controls: OrbitControlsType,
  bounds: PointCloudData["bounds"]
): { position: Vector3; target: Vector3; up: Vector3 } {
  const min = new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]);
  const max = new THREE.Vector3(bounds.max[0], bounds.max[1], bounds.max[2]);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  const radius = Math.max(size.length() * 0.5, 0.05);
  const distance = Math.max(0.35, radius / Math.tan((camera.fov * Math.PI / 180) / 2) * 1.35);
  const direction = new THREE.Vector3(0.18, -0.12, 1).normalize();
  const position = center.clone().addScaledVector(direction, distance);
  const up = new THREE.Vector3(0, 1, 0);

  camera.position.copy(position);
  camera.up.copy(up);
  camera.near = Math.max(0.001, radius / 1000);
  camera.far = Math.max(100, distance + radius * 100);
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.minDistance = Math.max(0.001, radius * 0.02);
  controls.maxDistance = Math.max(10, distance * 10);
  controls.update();
  return { position: position.clone(), target: center.clone(), up };
}
