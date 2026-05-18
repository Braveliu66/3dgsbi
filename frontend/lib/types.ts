export type ProjectStatus =
  | "CREATED"
  | "UPLOADING"
  | "PREPROCESSING"
  | "PREVIEW_RUNNING"
  | "PREVIEW_READY"
  | "FINE_QUEUED"
  | "GLOBAL_OPTIMIZING"
  | "FINE_RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELED";

export interface User {
  id: string;
  username: string;
  email?: string | null;
  role: "user" | "admin";
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: User;
}

export interface Project {
  id: string;
  owner_id: string;
  name: string;
  input_type: "images" | "video";
  status: ProjectStatus;
  tags: string[];
  total_size_bytes: number;
  preview_image_uri?: string | null;
  error_message?: string | null;
  source_version?: number;
  preview_source_version?: number | null;
  created_at: string;
  updated_at: string;
  media?: MediaAsset[];
  tasks?: Task[];
  artifacts?: Artifact[];
}

export interface MediaAsset {
  id: string;
  project_id: string;
  kind: "image" | "video";
  object_uri: string;
  thumbnail_uri?: string | null;
  file_name: string;
  file_size: number;
  width?: number | null;
  height?: number | null;
  duration_seconds?: number | null;
  quality_flags?: Record<string, unknown>;
  source_version?: number;
  client_order?: number;
  created_at: string;
}

export interface Task {
  id: string;
  project_id: string;
  project_name?: string | null;
  type: "preview" | "fine" | "lod" | "mesh_export";
  status: "queued" | "running" | "succeeded" | "failed" | "canceled";
  priority?: number;
  progress: number;
  worker_id?: string | null;
  options?: Record<string, unknown>;
  current_stage: string;
  eta_seconds?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  metrics?: Record<string, unknown>;
  logs?: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface AdminTaskSummary {
  id: string;
  project_id: string;
  type: Task["type"];
  status: Task["status"];
  progress: number;
  worker_id?: string | null;
  current_stage: string;
  eta_seconds?: number | null;
  logs?: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface AdminWorkerSummary {
  worker_id: string;
  hostname: string;
  gpu_index?: number | null;
  gpu_name?: string | null;
  gpu_memory_total?: number | null;
  gpu_memory_used?: number | null;
  gpu_utilization?: number | null;
  cpu_utilization?: number | null;
  last_seen_at: string;
}

export interface AdminProjectUsage {
  id: string;
  name: string;
  owner_id: string;
  owner_username?: string | null;
  status: ProjectStatus;
  input_type: Project["input_type"];
  total_size_bytes: number;
  created_at: string;
  updated_at: string;
  latest_task?: AdminTaskSummary | null;
  worker?: AdminWorkerSummary | null;
}

export interface AdminUserUsage extends User {
  project_count: number;
  total_size_bytes: number;
  feedback_count: number;
}

export interface FeedbackEntry {
  id: string;
  user_id: string;
  username?: string | null;
  project_id?: string | null;
  project_name?: string | null;
  title: string;
  content: string;
  status: string;
  created_at: string;
}

export interface ResourceSnapshotPoint {
  time: number;
  cpu: number;
  memory: number;
  gpu: number;
  vram: number;
}

export interface Artifact {
  id: string;
  project_id: string;
  task_id: string;
  kind: string;
  object_uri: string;
  file_name: string;
  file_size: number;
  format?: "spz" | "rad" | "ply";
  checksum?: string | null;
  metadata?: Record<string, unknown>;
  source_version?: number;
  created_at: string;
}

export interface ViewerLod {
  artifact_id: string;
  model_url: string;
  format: "rad";
  lod: number;
  target_gaussians?: number | null;
  actual_gaussians?: number | null;
  file_size?: number | null;
}

export interface ViewerConfig {
  status: "ready" | "unavailable";
  mode?: "single";
  source?: "final" | "preview";
  artifact_id?: string;
  artifact_kind?: string;
  artifact_file_name?: string;
  file_size?: number | null;
  model_url?: string | null;
  download_spz_url?: string | null;
  download_ply_url?: string | null;
  gaussian_ply_url?: string | null;
  debug_points_ply_url?: string | null;
  debug_splats_ply_url?: string | null;
  viewer_meta_url?: string | null;
  preview_meta_url?: string | null;
  camera_path_url?: string | null;
  quality_warning?: string | null;
  point_source?: string | null;
  scene_type?: "indoor" | "outdoor" | string | null;
  artifact_display?: string | null;
  viewer_default_point_size?: number | null;
  viewer_default_downsample_factor?: number | null;
  viewer_default_conf_threshold?: number | null;
  lods?: ViewerLod[];
  format?: "spz" | "rad" | "ply";
  message?: string;
  stale?: boolean;
}

export interface SharedProject {
  id: string;
  name: string;
  tags: string[];
  total_size_bytes?: number;
  created_at: string;
  updated_at: string;
  viewer: ViewerConfig;
}

export interface ProjectShareResponse {
  share_token: string;
  share_url: string;
  project: SharedProject;
}

export interface AlgorithmEntry {
  name: string;
  repo_url?: string | null;
  license?: string | null;
  commit_hash?: string | null;
  weight_source?: string | null;
  local_path?: string | null;
  enabled: boolean;
  notes?: string | null;
  commands?: Record<string, string[]>;
  weight_paths?: string[];
  source_type?: string;
  bundled?: boolean;
  license_notice?: string | null;
}

export interface RuntimePreflightAlgorithm {
  name: string;
  enabled: boolean;
  ready: boolean;
  repo_url?: string | null;
  license?: string | null;
  commit_hash?: string | null;
  local_path?: string | null;
  weight_paths: string[];
  commands: Record<string, string[]>;
  source_type?: string;
  bundled?: boolean;
  license_notice?: string | null;
  weights_ready?: boolean;
  extensions_ready?: boolean;
  spz_converter_ready?: boolean;
  module?: Record<string, unknown>;
  issues: string[];
}

export interface RuntimePreflight {
  python: Record<string, unknown>;
  gpu: Record<string, unknown>;
  torch: Record<string, unknown>;
  transformer_engine: Record<string, unknown>;
  fine_runtime?: Record<string, unknown>;
  spz_converter?: Record<string, unknown>;
  algorithms: RuntimePreflightAlgorithm[];
  errors: string[];
  warnings: string[];
}

export type PipelineSceneType = "indoor" | "outdoor";
export type PipelineParameterType = "number" | "nullable_number" | "boolean" | "select" | "deblur_switch" | "text";

export interface PipelineParameterField {
  key: string;
  label: string;
  type: PipelineParameterType;
  group: string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  option_labels?: Record<string, string>;
}

export interface PipelineParameterSchemaItem {
  pipeline: string;
  label: string;
  defaults: Record<PipelineSceneType, Record<string, unknown>>;
  fields: PipelineParameterField[];
}

export interface PipelineParameterSchema {
  scene_types: Array<{ value: PipelineSceneType; label: string }>;
  pipelines: PipelineParameterSchemaItem[];
}

export interface PipelineParameterDefaultsResponse {
  defaults: Record<string, Record<PipelineSceneType, Record<string, unknown>>>;
}
