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
  input_type: "images" | "video" | "camera";
  status: ProjectStatus;
  tags: string[];
  total_size_bytes: number;
  preview_image_uri?: string | null;
  error_message?: string | null;
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
  created_at: string;
}

export interface Task {
  id: string;
  project_id: string;
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

export interface Artifact {
  id: string;
  project_id: string;
  task_id: string;
  kind: string;
  object_uri: string;
  file_name: string;
  file_size: number;
  checksum?: string | null;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface ViewerSegment {
  artifact_id: string;
  model_url: string;
  format: "spz" | "rad";
  segment_index: number;
  segment_start_seconds?: number | null;
  segment_end_seconds?: number | null;
  lod?: number | null;
  estimated_splats?: number | null;
  file_size?: number | null;
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
  mode?: "single" | "progressive";
  source?: "final" | "preview";
  artifact_id?: string;
  model_url?: string | null;
  segments?: ViewerSegment[];
  lods?: ViewerLod[];
  progressive?: boolean;
  format?: "spz";
  message?: string;
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
  issues: string[];
}

export interface RuntimePreflight {
  python: Record<string, unknown>;
  gpu: Record<string, unknown>;
  torch: Record<string, unknown>;
  transformer_engine: Record<string, unknown>;
  edgs_cuda_extensions?: Record<string, unknown>;
  lingbot_runtime?: Record<string, unknown>;
  algorithms: RuntimePreflightAlgorithm[];
  errors: string[];
  warnings: string[];
}
