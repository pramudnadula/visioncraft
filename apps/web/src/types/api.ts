export type BlurMode = "none" | "gaussian" | "pixelate" | "mask" | "mesh";
export type ExecutionMode = "local_desktop" | "web_local_agent" | "cloud";

export interface HealthResponse {
  status: "ok";
  app: string;
  version: string;
  execution_mode: ExecutionMode;
}

export interface CapabilitiesResponse {
  execution_mode: ExecutionMode;
  local_processing: boolean;
  cloud_processing_enabled: boolean;
  storage_backend: string;
  max_upload_size_bytes: number;
  max_image_dimension: number;
  max_video_duration_seconds: number;
  supported_image_extensions: string[];
  supported_video_extensions: string[];
}

export interface MediaAsset {
  id: string;
  original_filename: string;
  media_type: "image" | "video";
  mime_type: string;
  extension: string;
  size_bytes: number;
  sha256: string;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
  created_at: string;
  duplicate_of: string | null;
}

export interface DetectionBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number | null;
  tracking_id?: number | null;
}

export interface JobResponse {
  id: string;
  job_type: "image" | "video" | "batch";
  status: "QUEUED" | "VALIDATING" | "PROCESSING" | "FINALIZING" | "COMPLETED" | "FAILED" | "CANCELLED";
  progress: number;
  model_id: string;
  processing_mode: ExecutionMode;
  input_media_ids: string[];
  faces_detected: number;
  error_message: string | null;
  output_manifest: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface JobCreateRequest {
  job_type: "image" | "video" | "batch";
  media_ids: string[];
  model_id: string;
  processing_mode: ExecutionMode;
  cloud_consent: boolean;
  blur_mode: BlurMode;
  confidence_threshold?: number;
  min_face_size?: number;
  enable_tracking?: boolean;
  save_crops?: boolean;
  save_privacy_image?: boolean;
  save_privacy_video?: boolean;
  selected_face_indices?: number[] | null;
  target_fps?: number | null;
  show_hud?: boolean;
  draw_boxes?: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  description: string;
  supports_confidence: boolean;
  supports_landmarks: boolean;
  supports_gpu: boolean;
  gpu_active: boolean;
  is_optional: boolean;
  available: boolean;
  license: string;
  unavailable_reason: string | null;
}

export interface SettingsData {
  default_model_id: string;
  execution_mode: ExecutionMode;
  default_blur_mode: BlurMode;
  default_confidence_threshold: number;
  allow_cloud_processing: boolean;
  cloud_api_url: string | null;
  retention_days: number;
}

export interface CloudTestResponse {
  reachable: boolean;
  status_code: number | null;
  message: string;
}

export interface DashboardStats {
  total_media: number;
  total_jobs: number;
  completed_jobs: number;
  active_jobs: number;
  failed_jobs: number;
  total_faces_detected: number;
  execution_mode: ExecutionMode;
  cloud_processing_enabled: boolean;
}
