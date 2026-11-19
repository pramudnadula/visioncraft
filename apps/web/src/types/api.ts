export type BlurMode = "none" | "gaussian" | "pixelate" | "mask" | "mesh" | "geometric" | "mesh_3d";

export interface ImagePreviewResponse {
  preview_data_url: string;
  output_format: string;
  width: number;
  height: number;
  original_size_bytes: number;
  estimated_size_bytes: number;
  reduction_percent: number;
  target_achieved: boolean;
}
export type EmotionMode = "off" | "hybrid" | "ferplus" | "blendshapes";

export interface FaceEmotionFace {
  face_index: number;
  tracking_id: number | null;
  dominant_emotion: string;
  confidence: number;
  source: string;
  scores: Array<{ emotion: string; score: number }>;
}

export interface FaceEmotionResult {
  mode: EmotionMode;
  face_count: number;
  faces: FaceEmotionFace[];
  disclaimer?: string;
}

export interface FaceMeshSettings {
  enabled: boolean;
  max_faces: number;
  min_confidence: number;
  color: string;
  feature_color: string;
  opacity: number;
  thickness: number;
  show_landmarks: boolean;
  landmark_size: number;
  show_tessellation: boolean;
  show_contours: boolean;
  show_face_oval: boolean;
  show_eyes: boolean;
  show_eyebrows: boolean;
  show_lips: boolean;
  show_nose: boolean;
  show_irises: boolean;
  include_landmarks_in_report: boolean;
}

export interface FaceMeshFaceSummary {
  face_index: number;
  tracking_id: number | null;
  landmark_count: number;
  head_pose: { pitch_deg: number; yaw_deg: number; roll_deg: number } | null;
  expression_indicators: Array<{ name: string; score: number }>;
}

export interface FaceMeshResult {
  face_count: number;
  faces: FaceMeshFaceSummary[];
  disclaimer?: string;
  frames_analyzed?: number;
  frames_with_faces?: number;
  unique_face_tracks?: number;
  max_faces_in_frame?: number;
}

export interface FaceMeshStatus {
  available: boolean;
  model: string;
  license: string;
  landmarks: number;
  offline: boolean;
  checksum_verified: boolean | null;
  unavailable_reason: string | null;
  disclaimer: string;
}
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
  privacy_modes?: string[];
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
  face_mesh?: FaceMeshSettings | null;
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

export interface ModelSearchResult {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  url: string;
}

export interface CustomModelFile {
  filename: string;
  size_bytes: number;
}

export interface ImageOperationResponse {
  operation_id: string;
  operation_type: string;
  output_filename: string;
  output_format: string;
  width: number;
  height: number;
  original_size_bytes: number;
  output_size_bytes: number;
  reduction_percent: number;
  target_achieved: boolean;
  animation_preserved: boolean;
  note: string | null;
  download_url: string;
}

export interface ImageMetadataInfo {
  format: string;
  mode: string;
  width: number;
  height: number;
  size_bytes: number;
  is_animated: boolean;
  frame_count: number;
  has_alpha: boolean;
  exif_keys: string[];
}

export interface GeneratedReportItem {
  id: string;
  job_id: string;
  title: string;
  report_type: string;
  format: "pdf" | "csv" | "json";
  created_at: string;
  download_url: string;
}

export interface VisionCapabilities {
  face_detection: boolean;
  face_mesh_3d_ready: boolean;
  face_mask_ready: boolean;
  facial_analysis_ready: boolean;
  hand_landmarks_ready: boolean;
  pose_estimation_ready: boolean;
  object_detection_ready: boolean;
  vehicle_analytics_ready: boolean;
  sign_languages: Array<{
    code: string;
    name: string;
    model_available: boolean;
    note: string;
  }>;
  object_model_hint: string;
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
