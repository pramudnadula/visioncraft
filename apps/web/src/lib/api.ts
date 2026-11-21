import type {
  CapabilitiesResponse,
  CloudTestResponse,
  CustomModelFile,
  DashboardStats,
  FaceMeshStatus,
  GeneratedReportItem,
  HealthResponse,
  ImageMetadataInfo,
  ImageOperationResponse,
  ImagePreviewResponse,
  JobCreateRequest,
  JobResponse,
  MediaAsset,
  ModelInfo,
  ModelSearchResult,
  SettingsData,
  VisionCapabilities,
} from "../types/api";

const DEFAULT_API_BASE = "/api/v1";

function getApiBase(): string {
  const win = window as unknown as { __FACEVISION_API_BASE__?: string };
  return win.__FACEVISION_API_BASE__ ?? DEFAULT_API_BASE;
}

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem("facevision_api_token");
  return token ? { "X-FaceVision-Token": token } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      ...getAuthHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const err = (await res.json()) as {
        message?: string;
        detail?: string | Array<{ msg?: string; loc?: Array<string | number> }>;
      };
      if (typeof err.message === "string" && err.message) {
        msg = err.message;
      } else if (typeof err.detail === "string" && err.detail) {
        msg = err.detail;
      } else if (Array.isArray(err.detail) && err.detail.length > 0) {
        msg = err.detail
          .map((d) => {
            const loc = Array.isArray(d.loc) ? d.loc.join(".") : "";
            return loc ? `${loc}: ${d.msg ?? "Invalid value"}` : (d.msg ?? "Validation error");
          })
          .join("; ");
      }
    } catch {
      // ignore JSON parse error
    }
    throw new Error(msg);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  capabilities: () => request<CapabilitiesResponse>("/capabilities"),
  stats: () => request<DashboardStats>("/stats"),
  models: () => request<ModelInfo[]>("/models"),
  modelCompatibility: () => request<Record<string, unknown>>("/models/compatibility"),
  benchmarkModel: (model_id: string, iterations = 10) =>
    request<{
      model_id: string;
      model_name: string;
      initialization_ms: number;
      avg_inference_ms: number;
      estimated_fps: number;
      execution_provider: string;
    }>("/models/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id, iterations }),
    }),
  deleteCustomModel: (filename: string) =>
    request<void>(`/models/custom/${encodeURIComponent(filename)}`, { method: "DELETE" }),
  listCustomModels: () => request<CustomModelFile[]>("/models/custom"),
  searchModels: (q: string) =>
    request<ModelSearchResult[]>(`/models/search?q=${encodeURIComponent(q)}`),
  downloadModel: (url: string, filename?: string) =>
    request<CustomModelFile>("/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, filename }),
    }),
  importModel: async (file: File): Promise<CustomModelFile> => {
    const form = new FormData();
    form.append("file", file);
    return request<CustomModelFile>("/models/import", { method: "POST", body: form });
  },

  imageMetadata: (mediaId: string) =>
    request<ImageMetadataInfo>(`/media/image/${mediaId}/metadata`),
  cropImage: (payload: {
    media_id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    rotate_deg?: number;
    flip_h?: boolean;
    flip_v?: boolean;
  }) =>
    request<ImageOperationResponse>("/media/image/crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  resizeImage: (payload: {
    media_id: string;
    width?: number | null;
    height?: number | null;
    scale_percent?: number | null;
    preserve_aspect?: boolean;
  }) =>
    request<ImageOperationResponse>("/media/image/resize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  convertImage: (payload: {
    media_id: string;
    target_format: string;
    quality?: number;
    background_color?: string;
    preserve_metadata?: boolean;
    preserve_animation?: boolean;
  }) =>
    request<ImageOperationResponse>("/media/image/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  compressImage: (payload: {
    media_id: string;
    quality?: number;
    target_kb?: number | null;
    target_format?: string;
    max_width?: number | null;
    max_height?: number | null;
  }) =>
    request<ImageOperationResponse>("/media/image/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  removeBackground: (payload: {
    media_id: string;
    model_type?: "universal" | "portrait";
    background_mode?: "transparent" | "color" | "blur";
    background_color?: string;
  }) =>
    request<ImageOperationResponse>("/media/image/remove-background", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  previewImage: (payload: {
    media_id: string;
    mode: "convert" | "compress" | "remove_bg";
    target_format?: string;
    quality?: number;
    target_kb?: number | null;
    background_color?: string;
    model_type?: "universal" | "portrait";
    background_mode?: "transparent" | "color" | "blur";
  }) =>
    request<ImagePreviewResponse>("/media/image/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  visionCapabilities: () => request<VisionCapabilities>("/vision/capabilities"),
  detectObjects: (payload: { media_id: string; confidence_threshold?: number; vehicles_only?: boolean }) =>
    request<{ count: number; detections: Array<Record<string, unknown>> }>("/vision/objects/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  countVehicles: (payload: { media_id: string; line_y_ratio?: number; max_frames?: number }) =>
    request<Record<string, number>>("/vision/vehicles/count", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  detectFaceMasks: (payload: { media_id: string; confidence_threshold?: number }) =>
    request<Record<string, unknown>>("/vision/faces/mask-detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  faceMeshStatus: () => request<FaceMeshStatus>("/vision/faces/mesh/status"),
  analyzeFacialLandmarks: (payload: {
    media_id: string;
    settings?: { enabled: boolean; max_faces?: number; include_landmarks_in_report?: boolean };
  }) =>
    request<Record<string, unknown>>("/vision/faces/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  detectHands: (payload: { media_id: string }) =>
    request<Record<string, unknown>>("/vision/hands/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  detectPose: (payload: { media_id: string }) =>
    request<Record<string, unknown>>("/vision/pose/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  detectSignLanguage: (payload: { media_id: string; confidence_threshold?: number }) =>
    request<Record<string, unknown>>("/vision/sign-language/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  analyze3DMesh: (payload: { media_id: string }) =>
    request<Record<string, unknown>>("/vision/faces/mesh-3d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listReports: () => request<GeneratedReportItem[]>("/reports"),
  generateReport: (payload: { job_id: string; title: string; template: string; format: "pdf" | "csv" | "json" }) =>
    request<GeneratedReportItem>("/reports/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getSettings: () => request<SettingsData>("/settings"),
  updateSettings: (payload: Partial<SettingsData>) =>
    request<SettingsData>("/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  testCloudConnection: (cloud_api_url: string) =>
    request<CloudTestResponse>("/settings/test-cloud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cloud_api_url }),
    }),
  listMedia: () => request<MediaAsset[]>("/media"),
  uploadMedia: async (file: File): Promise<MediaAsset> => {
    const form = new FormData();
    form.append("file", file);
    return request<MediaAsset>("/media/upload", { method: "POST", body: form });
  },
  deleteMedia: (id: string) => request<void>(`/media/${id}`, { method: "DELETE" }),
  mediaFileUrl: (id: string) => `${getApiBase()}/media/${id}/file`,

  listJobs: (status?: string) =>
    request<JobResponse[]>(status ? `/jobs?status=${encodeURIComponent(status)}` : "/jobs"),
  getJob: (id: string) => request<JobResponse>(`/jobs/${id}`),
  createJob: (payload: JobCreateRequest) =>
    request<JobResponse>("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  cancelJob: (id: string) => request<JobResponse>(`/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => request<JobResponse>(`/jobs/${id}/retry`, { method: "POST" }),
  deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: "DELETE" }),
  artifactUrl: (jobId: string, artifactPath: string) =>
    `${getApiBase()}/jobs/${jobId}/artifacts/${artifactPath}`,

  wsUrl: (path: string) => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const base = getApiBase();
    if (base.startsWith("http")) {
      const url = new URL(base);
      const wsProto = url.protocol === "https:" ? "wss:" : "ws:";
      return `${wsProto}//${url.host}${path}`;
    }
    return `${proto}//${window.location.host}${path}`;
  },
};
