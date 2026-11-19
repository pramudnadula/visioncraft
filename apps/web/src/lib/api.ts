import type {
  CapabilitiesResponse,
  DashboardStats,
  HealthResponse,
  JobCreateRequest,
  JobResponse,
  MediaAsset,
  ModelInfo,
  SettingsData,
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
      const err = (await res.json()) as { message?: string; detail?: string };
      msg = err.message || err.detail || msg;
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
  getSettings: () => request<SettingsData>("/settings"),
  updateSettings: (payload: Partial<SettingsData>) =>
    request<SettingsData>("/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
