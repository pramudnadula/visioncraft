import type { FaceMeshSettings } from "../types/api";

const KEY = "facevision_face_mesh";

export const DEFAULT_FACE_MESH: FaceMeshSettings = {
  enabled: false,
  max_faces: 3,
  min_confidence: 0.5,
  color: "#00E5FF",
  feature_color: "#FFD400",
  opacity: 0.7,
  thickness: 1,
  show_landmarks: false,
  landmark_size: 1,
  show_tessellation: true,
  show_contours: false,
  show_face_oval: true,
  show_eyes: true,
  show_eyebrows: true,
  show_lips: true,
  show_nose: true,
  show_irises: true,
  include_landmarks_in_report: false,
};

export const MESH_PRESETS: Array<{ id: string; label: string; patch: Partial<FaceMeshSettings> }> = [
  { id: "full", label: "Full mesh", patch: { show_tessellation: true, show_contours: false, show_landmarks: false } },
  { id: "contours", label: "Contours", patch: { show_tessellation: false, show_contours: true, show_landmarks: false } },
  { id: "features", label: "Features only", patch: { show_tessellation: false, show_contours: false, show_landmarks: false } },
  {
    id: "dots",
    label: "Landmark dots",
    patch: {
      show_tessellation: false,
      show_contours: false,
      show_face_oval: false,
      show_eyes: false,
      show_eyebrows: false,
      show_lips: false,
      show_nose: false,
      show_irises: false,
      show_landmarks: true,
    },
  },
];

export function loadFaceMesh(): FaceMeshSettings {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? { ...DEFAULT_FACE_MESH, ...(JSON.parse(raw) as Partial<FaceMeshSettings>) } : DEFAULT_FACE_MESH;
  } catch {
    return DEFAULT_FACE_MESH;
  }
}

export function saveFaceMesh(settings: FaceMeshSettings): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // storage unavailable (private mode / quota) - settings simply won't persist
  }
}
