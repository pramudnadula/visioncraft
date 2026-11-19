import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Layers } from "lucide-react";
import { api } from "../lib/api";
import { DEFAULT_FACE_MESH, loadFaceMesh, MESH_PRESETS, saveFaceMesh } from "../lib/faceMesh";
import type { FaceMeshSettings } from "../types/api";

interface Props {
  value: FaceMeshSettings;
  onChange: (next: FaceMeshSettings) => void;
  /** Show developer options (landmark size, report landmarks, min confidence). */
  advanced?: boolean;
  /** Hide options that only apply to saved jobs (e.g. include landmarks in report). */
  live?: boolean;
}

/** Persisted face mesh settings shared by Image, Video and Webcam studios. */
export function useFaceMeshSettings(): [FaceMeshSettings, (next: FaceMeshSettings) => void] {
  const [settings, setSettings] = useState<FaceMeshSettings>(() => loadFaceMesh());
  useEffect(() => saveFaceMesh(settings), [settings]);
  return [settings, setSettings];
}

const REGIONS: Array<{ key: keyof FaceMeshSettings; label: string }> = [
  { key: "show_tessellation", label: "Mesh" },
  { key: "show_contours", label: "Contours" },
  { key: "show_face_oval", label: "Face outline" },
  { key: "show_eyes", label: "Eyes" },
  { key: "show_eyebrows", label: "Eyebrows" },
  { key: "show_lips", label: "Lips" },
  { key: "show_nose", label: "Nose" },
  { key: "show_irises", label: "Irises" },
];

export function FaceMeshControls({ value, onChange, advanced = false, live = false }: Props) {
  const { data: status } = useQuery({ queryKey: ["face-mesh-status"], queryFn: api.faceMeshStatus });
  const set = <K extends keyof FaceMeshSettings>(key: K, v: FaceMeshSettings[K]) =>
    onChange({ ...value, [key]: v });
  const unavailable = status && !status.available;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 text-xs">
      <label className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-2 font-semibold text-sm">
          <Layers className="h-4 w-4 text-cyan-500" /> Face mesh overlay
        </span>
        <input
          type="checkbox"
          aria-label="Enable face mesh"
          checked={value.enabled}
          disabled={!!unavailable}
          onChange={(e) => set("enabled", e.target.checked)}
        />
      </label>

      {unavailable && (
        <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
          {status.unavailable_reason}
        </div>
      )}

      {value.enabled && (
        <>
          <p className="text-slate-600 dark:text-slate-400">
            478 3D landmarks per face, processed locally{status?.offline ? " (works offline)" : ""}.
          </p>

          <div className="flex flex-wrap gap-1.5">
            {MESH_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => onChange({ ...value, ...p.patch })}
                className="px-2 py-1 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="block text-slate-600 dark:text-slate-400">Mesh color</span>
              <input
                type="color"
                value={value.color}
                onChange={(e) => set("color", e.target.value)}
                className="h-8 w-full rounded border border-slate-300 dark:border-slate-700"
              />
            </label>
            <label className="space-y-1">
              <span className="block text-slate-600 dark:text-slate-400">Feature color</span>
              <input
                type="color"
                value={value.feature_color}
                onChange={(e) => set("feature_color", e.target.value)}
                className="h-8 w-full rounded border border-slate-300 dark:border-slate-700"
              />
            </label>
          </div>

          <label className="block space-y-1">
            <span className="text-slate-600 dark:text-slate-400">Opacity: {Math.round(value.opacity * 100)}%</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={value.opacity}
              onChange={(e) => set("opacity", Number(e.target.value))}
              className="w-full"
            />
          </label>

          <label className="block space-y-1">
            <span className="text-slate-600 dark:text-slate-400">Line thickness: {value.thickness}px</span>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={value.thickness}
              onChange={(e) => set("thickness", Number(e.target.value))}
              className="w-full"
            />
          </label>

          <label className="block space-y-1">
            <span className="text-slate-600 dark:text-slate-400">Max faces: {value.max_faces}</span>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={value.max_faces}
              onChange={(e) => set("max_faces", Number(e.target.value))}
              className="w-full"
            />
          </label>

          <div className="grid grid-cols-2 gap-1.5">
            {REGIONS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={Boolean(value[key])}
                  onChange={(e) => set(key, e.target.checked as never)}
                />
                {label}
              </label>
            ))}
            <label className="flex items-center gap-1.5 col-span-2">
              <input
                type="checkbox"
                checked={value.show_landmarks}
                onChange={(e) => set("show_landmarks", e.target.checked)}
              />
              Show landmark points
            </label>
          </div>

          {advanced && (
            <div className="space-y-2 pt-2 border-t border-slate-200 dark:border-slate-800">
              <label className="block space-y-1">
                <span className="text-slate-600 dark:text-slate-400">Landmark size: {value.landmark_size}px</span>
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={value.landmark_size}
                  onChange={(e) => set("landmark_size", Number(e.target.value))}
                  className="w-full"
                />
              </label>
              <label className="block space-y-1">
                <span className="text-slate-600 dark:text-slate-400">
                  Detection confidence: {Math.round(value.min_confidence * 100)}%
                </span>
                <input
                  type="range"
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  value={value.min_confidence}
                  onChange={(e) => set("min_confidence", Number(e.target.value))}
                  className="w-full"
                />
              </label>
              {!live && (
                <label className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={value.include_landmarks_in_report}
                    onChange={(e) => set("include_landmarks_in_report", e.target.checked)}
                  />
                  Include all 478 landmark coordinates in the JSON export
                </label>
              )}
            </div>
          )}

          <button
            type="button"
            onClick={() => onChange({ ...DEFAULT_FACE_MESH, enabled: true })}
            className="text-indigo-500 hover:underline"
          >
            Reset mesh style
          </button>
        </>
      )}
    </div>
  );
}
