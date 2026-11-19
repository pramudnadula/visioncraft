import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Download, Play, Sparkles } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { FaceMeshControls, useFaceMeshSettings } from "../components/FaceMeshControls";
import { FaceMeshResults } from "../components/FaceMeshResults";
import { JobProgressCard } from "../components/JobProgressCard";
import { useDefaultDetectionSettings } from "../hooks/useDefaultDetectionSettings";
import { api } from "../lib/api";
import type { BlurMode, DetectionBox, FaceMeshResult, JobResponse, MediaAsset } from "../types/api";

export function ImageStudioPage() {
  const outlet = useOutletContext<{ uiMode?: "simple" | "advanced" } | null>();
  const isAdvanced = outlet?.uiMode === "advanced";
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const { settings, modelId, setModelId, blurMode, setBlurMode, confidence, setConfidence } =
    useDefaultDetectionSettings();

  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [saveCrops, setSaveCrops] = useState(true);
  const [useCloud, setUseCloud] = useState(false);
  const [cloudConsent, setCloudConsent] = useState(false);
  const [selectedFaces, setSelectedFaces] = useState<number[] | null>(null);
  const [viewMode, setViewMode] = useState<"annotated" | "original" | "privacy">("annotated");
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [faceMesh, setFaceMesh] = useFaceMeshSettings();

  const handleUpload = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    setJob(null);
    setSelectedFaces(null);
    try {
      const uploaded = await api.uploadMedia(file);
      setMedia(uploaded);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const startProcessing = async () => {
    if (!media) return;
    setError(null);
    try {
      const created = await api.createJob({
        job_type: "image",
        media_ids: [media.id],
        model_id: modelId || settings?.default_model_id || "haar",
        processing_mode: useCloud ? "cloud" : (settings?.execution_mode ?? "web_local_agent"),
        cloud_consent: useCloud ? cloudConsent : false,
        blur_mode: blurMode,
        confidence_threshold: confidence,
        save_crops: saveCrops,
        save_privacy_image: true,
        face_mesh: faceMesh.enabled ? faceMesh : null,
        selected_face_indices: selectedFaces,
      });
      setJob(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start job");
    }
  };

  const manifest = job?.output_manifest as
    | {
        face_count?: number;
        duration_ms?: number;
        detections?: DetectionBox[];
        face_mesh?: FaceMeshResult | null;
        files?: { annotated?: string; privacy_blurred?: string; crops?: string[] };
      }
    | undefined;

  const toggleFaceIndex = (idx: number, total: number) => {
    const current = selectedFaces ?? Array.from({ length: total }, (_, i) => i);
    const next = current.includes(idx) ? current.filter((i) => i !== idx) : [...current, idx];
    setSelectedFaces(next);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Image Detection Studio</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Upload any image, configure detection &amp; privacy blur, and compare results.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <DropZone
            accept="image/png,image/jpeg,image/webp,image/bmp"
            label={uploading ? "Uploading..." : "Drop an image or click to upload"}
            onFilesSelected={handleUpload}
            disabled={uploading}
          />

          {media?.duplicate_of && (
            <div className="text-xs px-3 py-2 rounded-lg bg-amber-500/15 text-amber-400">
              Duplicate image detected (matches existing asset {media.duplicate_of.slice(0, 8)}).
            </div>
          )}

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Detection Model</label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm"
              >
                {models?.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                    {m.name} {m.available ? "" : "(Unavailable)"}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Privacy Blur</label>
              <select
                value={blurMode}
                onChange={(e) => setBlurMode(e.target.value as BlurMode)}
                className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm"
              >
                <option value="none" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">None (Boxes Only)</option>
                <option value="gaussian" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Gaussian Blur</option>
                <option value="pixelate" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Pixelate</option>
          <option value="mask" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Solid Cover (Mask)</option>
          <option value="mesh" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Privacy Net (Mesh)</option>
              </select>
            </div>

            {isAdvanced && (
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  Confidence Threshold: {Math.round(confidence * 100)}%
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={0.95}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={saveCrops}
                onChange={(e) => setSaveCrops(e.target.checked)}
              />
              <span>Extract face crops (PNG)</span>
            </label>

            {settings?.allow_cloud_processing && (
              <div className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 space-y-2 text-xs">
                <label className="flex items-center justify-between">
                  <span className="font-medium">Processing Mode</span>
                  <select
                    value={useCloud ? "cloud" : "local"}
                    onChange={(e) => {
                      const nextCloud = e.target.value === "cloud";
                      setUseCloud(nextCloud);
                      if (!nextCloud) setCloudConsent(false);
                    }}
                    className="rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2 py-1"
                  >
                    <option value="local" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                      Local (Private)
                    </option>
                    <option value="cloud" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                      Cloud Server
                    </option>
                  </select>
                </label>
                {useCloud && (
                  <label className="flex items-start gap-2 text-amber-500">
                    <input
                      type="checkbox"
                      checked={cloudConsent}
                      onChange={(e) => setCloudConsent(e.target.checked)}
                      className="mt-0.5"
                    />
                    <span>
                      I consent to transferring this image to{" "}
                      {settings.cloud_api_url || "the configured cloud server"}.
                    </span>
                  </label>
                )}
              </div>
            )}

            <button
              type="button"
              disabled={!media || (useCloud && !cloudConsent)}
              onClick={startProcessing}
              className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-semibold inline-flex items-center justify-center gap-2"
            >
              <Play className="h-4 w-4" /> Start Processing
            </button>
          </div>

          <FaceMeshControls value={faceMesh} onChange={setFaceMesh} advanced={isAdvanced} />

          {error && (
            <div className="p-3 rounded-lg bg-rose-500/15 text-rose-400 text-xs">{error}</div>
          )}

          {job && <JobProgressCard job={job} onUpdated={setJob} />}
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2">
                {(["annotated", "original", "privacy"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize ${
                      viewMode === mode
                        ? "bg-indigo-600 text-white"
                        : "bg-slate-100 dark:bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>

              {job?.status === "COMPLETED" && (
                <div className="flex items-center gap-2 text-xs">
                  <a
                    href={api.artifactUrl(job.id, "annotated.png")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> PNG
                  </a>
                  <a
                    href={api.artifactUrl(job.id, "report.json")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> JSON
                  </a>
                  <a
                    href={api.artifactUrl(job.id, "summary.csv")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV
                  </a>
                </div>
              )}
            </div>

            {!media ? (
              <div className="h-80 flex flex-col items-center justify-center text-slate-500 text-sm">
                <Sparkles className="h-8 w-8 mb-2 opacity-50" />
                Upload an image to preview and detect faces.
              </div>
            ) : (
              <img
                src={
                  job?.status === "COMPLETED" && viewMode === "annotated"
                    ? `${api.artifactUrl(job.id, "annotated.png")}?t=${job.completed_at ?? ""}`
                    : job?.status === "COMPLETED" && viewMode === "privacy" && manifest?.files?.privacy_blurred
                    ? `${api.artifactUrl(job.id, manifest.files.privacy_blurred)}?t=${job.completed_at ?? ""}`
                    : api.mediaFileUrl(media.id)
                }
                alt={media.original_filename}
                className="max-h-[480px] mx-auto rounded-lg object-contain"
              />
            )}
          </div>

          {job?.status === "COMPLETED" && (
            <FaceMeshResults
              result={manifest?.face_mesh}
              downloadUrl={manifest?.face_mesh ? api.artifactUrl(job.id, "face_mesh.json") : undefined}
            />
          )}

          {manifest?.detections && manifest.detections.length > 0 && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold">
                  Detected Faces ({manifest.detections.length}) · {manifest.duration_ms} ms
                </h3>
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  Toggle faces for selective blur &amp; re-run
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {manifest.detections.map((det, idx) => {
                  const checked =
                    selectedFaces === null || selectedFaces.includes(idx);
                  return (
                    <label
                      key={idx}
                      className="flex items-center justify-between p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleFaceIndex(idx, manifest.detections!.length)}
                        />
                        <span>Face #{idx + 1}</span>
                      </div>
                      <span className="font-mono text-slate-600 dark:text-slate-400">
                        {det.width}×{det.height}
                        {det.confidence !== null ? ` · ${Math.round(det.confidence * 100)}%` : ""}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          )}

          {job?.status === "COMPLETED" && manifest?.files?.crops && manifest.files.crops.length > 0 && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
              <h3 className="text-sm font-semibold mb-3">Extracted Face Crops</h3>
              <div className="flex flex-wrap gap-3">
                {manifest.files.crops.map((crop) => (
                  <img
                    key={crop}
                    src={api.artifactUrl(job.id, crop)}
                    alt={crop}
                    className="h-20 w-20 rounded-lg object-cover border border-slate-700"
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
