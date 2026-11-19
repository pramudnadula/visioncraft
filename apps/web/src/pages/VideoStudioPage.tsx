import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Download, Film, Play } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { FaceMeshControls, useFaceMeshSettings } from "../components/FaceMeshControls";
import { FaceMeshResults } from "../components/FaceMeshResults";
import { JobProgressCard } from "../components/JobProgressCard";
import { api } from "../lib/api";
import type { BlurMode, FaceMeshResult, JobResponse, MediaAsset } from "../types/api";

export function VideoStudioPage() {
  const outlet = useOutletContext<{ uiMode?: "simple" | "advanced" } | null>();
  const isAdvanced = outlet?.uiMode === "advanced";
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [modelId, setModelId] = useState("haar");
  const [blurMode, setBlurMode] = useState<BlurMode>("none");
  const [enableTracking, setEnableTracking] = useState(true);
  const [savePrivacyVideo, setSavePrivacyVideo] = useState(false);
  const [targetFps, setTargetFps] = useState<string>("");
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
        job_type: "video",
        media_ids: [media.id],
        model_id: modelId,
        processing_mode: "web_local_agent",
        cloud_consent: false,
        blur_mode: blurMode,
        enable_tracking: enableTracking,
        face_mesh: faceMesh.enabled ? faceMesh : null,
        save_privacy_video: savePrivacyVideo,
        target_fps: targetFps ? Number(targetFps) : null,
      });
      setJob(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start video job");
    }
  };

  const manifest = job?.output_manifest as
    | {
        frames_processed?: number;
        total_detections?: number;
        unique_track_count?: number;
        duration_ms?: number;
        face_mesh?: FaceMeshResult | null;
      }
    | undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Video Detection Studio</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Streaming frame-by-frame face detection, temporary ID tracking, and privacy blurring.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <DropZone
            accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,.mkv"
            label={uploading ? "Uploading video..." : "Drop a video file or click to browse"}
            onFilesSelected={handleUpload}
            disabled={uploading}
          />

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
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Target Processing FPS (Optional)</label>
                <input
                  type="number"
                  min={1}
                  max={60}
                  placeholder="Native FPS"
                  value={targetFps}
                  onChange={(e) => setTargetFps(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm"
                />
              </div>
            )}

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={enableTracking}
                onChange={(e) => setEnableTracking(e.target.checked)}
              />
              <span>Enable temporary face tracking IDs</span>
            </label>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={savePrivacyVideo}
                onChange={(e) => setSavePrivacyVideo(e.target.checked)}
              />
              <span>Also export clean privacy-blurred MP4</span>
            </label>

            <button
              type="button"
              disabled={!media}
              onClick={startProcessing}
              className="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-semibold inline-flex items-center justify-center gap-2"
            >
              <Play className="h-4 w-4" /> Start Video Processing
            </button>
          </div>

          <FaceMeshControls value={faceMesh} onChange={setFaceMesh} advanced={isAdvanced} />

          {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-400 text-xs">{error}</div>}
          {job && <JobProgressCard job={job} onUpdated={setJob} />}
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
            {!media ? (
              <div className="h-80 flex flex-col items-center justify-center text-slate-500 text-sm">
                <Film className="h-8 w-8 mb-2 opacity-50" />
                Upload a video to preview and process.
              </div>
            ) : (
              <video
                key={job?.status === "COMPLETED" ? job.id : media.id}
                controls
                className="w-full max-h-[460px] rounded-lg bg-black"
                src={
                  job?.status === "COMPLETED"
                    ? api.artifactUrl(job.id, "processed.mp4")
                    : api.mediaFileUrl(media.id)
                }
              />
            )}

            {job?.status === "COMPLETED" && manifest && (
              <div className="mt-4 flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-slate-200 dark:border-slate-800 text-xs">
                <div className="flex flex-wrap gap-4">
                  <span>Frames: <strong>{manifest.frames_processed}</strong></span>
                  <span>Detections: <strong>{manifest.total_detections}</strong></span>
                  <span>Temp Tracks: <strong>{manifest.unique_track_count}</strong></span>
                  <span>Time: <strong>{manifest.duration_ms} ms</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href={api.artifactUrl(job.id, "processed.mp4")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> MP4
                  </a>
                  <a
                    href={api.artifactUrl(job.id, "report.json")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> Timeline JSON
                  </a>
                  <a
                    href={api.artifactUrl(job.id, "summary.csv")}
                    download
                    className="px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1"
                  >
                    <Download className="h-3.5 w-3.5" /> CSV
                  </a>
                </div>
              </div>
            )}
          </div>
          {job?.status === "COMPLETED" && (
            <FaceMeshResults
              result={manifest?.face_mesh}
              downloadUrl={manifest?.face_mesh ? api.artifactUrl(job.id, "face_mesh.json") : undefined}
            />
          )}
        </div>
      </div>
    </div>
  );
}
