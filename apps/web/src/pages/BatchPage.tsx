import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Download, Play, Trash2 } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { JobProgressCard } from "../components/JobProgressCard";
import { api } from "../lib/api";
import type { BlurMode, JobResponse, MediaAsset } from "../types/api";

export function BatchPage() {
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [assets, setAssets] = useState<MediaAsset[]>([]);
  const [modelId, setModelId] = useState("haar");
  const [blurMode, setBlurMode] = useState<BlurMode>("none");
  const [job, setJob] = useState<JobResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFiles = async (files: File[]) => {
    setError(null);
    setUploading(true);
    const imageFiles = files.filter((f) => /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
    try {
      const uploaded: MediaAsset[] = [];
      for (const file of imageFiles) {
        uploaded.push(await api.uploadMedia(file));
      }
      setAssets((prev) => [...prev, ...uploaded]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch upload failed");
    } finally {
      setUploading(false);
    }
  };

  const startBatch = async () => {
    if (assets.length === 0) return;
    setError(null);
    try {
      const created = await api.createJob({
        job_type: "batch",
        media_ids: assets.map((a) => a.id),
        model_id: modelId,
        processing_mode: "web_local_agent",
        cloud_consent: false,
        blur_mode: blurMode,
      });
      setJob(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start batch job");
    }
  };

  const manifest = job?.output_manifest as
    | {
        succeeded?: number;
        failed?: number;
        total_faces?: number;
        items?: Array<{ filename: string; status: string; face_count: number; error?: string }>;
      }
    | undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Batch Processing</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Upload multiple images or import a folder, process in background, and export as ZIP.
        </p>
      </div>

      <DropZone
        accept="image/png,image/jpeg,image/webp,image/bmp"
        multiple
        allowFolder
        label={uploading ? "Uploading files..." : "Drop multiple images or import a folder"}
        onFilesSelected={handleFiles}
        disabled={uploading}
      />

      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <select
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
        >
          {models?.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.available} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
              {m.name}
            </option>
          ))}
        </select>

        <select
          value={blurMode}
          onChange={(e) => setBlurMode(e.target.value as BlurMode)}
          className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
        >
          <option value="none" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">No Blur</option>
          <option value="gaussian" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Gaussian Blur</option>
          <option value="pixelate" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Pixelate</option>
          <option value="mask" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Solid Cover (Mask)</option>
          <option value="mesh" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Privacy Net (Mesh)</option>
        </select>

        <button
          type="button"
          disabled={assets.length === 0}
          onClick={startBatch}
          className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-semibold inline-flex items-center gap-2"
        >
          <Play className="h-4 w-4" /> Process {assets.length} Image(s)
        </button>

        {assets.length > 0 && (
          <button
            type="button"
            onClick={() => setAssets([])}
            className="ml-auto px-3 py-2 rounded-lg border border-slate-700 text-xs text-slate-600 dark:text-slate-400 hover:text-rose-400 inline-flex items-center gap-1"
          >
            <Trash2 className="h-3.5 w-3.5" /> Clear Queue
          </button>
        )}
      </div>

      {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-400 text-xs">{error}</div>}
      {job && <JobProgressCard job={job} onUpdated={setJob} />}

      {assets.length > 0 && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <h3 className="text-sm font-semibold mb-3">Queued Files ({assets.length})</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs">
            {assets.map((a) => (
              <div key={a.id} className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <span className="truncate">{a.original_filename}</span>
                {a.duplicate_of && <span className="text-amber-400 ml-2">dup</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {job?.status === "COMPLETED" && manifest && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm">
              Succeeded: <strong>{manifest.succeeded}</strong> · Failed:{" "}
              <strong>{manifest.failed}</strong> · Total Faces:{" "}
              <strong>{manifest.total_faces}</strong>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <a
                href={api.artifactUrl(job.id, "batch_outputs.zip")}
                download
                className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white inline-flex items-center gap-1"
              >
                <Download className="h-3.5 w-3.5" /> Download ZIP
              </a>
              <a
                href={api.artifactUrl(job.id, "batch_report.json")}
                download
                className="px-3 py-1.5 rounded-lg border border-slate-700 inline-flex items-center gap-1"
              >
                JSON
              </a>
              <a
                href={api.artifactUrl(job.id, "batch_summary.csv")}
                download
                className="px-3 py-1.5 rounded-lg border border-slate-700 inline-flex items-center gap-1"
              >
                CSV
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
