import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Car, ScanSearch } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { api } from "../lib/api";
import type { MediaAsset } from "../types/api";

export function VisionStudioPage() {
  const { data: caps } = useQuery({
    queryKey: ["vision-caps"],
    queryFn: api.visionCapabilities,
  });

  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [vehiclesOnly, setVehiclesOnly] = useState(false);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleUpload = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setError(null);
    setResults(null);
    setBusy(true);
    try {
      setMedia(await api.uploadMedia(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const runAnalysis = async () => {
    if (!media) return;
    setError(null);
    setBusy(true);
    try {
      if (media.media_type === "video") {
        setResults(await api.countVehicles({ media_id: media.id }));
      } else {
        setResults(await api.detectObjects({ media_id: media.id, vehicles_only: vehiclesOnly }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Object &amp; Vehicle AI Studio</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          General COCO object detection, vehicle classification, and line-crossing counting.
        </p>
      </div>

      {caps && !caps.object_detection_ready && (
        <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-xs text-amber-500">
          {caps.object_model_hint}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <DropZone
            accept="image/*,video/*"
            label={busy ? "Processing..." : "Drop an image or traffic video"}
            onFilesSelected={handleUpload}
            disabled={busy}
          />

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={vehiclesOnly}
                onChange={(e) => setVehiclesOnly(e.target.checked)}
              />
              <span>Filter to vehicles only (car, bus, truck, motorcycle, bicycle)</span>
            </label>
            <button
              type="button"
              disabled={!media || busy || !caps?.object_detection_ready}
              onClick={runAnalysis}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-semibold inline-flex items-center justify-center gap-2"
            >
              {media?.media_type === "video" ? <Car className="h-4 w-4" /> : <ScanSearch className="h-4 w-4" />}
              {media?.media_type === "video" ? "Run Vehicle Counter" : "Detect Objects"}
            </button>
          </div>
          {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-500 text-xs">{error}</div>}
        </div>

        <div className="lg:col-span-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          {results ? (
            <pre className="text-xs overflow-auto max-h-[460px] p-3 rounded-lg bg-slate-100 dark:bg-slate-950">
              {JSON.stringify(results, null, 2)}
            </pre>
          ) : (
            <div className="h-72 flex items-center justify-center text-sm text-slate-500">
              Upload media and run detection to inspect object &amp; vehicle analytics.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
