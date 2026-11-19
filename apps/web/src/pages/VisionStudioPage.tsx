import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Accessibility, Car, Hand, ScanSearch, ShieldCheck, Smile } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { api } from "../lib/api";
import type { MediaAsset } from "../types/api";

type VisionTask = "objects" | "vehicles" | "mask" | "facial" | "hands" | "pose";

export function VisionStudioPage() {
  const { data: caps } = useQuery({
    queryKey: ["vision-caps"],
    queryFn: api.visionCapabilities,
  });

  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [task, setTask] = useState<VisionTask>("objects");
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

  const isTaskReady = (t: VisionTask): boolean => {
    if (!caps) return false;
    if (t === "objects" || t === "vehicles") return caps.object_detection_ready;
    if (t === "mask") return caps.face_mask_ready;
    if (t === "facial") return caps.facial_analysis_ready;
    if (t === "hands") return caps.hand_landmarks_ready;
    if (t === "pose") return caps.pose_estimation_ready;
    return false;
  };

  const runAnalysis = async () => {
    if (!media) return;
    setError(null);
    setBusy(true);
    try {
      if (task === "vehicles") {
        if (media.media_type === "video") {
          setResults(await api.countVehicles({ media_id: media.id }));
        } else {
          setResults(await api.detectObjects({ media_id: media.id, vehicles_only: true }));
        }
      } else if (task === "objects") {
        setResults(await api.detectObjects({ media_id: media.id, vehicles_only: false }));
      } else if (task === "mask") {
        setResults(await api.detectFaceMasks({ media_id: media.id }));
      } else if (task === "facial") {
        setResults(await api.analyzeFacialLandmarks({ media_id: media.id }));
      } else if (task === "hands") {
        setResults(await api.detectHands({ media_id: media.id }));
      } else if (task === "pose") {
        setResults(await api.detectPose({ media_id: media.id }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  const tasks: Array<{ id: VisionTask; label: string; icon: typeof ScanSearch; ready: boolean }> = [
    { id: "objects", label: "Objects", icon: ScanSearch, ready: isTaskReady("objects") },
    { id: "vehicles", label: "Vehicles", icon: Car, ready: isTaskReady("vehicles") },
    { id: "mask", label: "Face Mask", icon: ShieldCheck, ready: isTaskReady("mask") },
    { id: "facial", label: "Facial Mesh", icon: Smile, ready: isTaskReady("facial") },
    { id: "hands", label: "Hands", icon: Hand, ready: isTaskReady("hands") },
    { id: "pose", label: "Body Pose", icon: Accessibility, ready: isTaskReady("pose") },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">AI Vision &amp; Analytics Studio</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Object detection, vehicle counting, face mask classification, facial geometry, hands, and pose.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <DropZone
            accept="image/*,video/*"
            label={busy ? "Processing..." : "Drop an image or traffic video"}
            onFilesSelected={handleUpload}
            disabled={busy}
          />

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 text-xs">
            <div className="font-semibold text-sm">Select Vision Task</div>
            <div className="grid grid-cols-2 gap-2">
              {tasks.map(({ id, label, icon: Icon, ready }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setTask(id)}
                  className={`p-2 rounded-lg border text-left flex items-center justify-between ${
                    task === id
                      ? "border-indigo-500 bg-indigo-500/10 text-indigo-400"
                      : "border-slate-200 dark:border-slate-800"
                  }`}
                >
                  <span className="inline-flex items-center gap-1.5 font-medium">
                    <Icon className="h-3.5 w-3.5" /> {label}
                  </span>
                  <span
                    className={`h-2 w-2 rounded-full ${ready ? "bg-emerald-500" : "bg-amber-500"}`}
                    title={ready ? "Ready" : "Optional model/dependency missing"}
                  />
                </button>
              ))}
            </div>

            {!isTaskReady(task) && (
              <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-500">
                {task === "objects" || task === "vehicles"
                  ? caps?.object_model_hint
                  : task === "mask"
                  ? "Place face_mask_classifier.onnx in models/ to enable face mask classification."
                  : "Install optional 'mediapipe' package in .venv to enable this analyzer."}
              </div>
            )}

            <button
              type="button"
              disabled={!media || busy || !isTaskReady(task)}
              onClick={runAnalysis}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-semibold"
            >
              Run {tasks.find((t) => t.id === task)?.label} Analysis
            </button>
          </div>

          {caps?.sign_languages && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-2 text-xs">
              <div className="font-semibold text-sm">Sign Language Registry</div>
              {caps.sign_languages.map((sl) => (
                <div key={sl.code} className="flex items-center justify-between text-slate-600 dark:text-slate-400">
                  <span>{sl.code} ({sl.name})</span>
                  <span className="text-amber-500">{sl.model_available ? "Ready" : "Model Required"}</span>
                </div>
              ))}
            </div>
          )}

          {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-500 text-xs">{error}</div>}
        </div>

        <div className="lg:col-span-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          {results ? (
            <pre className="text-xs overflow-auto max-h-[500px] p-3 rounded-lg bg-slate-100 dark:bg-slate-950">
              {JSON.stringify(results, null, 2)}
            </pre>
          ) : (
            <div className="h-80 flex items-center justify-center text-sm text-slate-500">
              Upload media and run an AI Vision task to inspect structured results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
