import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Accessibility,
  Box,
  Car,
  ChevronDown,
  FileImage,
  FileVideo,
  Hand,
  Loader2,
  ScanSearch,
  ShieldCheck,
  Smile,
  X,
} from "lucide-react";
import { DropZone } from "../components/DropZone";
import { api } from "../lib/api";
import type { MediaAsset } from "../types/api";

type VisionTask = "mesh3d" | "objects" | "vehicles" | "mask" | "facial" | "hands" | "pose" | "asl";
type Result = Record<string, unknown>;

interface Pt {
  x: number;
  y: number;
}
interface BoxShape {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color: string;
}
interface Overlay {
  boxes: BoxShape[];
  points: Pt[];
  skeletons: Array<{ pts: Array<Pt | null>; edges: Array<[number, number]>; color: string }>;
}

const HAND_EDGES: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const POSE_EDGES: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [28, 30], [30, 32],
  [0, 2], [0, 5], [2, 7], [5, 8], [9, 10],
];

const PERSON_COLORS = ["#22c55e", "#f59e0b", "#06b6d4", "#ec4899", "#a855f7", "#eab308", "#3b82f6", "#f43f5e"];

const MASK_COLORS: Record<string, string> = {
  mask_detected: "#10b981",
  no_mask: "#f43f5e",
  incorrect_mask: "#f59e0b",
  uncertain: "#94a3b8",
};

const VIDEO_TASKS: VisionTask[] = ["vehicles"];

function arr<T = Result>(v: unknown): T[] {
  return Array.isArray(v) ? (v as T[]) : [];
}

function num(v: unknown): number {
  return typeof v === "number" ? v : 0;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildOverlay(task: VisionTask, r: Result): Overlay {
  const o: Overlay = { boxes: [], points: [], skeletons: [] };
  if (task === "objects" || task === "vehicles") {
    for (const d of arr(r.detections)) {
      o.boxes.push({
        x: num(d.x), y: num(d.y), width: num(d.width), height: num(d.height),
        label: `${String(d.class_name ?? "object")} ${Math.round(num(d.confidence) * 100)}%`,
        color: "#6366f1",
      });
    }
  } else if (task === "mask") {
    for (const d of arr(r.detections)) {
      const status = String(d.status ?? "uncertain");
      o.boxes.push({
        x: num(d.x), y: num(d.y), width: num(d.width), height: num(d.height),
        label: `${humanize(status)} ${Math.round(num(d.confidence) * 100)}%`,
        color: MASK_COLORS[status] ?? "#94a3b8",
      });
    }
  } else if (task === "facial") {
    for (const f of arr(r.faces)) {
      const b = (f.bbox ?? {}) as Result;
      o.boxes.push({
        x: num(b.x), y: num(b.y), width: num(b.width), height: num(b.height),
        label: `Face ${num(f.face_index) + 1}`,
        color: "#06b6d4",
      });
      for (const p of arr<number[]>(f.landmarks)) o.points.push({ x: p[0], y: p[1] });
    }
  } else if (task === "mesh3d") {
    for (const f of arr(r.faces)) {
      for (const p of arr(f.landmarks_3d)) o.points.push({ x: num(p.x), y: num(p.y) });
    }
  } else if (task === "hands") {
    for (const h of arr(r.hands)) {
      o.skeletons.push({
        pts: arr(h.landmarks).map((p) => ({ x: num(p.x), y: num(p.y) })),
        edges: HAND_EDGES,
        color: "#f59e0b",
      });
    }
  } else if (task === "pose") {
    // Newer backends return one entry per person in `poses`; older ones only `landmarks`.
    const people = arr(r.poses).length ? arr(r.poses).map((p) => arr(p.landmarks)) : [arr(r.landmarks)];
    people.forEach((lms, i) => {
      // Hide points the model marks as not visible (occluded or out of frame).
      const pts = lms.map((p) =>
        typeof p.visibility === "number" && p.visibility < 0.5 ? null : { x: num(p.x), y: num(p.y) }
      );
      if (pts.some(Boolean)) o.skeletons.push({ pts, edges: POSE_EDGES, color: PERSON_COLORS[i % PERSON_COLORS.length] });
    });
  } else if (task === "asl") {
    for (const l of arr(r.asl_letters)) {
      const b = (l.bbox ?? {}) as Result;
      o.boxes.push({
        x: num(b.x), y: num(b.y), width: num(b.width), height: num(b.height),
        label: `${String(l.letter)} ${Math.round(num(l.confidence) * 100)}%`,
        color: "#ec4899",
      });
    }
  }
  return o;
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 p-3">
      <div className="text-[11px] text-slate-500 dark:text-slate-400">{label}</div>
      <div className={`text-lg font-bold ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

function HeadPose({ pose }: { pose: Result | undefined }) {
  if (!pose) return null;
  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {(["yaw_deg", "pitch_deg", "roll_deg"] as const).map((k) => (
        <span key={k} className="px-2 py-1 rounded-md bg-slate-100 dark:bg-slate-800">
          {humanize(k.replace("_deg", ""))}: <b>{num(pose[k]).toFixed(1)}°</b>
        </span>
      ))}
    </div>
  );
}

function Summary({ task, r }: { task: VisionTask; r: Result }) {
  if (task === "objects" || (task === "vehicles" && "detections" in r)) {
    const dets = arr(r.detections);
    const counts = dets.reduce<Record<string, number>>((acc, d) => {
      const k = String(d.class_name ?? "object");
      acc[k] = (acc[k] ?? 0) + 1;
      return acc;
    }, {});
    return (
      <div className="space-y-3">
        <Stat label={task === "vehicles" ? "Vehicles found" : "Objects found"} value={dets.length} />
        {Object.keys(counts).length > 0 && (
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(counts).map(([k, v]) => (
              <span key={k} className="px-2 py-1 rounded-md bg-indigo-500/10 text-indigo-600 dark:text-indigo-300">
                {k}: <b>{v}</b>
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }
  if (task === "vehicles") {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Stat label="Total vehicles" value={num(r.total_vehicles)} tone="text-indigo-500" />
        <Stat label="Cars" value={num(r.cars)} />
        <Stat label="Trucks" value={num(r.trucks)} />
        <Stat label="Buses" value={num(r.buses)} />
        <Stat label="Motorcycles" value={num(r.motorcycles)} />
        <Stat label="Bicycles" value={num(r.bicycles)} />
        <Stat label="Entering" value={num(r.entry_count)} />
        <Stat label="Exiting" value={num(r.exit_count)} />
        <div className="col-span-full text-xs text-slate-500">Frames analysed: {num(r.frames_processed)}</div>
      </div>
    );
  }
  if (task === "mask") {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <Stat label="Faces" value={num(r.face_count)} />
        <Stat label="Mask on" value={num(r.mask_detected)} tone="text-emerald-500" />
        <Stat label="No mask" value={num(r.no_mask)} tone="text-rose-500" />
        <Stat label="Worn incorrectly" value={num(r.incorrect_mask)} tone="text-amber-500" />
      </div>
    );
  }
  if (task === "facial" || task === "mesh3d") {
    const faces = arr(r.faces);
    return (
      <div className="space-y-3">
        <Stat label="Faces found" value={num(r.face_count)} />
        {faces.map((f, i) => (
          <div key={i} className="rounded-lg border border-slate-200 dark:border-slate-800 p-3 space-y-2">
            <div className="text-sm font-semibold">
              Face {i + 1} · {num(f.landmark_count)} landmarks
            </div>
            <HeadPose pose={f.head_pose as Result | undefined} />
            {arr(f.expression_indicators).length > 0 && (
              <div className="flex flex-wrap gap-1.5 text-[11px]">
                {arr(f.expression_indicators).slice(0, 6).map((e) => (
                  <span key={String(e.name)} className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-600 dark:text-cyan-300">
                    {String(e.name)} {Math.round(num(e.score) * 100)}%
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (task === "hands") {
    const hands = arr(r.hands);
    return (
      <div className="space-y-2">
        <Stat label="Hands found" value={num(r.hand_count)} />
        {hands.map((h, i) => (
          <div key={i} className="text-xs text-slate-600 dark:text-slate-300">
            Hand {i + 1}: <b>{String(h.handedness)}</b> · {arr(h.landmarks).length} points
          </div>
        ))}
      </div>
    );
  }
  if (task === "pose") {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <Stat
            label="People with a pose"
            value={arr(r.poses).length || (r.pose_detected ? 1 : 0)}
            tone="text-emerald-500"
          />
          <Stat label="Landmarks per person" value={num(r.landmark_count)} />
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Each person is drawn in a different colour. Points the model can't see (hidden behind someone, or out of
          frame) are left out, so people standing behind others may show only a partial skeleton.
        </p>
      </div>
    );
  }
  if (task === "asl") {
    const letters = arr(r.asl_letters);
    const gestures = arr(r.gestures);
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <Stat label="ASL letters" value={letters.map((l) => String(l.letter)).join(" ") || "—"} tone="text-pink-500" />
          <Stat label="Hand gestures" value={gestures.length} />
        </div>
        {gestures.map((g, i) => (
          <div key={i} className="text-xs text-slate-600 dark:text-slate-300">
            {String(g.handedness)} hand: <b>{humanize(String(g.gesture))}</b> ({Math.round(num(g.confidence) * 100)}%)
          </div>
        ))}
      </div>
    );
  }
  return null;
}

function isEmpty(task: VisionTask, r: Result): boolean {
  if (task === "objects" || (task === "vehicles" && "detections" in r)) return arr(r.detections).length === 0;
  if (task === "vehicles") return num(r.total_vehicles) === 0;
  if (task === "mask" || task === "facial" || task === "mesh3d") return num(r.face_count) === 0;
  if (task === "hands") return num(r.hand_count) === 0;
  if (task === "pose") return !r.pose_detected;
  if (task === "asl") return arr(r.asl_letters).length === 0 && arr(r.gestures).length === 0;
  return false;
}

const EMPTY_MESSAGES: Record<VisionTask, string> = {
  objects: "No objects were detected in this image.",
  vehicles: "No vehicles were detected.",
  mask: "No faces were found, so no mask check could be made.",
  facial: "No faces were found in this image.",
  mesh3d: "No faces were found in this image.",
  hands: "No hands were found. Try a photo where hands are clearly visible.",
  pose: "No person was found. Try a photo showing a person's upper body or full body.",
  asl: "No ASL letters or hand gestures were recognised. Try a close-up of one hand signing.",
};

export function VisionStudioPage() {
  const { data: caps } = useQuery({
    queryKey: ["vision-caps"],
    queryFn: api.visionCapabilities,
  });

  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [task, setTask] = useState<VisionTask>("mesh3d");
  const [results, setResults] = useState<Result | null>(null);
  const [resultTask, setResultTask] = useState<VisionTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  // Release the local object URL when it is replaced or the page unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const clearMedia = () => {
    setMedia(null);
    setPreviewUrl(null);
    setNaturalSize(null);
    setResults(null);
    setResultTask(null);
    setError(null);
  };

  const handleUpload = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setError(null);
    setResults(null);
    setResultTask(null);
    setNaturalSize(null);
    setMedia(null);
    setPreviewUrl(URL.createObjectURL(file));
    setUploading(true);
    try {
      const asset = await api.uploadMedia(file);
      setMedia(asset);
      if (asset.media_type === "video" && !VIDEO_TASKS.includes(task)) setTask("vehicles");
    } catch (err) {
      setPreviewUrl(null);
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const isTaskReady = (t: VisionTask): boolean => {
    if (!caps) return false;
    if (t === "mesh3d") return caps.face_mesh_3d_ready;
    if (t === "objects" || t === "vehicles") return caps.object_detection_ready;
    if (t === "mask") return caps.face_mask_ready;
    if (t === "facial") return caps.facial_analysis_ready;
    if (t === "hands") return caps.hand_landmarks_ready;
    if (t === "pose") return caps.pose_estimation_ready;
    if (t === "asl") return caps.sign_languages?.some((s) => s.code === "ASL" && s.model_available) ?? false;
    return false;
  };

  const isVideo = media?.media_type === "video";
  const taskSupportsMedia = !isVideo || VIDEO_TASKS.includes(task);

  const runAnalysis = async () => {
    if (!media) return;
    setError(null);
    setRunning(true);
    try {
      let r: Result;
      if (task === "mesh3d") r = await api.analyze3DMesh({ media_id: media.id });
      else if (task === "vehicles")
        r = isVideo
          ? await api.countVehicles({ media_id: media.id })
          : await api.detectObjects({ media_id: media.id, vehicles_only: true });
      else if (task === "objects") r = await api.detectObjects({ media_id: media.id, vehicles_only: false });
      else if (task === "mask") r = await api.detectFaceMasks({ media_id: media.id });
      else if (task === "facial") r = await api.analyzeFacialLandmarks({ media_id: media.id });
      else if (task === "hands") r = await api.detectHands({ media_id: media.id });
      else if (task === "pose") r = await api.detectPose({ media_id: media.id });
      else r = await api.detectSignLanguage({ media_id: media.id });
      setResults(r);
      setResultTask(task);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setRunning(false);
    }
  };

  const tasks: Array<{ id: VisionTask; label: string; icon: typeof ScanSearch }> = [
    { id: "mesh3d", label: "3D Face Mesh", icon: Box },
    { id: "objects", label: "Objects", icon: ScanSearch },
    { id: "vehicles", label: "Vehicles", icon: Car },
    { id: "mask", label: "Face Mask", icon: ShieldCheck },
    { id: "facial", label: "Face Mesh Data", icon: Smile },
    { id: "hands", label: "Hands", icon: Hand },
    { id: "pose", label: "Body Pose", icon: Accessibility },
    { id: "asl", label: "ASL Sign", icon: Hand },
  ];
  const taskLabel = tasks.find((t) => t.id === task)?.label;

  const disabledReason = !media
    ? uploading
      ? "Uploading…"
      : "Upload an image or video first."
    : !isTaskReady(task)
    ? null
    : !taskSupportsMedia
    ? `${taskLabel} works on images only. Upload a photo, or choose Vehicles to count traffic in a video.`
    : null;

  const overlay = results && resultTask && !isVideo ? buildOverlay(resultTask, results) : null;
  const imgW = naturalSize?.w ?? media?.width ?? 0;
  const imgH = naturalSize?.h ?? media?.height ?? 0;
  const stroke = Math.max(1.5, Math.min(imgW, imgH) / 300);

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
            label={uploading ? "Uploading…" : media ? "Drop another file to replace" : "Drop an image or traffic video"}
            onFilesSelected={handleUpload}
            disabled={uploading || running}
          />

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 text-xs">
            <div className="font-semibold text-sm">Select Vision Task</div>
            <div className="grid grid-cols-2 gap-2">
              {tasks.map(({ id, label, icon: Icon }) => {
                const ready = isTaskReady(id);
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setTask(id)}
                    className={`p-2 rounded-lg border text-left flex items-center justify-between transition-colors ${
                      task === id
                        ? "border-indigo-500 bg-indigo-500/10 text-indigo-600 dark:text-indigo-300"
                        : "border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60"
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
                );
              })}
            </div>

            {caps && !isTaskReady(task) && (
              <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
                {task === "objects" || task === "vehicles"
                  ? caps.object_model_hint
                  : task === "mask"
                  ? "Place face_mask_classifier.onnx in models/ to enable face mask classification."
                  : task === "facial"
                  ? "Face mesh needs the mediapipe package and models/face_landmarker.task (see docs/models/face-mesh.md)."
                  : "This task needs the optional mediapipe package and an extra model (see Model Hub)."}
              </div>
            )}

            <button
              type="button"
              disabled={!!disabledReason || !media || running || uploading || !isTaskReady(task)}
              onClick={runAnalysis}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold inline-flex items-center justify-center gap-2"
            >
              {running && <Loader2 className="h-4 w-4 animate-spin" />}
              {running ? `Running ${taskLabel}…` : `Run ${taskLabel} Analysis`}
            </button>
            {disabledReason && <p className="text-slate-500 dark:text-slate-400">{disabledReason}</p>}
          </div>

          {caps?.sign_languages && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-2 text-xs">
              <div className="font-semibold text-sm">Sign Language Registry</div>
              {caps.sign_languages.map((sl) => (
                <div key={sl.code} title={sl.note} className="flex items-center justify-between text-slate-600 dark:text-slate-400">
                  <span>{sl.code} ({sl.name})</span>
                  <span
                    className={
                      sl.model_available
                        ? "text-emerald-500 font-medium"
                        : sl.pipeline_ready
                        ? "text-sky-400 font-medium"
                        : "text-amber-500"
                    }
                  >
                    {sl.status_label ?? (sl.model_available ? "Ready" : "Model Required")}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="p-3 rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-400 text-sm">
              {error}
            </div>
          )}

          {!previewUrl ? (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 h-80 flex flex-col items-center justify-center gap-2 text-sm text-slate-500">
              <ScanSearch className="h-8 w-8 text-slate-400" />
              Upload an image or video, pick a task, then press Run.
            </div>
          ) : (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
              <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 text-xs">
                <div className="flex items-center gap-2 min-w-0">
                  {isVideo ? (
                    <FileVideo className="h-4 w-4 shrink-0 text-indigo-500" />
                  ) : (
                    <FileImage className="h-4 w-4 shrink-0 text-indigo-500" />
                  )}
                  <span className="font-medium truncate">{media?.original_filename ?? "Uploading…"}</span>
                  {media && (
                    <span className="text-slate-500 shrink-0">
                      {imgW > 0 && `${imgW}×${imgH} · `}
                      {formatBytes(media.size_bytes)}
                    </span>
                  )}
                  {uploading && <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" />}
                  {media && !uploading && (
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 shrink-0">
                      Uploaded
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={clearMedia}
                  aria-label="Remove file"
                  className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="relative bg-slate-100 dark:bg-slate-950 flex items-center justify-center">
                {isVideo ? (
                  <video src={previewUrl} controls className="max-h-[520px] w-auto" />
                ) : (
                  <div className="relative inline-block">
                    <img
                      src={previewUrl}
                      alt={media?.original_filename ?? "Uploaded image"}
                      className="block max-h-[520px] w-auto max-w-full"
                      onLoad={(e) =>
                        setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
                      }
                    />
                    {overlay && imgW > 0 && (
                      <svg
                        className="absolute inset-0 h-full w-full pointer-events-none"
                        viewBox={`0 0 ${imgW} ${imgH}`}
                        preserveAspectRatio="none"
                      >
                        {overlay.points.map((p, i) => (
                          <circle key={`p${i}`} cx={p.x} cy={p.y} r={stroke * 0.8} fill="#22d3ee" opacity={0.85} />
                        ))}
                        {overlay.skeletons.map((s, si) => (
                          <g key={`s${si}`}>
                            {s.edges.map(([a, b]) => {
                              const pa = s.pts[a];
                              const pb = s.pts[b];
                              return pa && pb ? (
                                <line
                                  key={`${a}-${b}`}
                                  x1={pa.x}
                                  y1={pa.y}
                                  x2={pb.x}
                                  y2={pb.y}
                                  stroke={s.color}
                                  strokeWidth={stroke * 1.5}
                                  strokeLinecap="round"
                                />
                              ) : null;
                            })}
                            {s.pts.map((p, i) =>
                              p ? (
                                <circle key={i} cx={p.x} cy={p.y} r={stroke * 2} fill="#fff" stroke={s.color} strokeWidth={stroke} />
                              ) : null
                            )}
                          </g>
                        ))}
                        {overlay.boxes.map((b, i) => {
                          const fs = Math.max(12, Math.min(imgW, imgH) / 28);
                          return (
                            <g key={`b${i}`}>
                              <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="none" stroke={b.color} strokeWidth={stroke * 2} rx={stroke * 2} />
                              <rect x={b.x} y={Math.max(0, b.y - fs * 1.4)} width={b.label.length * fs * 0.6 + fs * 0.6} height={fs * 1.4} fill={b.color} rx={stroke * 2} />
                              <text x={b.x + fs * 0.3} y={Math.max(0, b.y - fs * 1.4) + fs * 1.05} fill="#fff" fontSize={fs} fontWeight={600} fontFamily="ui-sans-serif, system-ui">
                                {b.label}
                              </text>
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>
                )}
                {running && (
                  <div className="absolute inset-0 bg-slate-950/40 flex items-center justify-center">
                    <div className="px-3 py-2 rounded-lg bg-slate-900/90 text-white text-sm inline-flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" /> Analysing…
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {results && resultTask && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold">{tasks.find((t) => t.id === resultTask)?.label} Results</h2>
              </div>
              {isEmpty(resultTask, results) ? (
                <p className="text-sm text-slate-500 dark:text-slate-400">{EMPTY_MESSAGES[resultTask]}</p>
              ) : (
                <Summary task={resultTask} r={results} />
              )}
              <div>
                <button
                  type="button"
                  onClick={() => setShowRaw((v) => !v)}
                  className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showRaw ? "rotate-180" : ""}`} />
                  {showRaw ? "Hide" : "Show"} raw JSON
                </button>
                {showRaw && (
                  <pre className="mt-2 text-xs overflow-auto max-h-[400px] p-3 rounded-lg bg-slate-100 dark:bg-slate-950">
                    {JSON.stringify(results, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
