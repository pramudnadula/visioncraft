import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Camera, CameraOff, Download } from "lucide-react";
import { api } from "../lib/api";
import type { BlurMode } from "../types/api";

export function WebcamPage() {
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [modelId, setModelId] = useState("haar");
  const [blurMode, setBlurMode] = useState<BlurMode>("none");
  const modelIdRef = useRef(modelId);
  const blurModeRef = useRef(blurMode);
  const [annotatedSrc, setAnnotatedSrc] = useState<string | null>(null);

  useEffect(() => {
    modelIdRef.current = modelId;
  }, [modelId]);

  useEffect(() => {
    blurModeRef.current = blurMode;
  }, [blurMode]);
  const [faceCount, setFaceCount] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    void navigator.mediaDevices.enumerateDevices().then((list) => {
      const cams = list.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      if (cams[0]) setSelectedDeviceId(cams[0].deviceId);
    });
    return () => {
      stopWebcam();
    };
  }, []);

  const stopWebcam = () => {
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setRunning(false);
  };

  const startWebcam = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera API is not available in this browser context.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true,
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(api.wsUrl("/ws/v1/webcam"));
      wsRef.current = ws;
      ws.onopen = () => {
        setRunning(true);
        sendNextFrame();
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as {
          face_count: number;
          latency_ms: number;
          annotated_frame?: string;
        };
        setFaceCount(data.face_count);
        setLatencyMs(data.latency_ms);
        if (data.annotated_frame) setAnnotatedSrc(data.annotated_frame);
        window.requestAnimationFrame(sendNextFrame);
      };
      ws.onerror = () => {
        setError("WebSocket connection to webcam processor failed.");
        stopWebcam();
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera permission denied.");
      stopWebcam();
    }
  };

  const sendNextFrame = () => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      window.setTimeout(sendNextFrame, 50);
      return;
    }
    canvas.width = 640;
    canvas.height = Math.round((video.videoHeight / video.videoWidth) * 640);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL("image/jpeg", 0.75);
    ws.send(
      JSON.stringify({
        frame,
        model_id: modelIdRef.current,
        blur_mode: blurModeRef.current,
        draw_boxes: true,
      })
    );
  };

  const captureScreenshot = () => {
    if (!annotatedSrc) return;
    const a = document.createElement("a");
    a.href = annotatedSrc;
    a.download = `facevision-webcam-${Date.now()}.jpg`;
    a.click();
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Live Webcam Detection</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Real-time face detection and live privacy blur powered by the shared Python engine.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <select
          value={selectedDeviceId}
          onChange={(e) => setSelectedDeviceId(e.target.value)}
          disabled={running}
          className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
        >
          {devices.length === 0 && <option value="">Default Camera</option>}
          {devices.map((d, idx) => (
            <option key={d.deviceId || idx} value={d.deviceId} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
              {d.label || `Camera ${idx + 1}`}
            </option>
          ))}
        </select>

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
        </select>

        {!running ? (
          <button
            type="button"
            onClick={startWebcam}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold inline-flex items-center gap-2"
          >
            <Camera className="h-4 w-4" /> Start Camera
          </button>
        ) : (
          <button
            type="button"
            onClick={stopWebcam}
            className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-sm font-semibold inline-flex items-center gap-2"
          >
            <CameraOff className="h-4 w-4" /> Stop Camera
          </button>
        )}

        {annotatedSrc && (
          <button
            type="button"
            onClick={captureScreenshot}
            className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-sm inline-flex items-center gap-1.5"
          >
            <Download className="h-4 w-4" /> Capture Snapshot
          </button>
        )}

        <div className="ml-auto text-xs font-mono text-slate-600 dark:text-slate-400">
          Faces: <strong className="text-slate-900 dark:text-slate-100">{faceCount}</strong> · Latency:{" "}
          <strong className="text-slate-900 dark:text-slate-100">{latencyMs.toFixed(0)} ms</strong>
        </div>
      </div>

      {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-400 text-sm">{error}</div>}

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-black p-2 flex items-center justify-center h-[calc(100vh-14rem)] min-h-[480px]">
        <video ref={videoRef} className="hidden" muted playsInline />
        <canvas ref={canvasRef} className="hidden" />
        {annotatedSrc ? (
          <img
            src={annotatedSrc}
            alt="Live webcam detection"
            className="w-full h-full object-contain rounded-lg"
          />
        ) : (
          <p className="text-sm text-slate-500">Click Start Camera to begin live face detection.</p>
        )}
      </div>
    </div>
  );
}
