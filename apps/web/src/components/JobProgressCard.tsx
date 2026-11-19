import { useEffect, useState } from "react";
import { RotateCcw, XCircle } from "lucide-react";
import { api } from "../lib/api";
import type { JobResponse } from "../types/api";

interface JobProgressCardProps {
  job: JobResponse;
  onUpdated: (job: JobResponse) => void;
}

export function JobProgressCard({ job, onUpdated }: JobProgressCardProps) {
  const [liveStatus, setLiveStatus] = useState(job.status);
  const [liveProgress, setLiveProgress] = useState(job.progress);
  const [liveFaces, setLiveFaces] = useState(job.faces_detected);

  const isTerminal =
    liveStatus === "COMPLETED" || liveStatus === "FAILED" || liveStatus === "CANCELLED";

  useEffect(() => {
    setLiveStatus(job.status);
    setLiveProgress(job.progress);
    setLiveFaces(job.faces_detected);
  }, [job.id, job.status, job.progress, job.faces_detected]);

  useEffect(() => {
    if (isTerminal) return;

    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(api.wsUrl(`/ws/v1/jobs/${job.id}`));
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data) as {
          status?: JobResponse["status"];
          progress?: number;
          faces_detected?: number;
        };
        if (data.status) setLiveStatus(data.status);
        if (typeof data.progress === "number") setLiveProgress(data.progress);
        if (typeof data.faces_detected === "number") setLiveFaces(data.faces_detected);
        if (
          data.status === "COMPLETED" ||
          data.status === "FAILED" ||
          data.status === "CANCELLED"
        ) {
          void api.getJob(job.id).then(onUpdated);
        }
      };
    } catch {
      // fallback to polling below
    }

    const timer = window.setInterval(async () => {
      try {
        const latest = await api.getJob(job.id);
        setLiveStatus(latest.status);
        setLiveProgress(latest.progress);
        setLiveFaces(latest.faces_detected);
        if (
          latest.status === "COMPLETED" ||
          latest.status === "FAILED" ||
          latest.status === "CANCELLED"
        ) {
          onUpdated(latest);
        }
      } catch {
        // ignore transient poll errors
      }
    }, 1000);

    return () => {
      ws?.close();
      window.clearInterval(timer);
    };
  }, [job.id, isTerminal, onUpdated]);

  const pct = Math.round(liveProgress * 100);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-mono text-slate-500">Job {job.id.slice(0, 8)}</span>
          <div className="text-sm font-semibold">
            {liveStatus} · {pct}% · {liveFaces} face(s)
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <button
              type="button"
              onClick={async () => onUpdated(await api.cancelJob(job.id))}
              className="px-3 py-1.5 rounded-lg bg-rose-600/15 text-rose-500 hover:bg-rose-600/25 text-xs font-medium inline-flex items-center gap-1"
            >
              <XCircle className="h-3.5 w-3.5" /> Cancel
            </button>
          )}
          {(liveStatus === "FAILED" || liveStatus === "CANCELLED") && (
            <button
              type="button"
              onClick={async () => onUpdated(await api.retryJob(job.id))}
              className="px-3 py-1.5 rounded-lg bg-indigo-600/15 text-indigo-400 hover:bg-indigo-600/25 text-xs font-medium inline-flex items-center gap-1"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Retry
            </button>
          )}
        </div>
      </div>

      <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            liveStatus === "FAILED"
              ? "bg-rose-500"
              : liveStatus === "COMPLETED"
              ? "bg-emerald-500"
              : "bg-indigo-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {job.error_message && (
        <p className="text-xs text-rose-500">{job.error_message}</p>
      )}
    </div>
  );
}
