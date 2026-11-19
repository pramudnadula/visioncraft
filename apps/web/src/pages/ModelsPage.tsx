import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, Cpu, XCircle } from "lucide-react";
import { api } from "../lib/api";

export function ModelsPage() {
  const { data: models, isLoading } = useQuery({ queryKey: ["models"], queryFn: api.models });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Management</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Available local detection backends and optional ONNX model adapters.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-slate-500">Loading models...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models?.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <Cpu className="h-5 w-5 text-indigo-400" />
                  <div>
                    <div className="font-semibold">{m.name}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      {m.id} · v{m.version} · {m.license}
                    </div>
                  </div>
                </div>
                {m.available ? (
                  <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                    <XCircle className="h-3.5 w-3.5" /> Unavailable
                  </span>
                )}
              </div>

              <p className="text-xs text-slate-600 dark:text-slate-400">{m.description}</p>

              <div className="flex flex-wrap gap-2 text-[11px]">
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  Confidence: {m.supports_confidence ? "Yes" : "No"}
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  Landmarks: {m.supports_landmarks ? "Yes" : "No"}
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  GPU: {m.supports_gpu ? (m.gpu_active ? "Active" : "Supported") : "CPU Only"}
                </span>
              </div>

              {m.unavailable_reason && (
                <div className="text-xs text-amber-400 bg-amber-500/10 rounded-lg p-2.5">
                  {m.unavailable_reason}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
