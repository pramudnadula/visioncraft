import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, RotateCcw, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { StatusBadge } from "../components/StatusBadge";

export function HistoryPage() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>("");
  const { data: jobs, isLoading } = useQuery({
    queryKey: ["jobs", statusFilter],
    queryFn: () => api.listJobs(statusFilter || undefined),
    refetchInterval: 4000,
  });

  const handleDelete = async (id: string) => {
    await api.deleteJob(id);
    await queryClient.invalidateQueries({ queryKey: ["jobs"] });
  };

  const handleRetry = async (id: string) => {
    await api.retryJob(id);
    await queryClient.invalidateQueries({ queryKey: ["jobs"] });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Processing History</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Inspect past detection jobs, download reports, or clean up artifacts.
          </p>
        </div>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="rounded-lg border border-slate-300 dark:border-slate-700 bg-transparent px-3 py-2 text-sm"
        >
          <option value="" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">All Statuses</option>
          <option value="COMPLETED" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Completed</option>
          <option value="PROCESSING" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Processing</option>
          <option value="FAILED" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Failed</option>
          <option value="CANCELLED" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Cancelled</option>
        </select>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm text-slate-500">Loading job history...</div>
        ) : !jobs || jobs.length === 0 ? (
          <div className="p-6 text-sm text-slate-500">No jobs match the current filter.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800/60 text-left text-xs text-slate-600 dark:text-slate-400">
                <tr>
                  <th className="p-3">Job ID</th>
                  <th className="p-3">Type</th>
                  <th className="p-3">Model</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Faces</th>
                  <th className="p-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td className="p-3 font-mono text-xs">{job.id.slice(0, 8)}</td>
                    <td className="p-3 capitalize">{job.job_type}</td>
                    <td className="p-3">{job.model_id}</td>
                    <td className="p-3">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="p-3">{job.faces_detected}</td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        {job.status === "COMPLETED" && (
                          <a
                            href={api.artifactUrl(
                              job.id,
                              job.job_type === "batch" ? "batch_report.json" : "report.json"
                            )}
                            download
                            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
                            title="Download JSON report"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </a>
                        )}
                        {(job.status === "FAILED" || job.status === "CANCELLED") && (
                          <button
                            type="button"
                            onClick={() => handleRetry(job.id)}
                            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
                            title="Retry job"
                          >
                            <RotateCcw className="h-3.5 w-3.5" />
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => handleDelete(job.id)}
                          className="p-1.5 rounded-lg border border-slate-700 hover:text-rose-400"
                          title="Delete job"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
