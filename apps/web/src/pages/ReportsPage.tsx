import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FileText, PlusCircle } from "lucide-react";
import { api } from "../lib/api";

export function ReportsPage() {
  const queryClient = useQueryClient();
  const { data: jobs } = useQuery({
    queryKey: ["jobs", "COMPLETED"],
    queryFn: () => api.listJobs("COMPLETED"),
  });
  const { data: reports } = useQuery({
    queryKey: ["reports"],
    queryFn: api.listReports,
  });

  const [jobId, setJobId] = useState("");
  const [title, setTitle] = useState("FaceVision Analysis Report");
  const [template, setTemplate] = useState("simple_summary");
  const [format, setFormat] = useState<"pdf" | "csv" | "json">("pdf");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    const selectedJob = jobId || jobs?.[0]?.id;
    if (!selectedJob) return;
    setError(null);
    setBusy(true);
    try {
      await api.generateReport({ job_id: selectedJob, title, template, format });
      await queryClient.invalidateQueries({ queryKey: ["reports"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate report");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Reports &amp; Exports</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Generate plain-language PDF, CSV, and JSON reports from completed processing jobs.
        </p>
      </div>

      <form
        onSubmit={handleGenerate}
        className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 grid grid-cols-1 md:grid-cols-4 gap-3 items-end"
      >
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            Completed Job
          </label>
          <select
            value={jobId || jobs?.[0]?.id || ""}
            onChange={(e) => setJobId(e.target.value)}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          >
            {(!jobs || jobs.length === 0) && <option value="">No completed jobs</option>}
            {jobs?.map((j) => (
              <option key={j.id} value={j.id} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                {j.id.slice(0, 8)} · {j.job_type} ({j.faces_detected} faces)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            Report Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Template
            </label>
            <select
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-2 text-sm"
            >
              <option value="simple_summary" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Simple</option>
              <option value="business_report" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Business</option>
              <option value="detailed_technical" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Technical</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
              Format
            </label>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as "pdf" | "csv" | "json")}
              className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-2 text-sm"
            >
              <option value="pdf" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">PDF</option>
              <option value="csv" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">CSV</option>
              <option value="json" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">JSON</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={busy || !jobs || jobs.length === 0}
          className="py-2 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-semibold inline-flex items-center justify-center gap-2"
        >
          <PlusCircle className="h-4 w-4" />
          {busy ? "Generating..." : "Generate Report"}
        </button>
      </form>

      {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-500 text-xs">{error}</div>}

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
        <h2 className="font-semibold mb-3">Generated Reports</h2>
        {!reports || reports.length === 0 ? (
          <p className="text-sm text-slate-500">No reports generated yet.</p>
        ) : (
          <div className="space-y-2">
            {reports.map((r) => (
              <div
                key={r.id}
                className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between text-sm"
              >
                <div className="flex items-center gap-2.5">
                  <FileText className="h-4 w-4 text-indigo-400" />
                  <div>
                    <div className="font-medium">{r.title}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      {r.format.toUpperCase()} · {r.report_type} · Job {r.job_id.slice(0, 8)}
                    </div>
                  </div>
                </div>
                <a
                  href={r.download_url}
                  download
                  className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-medium inline-flex items-center gap-1.5"
                >
                  <Download className="h-3.5 w-3.5" /> Download
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
