import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Activity, Camera, CheckCircle2, Film, Image as ImageIcon, Layers, ShieldAlert } from "lucide-react";
import { api } from "../lib/api";

export function DashboardPage() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ["stats"],
    queryFn: api.stats,
    refetchInterval: 5000,
  });

  const { data: recentJobs } = useQuery({
    queryKey: ["jobs", "recent"],
    queryFn: () => api.listJobs(),
    refetchInterval: 5000,
  });

  const cards = [
    { label: "Media Uploaded", value: stats?.total_media ?? 0, icon: ImageIcon, color: "text-indigo-400" },
    { label: "Faces Detected", value: stats?.total_faces_detected ?? 0, icon: CheckCircle2, color: "text-emerald-400" },
    { label: "Active Jobs", value: stats?.active_jobs ?? 0, icon: Activity, color: "text-amber-400" },
    { label: "Failed Jobs", value: stats?.failed_jobs ?? 0, icon: ShieldAlert, color: "text-rose-400" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Local-first AI face detection, tracking, and privacy anonymization.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
          <span className="text-slate-500">Mode:</span>
          <span className="font-semibold">
            {stats?.execution_mode.replace(/_/g, " ") ?? "loading..."}
          </span>
          <span className="mx-1 text-slate-600">·</span>
          <span className={stats?.cloud_processing_enabled ? "text-amber-400" : "text-emerald-400"}>
            {stats?.cloud_processing_enabled ? "Cloud Opt-In Enabled" : "Local Only"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map(({ label, value, icon: Icon, color }) => (
          <div
            key={label}
            className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</span>
              <Icon className={`h-4 w-4 ${color}`} />
            </div>
            <div className="mt-2 text-2xl font-bold">
              {isLoading ? "—" : value.toLocaleString()}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { to: "/images", title: "Image Studio", desc: "Detect & blur faces with before/after view", icon: ImageIcon },
          { to: "/videos", title: "Video Studio", desc: "Frame-by-frame tracking & MP4 export", icon: Film },
          { to: "/webcam", title: "Live Webcam", desc: "Real-time privacy blur & FPS HUD", icon: Camera },
          { to: "/batch", title: "Batch Queue", desc: "Folder import & ZIP export", icon: Layers },
        ].map(({ to, title, desc, icon: Icon }) => (
          <Link
            key={to}
            to={to}
            className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 hover:border-indigo-500 transition-colors"
          >
            <Icon className="h-5 w-5 text-indigo-500 mb-2" />
            <div className="font-semibold text-sm">{title}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{desc}</div>
          </Link>
        ))}
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold">Recent Processing Activity</h2>
          <Link to="/history" className="text-xs text-indigo-400 hover:underline">
            View all
          </Link>
        </div>
        {!recentJobs || recentJobs.length === 0 ? (
          <p className="text-sm text-slate-500">
            No processing jobs yet. Upload an image or video to get started.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 border-b border-slate-200 dark:border-slate-800">
                  <th className="pb-2">Job</th>
                  <th className="pb-2">Type</th>
                  <th className="pb-2">Model</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Faces</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {recentJobs.slice(0, 6).map((job) => (
                  <tr key={job.id}>
                    <td className="py-2.5 font-mono text-xs">{job.id.slice(0, 8)}</td>
                    <td className="py-2.5 capitalize">{job.job_type}</td>
                    <td className="py-2.5">{job.model_id}</td>
                    <td className="py-2.5">
                      <span className="px-2 py-0.5 rounded-full text-xs bg-slate-800 text-slate-200">
                        {job.status}
                      </span>
                    </td>
                    <td className="py-2.5">{job.faces_detected}</td>
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
