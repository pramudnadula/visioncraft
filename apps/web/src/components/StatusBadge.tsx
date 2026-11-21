const STATUS_STYLES: Record<string, string> = {
  COMPLETED: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  FAILED: "bg-rose-500/15 text-rose-600 dark:text-rose-400",
  CANCELLED: "bg-slate-500/15 text-slate-600 dark:text-slate-400",
  PROCESSING: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  VALIDATING: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  FINALIZING: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  QUEUED: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
};

const FALLBACK = "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200";

export function StatusBadge({ status }: { status: string }) {
  const key = status.toUpperCase();
  const label = key.charAt(0) + key.slice(1).toLowerCase();
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_STYLES[key] ?? FALLBACK}`}>
      {label}
    </span>
  );
}
