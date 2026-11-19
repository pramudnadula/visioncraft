import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Camera,
  Cpu,
  Film,
  History,
  Image as ImageIcon,
  Layers,
  LayoutDashboard,
  Moon,
  Settings,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/images", label: "Image Detection", icon: ImageIcon },
  { to: "/videos", label: "Video Detection", icon: Film },
  { to: "/webcam", label: "Live Webcam", icon: Camera },
  { to: "/batch", label: "Batch Processing", icon: Layers },
  { to: "/history", label: "History", icon: History },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Layout() {
  const [dark, setDark] = useState(true);
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
  });

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
  };

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <aside className="w-64 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 flex flex-col">
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <div className="font-semibold tracking-tight">FaceVision</div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              AI Privacy Platform
            </div>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-indigo-600 text-white"
                    : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500 dark:text-slate-400">Backend</span>
            <span
              className={`px-2 py-0.5 rounded-full font-medium ${
                health?.status === "ok"
                  ? "bg-emerald-500/15 text-emerald-500"
                  : "bg-rose-500/15 text-rose-500"
              }`}
            >
              {health ? health.execution_mode.replace(/_/g, " ") : "offline"}
            </span>
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            {dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            <span>{dark ? "Light Theme" : "Dark Theme"}</span>
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
