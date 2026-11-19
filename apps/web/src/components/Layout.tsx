import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Camera,
  Cpu,
  FileText,
  Film,
  History,
  Image as ImageIcon,
  Layers,
  LayoutDashboard,
  Moon,
  ScanSearch,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { loadLocalPreferences, saveLocalPreferences } from "../lib/storage";

const NAV_SECTIONS = [
  {
    heading: "Overview",
    advancedOnly: false,
    items: [{ to: "/", label: "Dashboard", icon: LayoutDashboard, advancedOnly: false }],
  },
  {
    heading: "Face & Privacy",
    advancedOnly: false,
    items: [
      { to: "/images", label: "Image Detection", icon: ImageIcon, advancedOnly: false },
      { to: "/videos", label: "Video Detection", icon: Film, advancedOnly: false },
      { to: "/webcam", label: "Live Webcam", icon: Camera, advancedOnly: false },
      { to: "/batch", label: "Batch Queue", icon: Layers, advancedOnly: true },
    ],
  },
  {
    heading: "AI & Media Studios",
    advancedOnly: false,
    items: [
      { to: "/vision", label: "3D Mesh & AI Vision", icon: ScanSearch, advancedOnly: true },
      { to: "/media-studio", label: "Edit & Compress", icon: SlidersHorizontal, advancedOnly: false },
    ],
  },
  {
    heading: "Management",
    advancedOnly: false,
    items: [
      { to: "/reports", label: "Reports & PDF", icon: FileText, advancedOnly: false },
      { to: "/history", label: "History", icon: History, advancedOnly: true },
      { to: "/models", label: "AI Models", icon: Cpu, advancedOnly: true },
      { to: "/settings", label: "Settings", icon: Settings, advancedOnly: false },
    ],
  },
];

export function Layout() {
  const [dark, setDark] = useState(() => loadLocalPreferences().theme === "dark");
  const [uiMode, setUiMode] = useState<"simple" | "advanced">(() => loadLocalPreferences().uiMode);
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    saveLocalPreferences({ theme: next ? "dark" : "light" });
  };

  const toggleUiMode = () => {
    const next = uiMode === "simple" ? "advanced" : "simple";
    setUiMode(next);
    saveLocalPreferences({ uiMode: next });
  };

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <aside className="w-64 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/70 flex flex-col">
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-sm">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <div className="font-bold tracking-tight">FaceVision</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              AI Vision &amp; Privacy Suite
            </div>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-4 overflow-y-auto">
          {NAV_SECTIONS.map((section) => {
            const visibleItems = section.items.filter(
              (item) => uiMode === "advanced" || !item.advancedOnly
            );
            if (visibleItems.length === 0) return null;
            return (
              <div key={section.heading} className="space-y-1">
                <div className="px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  {section.heading}
                </div>
                {visibleItems.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={to === "/"}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isActive
                          ? "bg-indigo-600 text-white shadow-sm"
                          : "text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                      }`
                    }
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{label}</span>
                  </NavLink>
                ))}
              </div>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-200 dark:border-slate-800 space-y-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500 dark:text-slate-400">Engine</span>
            <span
              className={`px-2 py-0.5 rounded-full font-medium ${
                health?.status === "ok"
                  ? "bg-emerald-500/15 text-emerald-500"
                  : "bg-rose-500/15 text-rose-500"
              }`}
            >
              {health ? "Local Ready" : "Offline"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={toggleUiMode}
              className="flex items-center justify-center px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {uiMode === "simple" ? "Simple" : "Advanced"}
            </button>
            <button
              type="button"
              onClick={toggleTheme}
              className="flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              <span>{dark ? "Light" : "Dark"}</span>
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          <Outlet context={{ uiMode }} />
        </div>
      </main>
    </div>
  );
}
