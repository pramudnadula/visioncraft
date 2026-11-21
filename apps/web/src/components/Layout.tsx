import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Camera,
  Cpu,
  FileText,
  Film,
  History,
  Image as ImageIcon,
  Layers,
  LayoutDashboard,
  Menu,
  Moon,
  ScanSearch,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
  X,
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
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // Close the mobile drawer whenever the route changes.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
  });
  const { data: capabilities } = useQuery({
    queryKey: ["capabilities"],
    queryFn: api.capabilities,
  });
  const REQUIRED_MODES = ["mask", "mesh", "geometric", "mesh_3d"];
  const backendOutdated =
    !!capabilities &&
    REQUIRED_MODES.some((m) => !(capabilities.privacy_modes ?? []).includes(m));

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

  const engineReady = health?.status === "ok";

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      {/* Mobile top bar */}
      <header className="lg:hidden fixed top-0 inset-x-0 z-30 h-14 flex items-center justify-between px-4 border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 backdrop-blur">
        <button
          type="button"
          onClick={() => setNavOpen(true)}
          aria-label="Open navigation"
          className="p-2 -ml-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="flex items-center gap-2 font-bold tracking-tight">
          <ShieldCheck className="h-5 w-5 text-indigo-500" />
          FaceVision
        </div>
        <span
          className={`h-2.5 w-2.5 rounded-full ${engineReady ? "bg-emerald-500" : "bg-rose-500"}`}
          title={engineReady ? "Engine ready" : "Engine offline"}
        />
      </header>

      {navOpen && (
        <div
          className="lg:hidden fixed inset-0 z-40 bg-slate-950/50 backdrop-blur-sm"
          onClick={() => setNavOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col transition-transform duration-200 lg:sticky lg:top-0 lg:h-screen lg:translate-x-0 lg:z-auto ${
          navOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full"
        }`}
      >
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-sm">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="font-bold tracking-tight">FaceVision</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              AI Vision &amp; Privacy Suite
            </div>
          </div>
          <button
            type="button"
            onClick={() => setNavOpen(false)}
            aria-label="Close navigation"
            className="lg:hidden p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <X className="h-4 w-4" />
          </button>
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
              className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-medium ${
                engineReady ? "bg-emerald-500/15 text-emerald-500" : "bg-rose-500/15 text-rose-500"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${engineReady ? "bg-emerald-500" : "bg-rose-500"}`} />
              {engineReady ? "Local Ready" : "Offline"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={toggleUiMode}
              title={
                uiMode === "simple"
                  ? "Switch to Advanced mode to show Batch, AI Vision, History and Models"
                  : "Switch to Simple mode to hide advanced tools"
              }
              className="flex items-center justify-center px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {uiMode === "simple" ? "Simple mode" : "Advanced mode"}
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

      <main className="flex-1 min-w-0 px-4 pb-6 pt-20 sm:px-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          {backendOutdated && (
            <div className="mb-4 p-3 rounded-lg border border-amber-500/40 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              The running backend is older than this app, so some privacy modes (Solid Cover, Privacy Net,
              Geometric Mesh, 3D Mesh) will not be applied. Stop the backend (Ctrl+C) and start it again with{" "}
              <code>facevision-api</code>.
            </div>
          )}
          <Outlet context={{ uiMode }} />
        </div>
      </main>
    </div>
  );
}
