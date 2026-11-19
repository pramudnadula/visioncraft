import type { BlurMode } from "../types/api";

const PREFS_KEY = "facevision_local_prefs";

export interface LocalBrowserPreferences {
  theme: "dark" | "light";
  uiMode: "simple" | "advanced";
  defaultModelId: string;
  defaultBlurMode: BlurMode;
  confidence: number;
}

const DEFAULT_PREFS: LocalBrowserPreferences = {
  theme: "dark",
  uiMode: "simple",
  defaultModelId: "haar",
  defaultBlurMode: "none",
  confidence: 0.6,
};

export function loadLocalPreferences(): LocalBrowserPreferences {
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    if (!raw) return DEFAULT_PREFS;
    return { ...DEFAULT_PREFS, ...(JSON.parse(raw) as Partial<LocalBrowserPreferences>) };
  } catch {
    return DEFAULT_PREFS;
  }
}

export function saveLocalPreferences(patch: Partial<LocalBrowserPreferences>): LocalBrowserPreferences {
  const next = { ...loadLocalPreferences(), ...patch };
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(next));
  } catch {
    // Ignore storage quota / private mode errors
  }
  return next;
}

export function clearLocalPreferences(): void {
  try {
    window.localStorage.removeItem(PREFS_KEY);
    window.localStorage.removeItem("facevision_api_token");
  } catch {
    // Ignore storage errors
  }
}
