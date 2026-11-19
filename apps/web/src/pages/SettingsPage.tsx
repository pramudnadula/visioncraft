import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Save } from "lucide-react";
import { api } from "../lib/api";
import type { BlurMode } from "../types/api";

export function SettingsPage() {
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });

  const [defaultModelId, setDefaultModelId] = useState("haar");
  const [defaultBlurMode, setDefaultBlurMode] = useState<BlurMode>("none");
  const [confidence, setConfidence] = useState(0.6);
  const [allowCloud, setAllowCloud] = useState(false);
  const [retentionDays, setRetentionDays] = useState(30);
  const [apiToken, setApiToken] = useState(() => localStorage.getItem("facevision_api_token") ?? "");
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!settings) return;
    setDefaultModelId(settings.default_model_id);
    setDefaultBlurMode(settings.default_blur_mode);
    setConfidence(settings.default_confidence_threshold);
    setAllowCloud(settings.allow_cloud_processing);
    setRetentionDays(settings.retention_days);
  }, [settings]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavedMsg(null);
    if (apiToken) {
      localStorage.setItem("facevision_api_token", apiToken);
    } else {
      localStorage.removeItem("facevision_api_token");
    }
    await api.updateSettings({
      default_model_id: defaultModelId,
      default_blur_mode: defaultBlurMode,
      default_confidence_threshold: confidence,
      allow_cloud_processing: allowCloud,
      retention_days: retentionDays,
    });
    setSavedMsg("Settings saved.");
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Configure default detection parameters, privacy retention, and hybrid cloud consent.
        </p>
      </div>

      <form
        onSubmit={handleSave}
        className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 space-y-5"
      >
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Default Model</label>
          <select
            value={defaultModelId}
            onChange={(e) => setDefaultModelId(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-transparent px-3 py-2 text-sm"
          >
            {models?.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available} className="bg-slate-900">
                {m.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">Default Privacy Blur</label>
          <select
            value={defaultBlurMode}
            onChange={(e) => setDefaultBlurMode(e.target.value as BlurMode)}
            className="w-full rounded-lg border border-slate-700 bg-transparent px-3 py-2 text-sm"
          >
            <option value="none" className="bg-slate-900">None</option>
            <option value="gaussian" className="bg-slate-900">Gaussian Blur</option>
            <option value="pixelate" className="bg-slate-900">Pixelate</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Default Confidence Threshold ({Math.round(confidence * 100)}%)
          </label>
          <input
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Media Retention (Days)
          </label>
          <input
            type="number"
            min={1}
            max={3650}
            value={retentionDays}
            onChange={(e) => setRetentionDays(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-700 bg-transparent px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Local / Cloud API Token (Optional)
          </label>
          <input
            type="password"
            value={apiToken}
            onChange={(e) => setApiToken(e.target.value)}
            placeholder="Leave empty if local auth is disabled"
            className="w-full rounded-lg border border-slate-700 bg-transparent px-3 py-2 text-sm"
          />
        </div>

        <label className="flex items-start gap-3 p-3 rounded-lg border border-slate-800">
          <input
            type="checkbox"
            checked={allowCloud}
            onChange={(e) => setAllowCloud(e.target.checked)}
            className="mt-1"
          />
          <div className="text-xs">
            <div className="font-semibold text-sm">Allow Optional Cloud Processing</div>
            <p className="text-slate-400 mt-0.5">
              Local processing is always default. Enabling this allows you to explicitly opt in to
              transferring media to a configured remote worker.
            </p>
          </div>
        </label>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold inline-flex items-center gap-2"
          >
            <Save className="h-4 w-4" /> Save Settings
          </button>
          {savedMsg && <span className="text-xs text-emerald-400">{savedMsg}</span>}
        </div>
      </form>
    </div>
  );
}
