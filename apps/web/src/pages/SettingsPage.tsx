import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Cloud, Save, XCircle } from "lucide-react";
import { api } from "../lib/api";
import type { BlurMode, CloudTestResponse } from "../types/api";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });
  const { data: models } = useQuery({ queryKey: ["models"], queryFn: api.models });

  const [defaultModelId, setDefaultModelId] = useState("haar");
  const [defaultBlurMode, setDefaultBlurMode] = useState<BlurMode>("none");
  const [confidence, setConfidence] = useState(0.6);
  const [allowCloud, setAllowCloud] = useState(false);
  const [cloudUrl, setCloudUrl] = useState("");
  const [retentionDays, setRetentionDays] = useState(30);
  const [apiToken, setApiToken] = useState(() => localStorage.getItem("facevision_api_token") ?? "");
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [cloudTest, setCloudTest] = useState<CloudTestResponse | null>(null);
  const [testingCloud, setTestingCloud] = useState(false);

  useEffect(() => {
    if (!settings) return;
    setDefaultModelId(settings.default_model_id);
    setDefaultBlurMode(settings.default_blur_mode);
    setConfidence(settings.default_confidence_threshold);
    setAllowCloud(settings.allow_cloud_processing);
    setCloudUrl(settings.cloud_api_url ?? "");
    setRetentionDays(settings.retention_days);
  }, [settings]);

  const handleTestCloud = async () => {
    if (!cloudUrl.trim()) return;
    setTestingCloud(true);
    setCloudTest(null);
    try {
      const res = await api.testCloudConnection(cloudUrl.trim());
      setCloudTest(res);
    } catch (err) {
      setCloudTest({
        reachable: false,
        status_code: null,
        message: err instanceof Error ? err.message : "Connection test failed.",
      });
    } finally {
      setTestingCloud(false);
    }
  };

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
      cloud_api_url: cloudUrl.trim() || null,
      retention_days: retentionDays,
    });
    await queryClient.invalidateQueries({ queryKey: ["settings"] });
    await queryClient.invalidateQueries({ queryKey: ["stats"] });
    setSavedMsg("Settings saved.");
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Configure default detection parameters, privacy retention, and optional remote cloud server.
        </p>
      </div>

      <form
        onSubmit={handleSave}
        className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 space-y-5"
      >
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Default Model</label>
          <select
            value={defaultModelId}
            onChange={(e) => setDefaultModelId(e.target.value)}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          >
            {models?.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                {m.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Default Privacy Blur</label>
          <select
            value={defaultBlurMode}
            onChange={(e) => setDefaultBlurMode(e.target.value as BlurMode)}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          >
            <option value="none" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">None</option>
            <option value="gaussian" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Gaussian Blur</option>
            <option value="pixelate" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">Pixelate</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
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
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            Media Retention (Days)
          </label>
          <input
            type="number"
            min={1}
            max={3650}
            value={retentionDays}
            onChange={(e) => setRetentionDays(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
            Local / Cloud API Token (Optional)
          </label>
          <input
            type="password"
            value={apiToken}
            onChange={(e) => setApiToken(e.target.value)}
            placeholder="Only required if FACEVISION_API_TOKEN is set on the server"
            className="w-full rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          />
          <p className="text-[11px] text-slate-500 mt-1">
            Secures API requests with <code>X-FaceVision-Token</code> when running on a shared network or cloud server.
          </p>
        </div>

        <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 space-y-3">
          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              checked={allowCloud}
              onChange={(e) => setAllowCloud(e.target.checked)}
              className="mt-1"
            />
            <div className="text-xs">
              <div className="font-semibold text-sm flex items-center gap-1.5">
                <Cloud className="h-4 w-4 text-indigo-500" />
                Allow Optional Cloud Processing
              </div>
              <p className="text-slate-600 dark:text-slate-400 mt-0.5">
                Local processing is always default. Enabling this unlocks the Cloud mode option in
                Image/Video studios after you configure a remote FaceVision server URL below.
              </p>
            </div>
          </label>

          {allowCloud && (
            <div className="pt-2 space-y-2 border-t border-slate-200 dark:border-slate-800">
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-400">
                Remote FaceVision Server URL
              </label>
              <div className="flex gap-2">
                <input
                  type="url"
                  value={cloudUrl}
                  onChange={(e) => setCloudUrl(e.target.value)}
                  placeholder="https://facevision-api.yourdomain.com"
                  className="flex-1 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
                />
                <button
                  type="button"
                  disabled={!cloudUrl.trim() || testingCloud}
                  onClick={handleTestCloud}
                  className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 text-xs font-medium"
                >
                  {testingCloud ? "Testing..." : "Test Connection"}
                </button>
              </div>
              {cloudTest && (
                <div
                  className={`flex items-center gap-1.5 text-xs ${
                    cloudTest.reachable ? "text-emerald-500" : "text-rose-500"
                  }`}
                >
                  {cloudTest.reachable ? (
                    <CheckCircle2 className="h-3.5 w-3.5" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5" />
                  )}
                  <span>{cloudTest.message}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold inline-flex items-center gap-2"
          >
            <Save className="h-4 w-4" /> Save Settings
          </button>
          {savedMsg && <span className="text-xs text-emerald-500">{savedMsg}</span>}
        </div>
      </form>
    </div>
  );
}
