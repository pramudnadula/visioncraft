import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Cpu, Download, ExternalLink, Gauge, Search, Trash2, Upload, XCircle } from "lucide-react";
import { api } from "../lib/api";
import type { ModelSearchResult } from "../types/api";

export function ModelsPage() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: models, isLoading } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const { data: customModels } = useQuery({
    queryKey: ["custom-models"],
    queryFn: api.listCustomModels,
  });
  const { data: compat } = useQuery({
    queryKey: ["model-compat"],
    queryFn: api.modelCompatibility,
  });

  const [benchmarks, setBenchmarks] = useState<Record<string, { avg_inference_ms: number; estimated_fps: number; execution_provider: string }>>({});
  const [benchmarkingId, setBenchmarkingId] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("face detection");
  const [searchResults, setSearchResults] = useState<ModelSearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setErrorMsg(null);
    setSearching(true);
    try {
      const results = await api.searchModels(searchQuery.trim());
      setSearchResults(results);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Search failed");
    } finally {
      setSearching(false);
    }
  };

  const handleDownload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!downloadUrl.trim()) return;
    setErrorMsg(null);
    setStatusMsg(null);
    setDownloading(true);
    try {
      const res = await api.downloadModel(downloadUrl.trim());
      setStatusMsg(`Downloaded ${res.filename} (${(res.size_bytes / 1024 / 1024).toFixed(2)} MB).`);
      setDownloadUrl("");
      await queryClient.invalidateQueries({ queryKey: ["custom-models"] });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  };

  const handleImport = async (file: File | undefined) => {
    if (!file) return;
    setErrorMsg(null);
    setStatusMsg(null);
    setImporting(true);
    try {
      const res = await api.importModel(file);
      setStatusMsg(`Imported ${res.filename} (${(res.size_bytes / 1024 / 1024).toFixed(2)} MB).`);
      await queryClient.invalidateQueries({ queryKey: ["custom-models"] });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Import failed");
    } finally {
      setImporting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Management</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Built-in detectors plus deep learning ONNX search, download, and local import.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-slate-500">Loading models...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models?.map((m) => (
            <div
              key={m.id}
              className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <Cpu className="h-5 w-5 text-indigo-400" />
                  <div>
                    <div className="font-semibold">{m.name}</div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      {m.id} · v{m.version} · {m.license}
                    </div>
                  </div>
                </div>
                {m.available ? (
                  <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-500">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400">
                    <XCircle className="h-3.5 w-3.5" /> Unavailable
                  </span>
                )}
              </div>

              <p className="text-xs text-slate-600 dark:text-slate-400">{m.description}</p>

              <div className="flex flex-wrap gap-2 text-[11px]">
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  Confidence: {m.supports_confidence ? "Yes" : "No"}
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  Landmarks: {m.supports_landmarks ? "Yes" : "No"}
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  GPU: {m.supports_gpu ? (m.gpu_active ? "Active" : "Supported") : "CPU Only"}
                </span>
              </div>

              {m.unavailable_reason && (
                <div className="text-xs text-amber-500 bg-amber-500/10 rounded-lg p-2.5">
                  {m.unavailable_reason}
                </div>
              )}

              {m.available && (
                <div className="flex items-center justify-between pt-1">
                  <button
                    type="button"
                    disabled={benchmarkingId === m.id}
                    onClick={async () => {
                      setBenchmarkingId(m.id);
                      try {
                        const res = await api.benchmarkModel(m.id, 10);
                        setBenchmarks((prev) => ({ ...prev, [m.id]: res }));
                      } finally {
                        setBenchmarkingId(null);
                      }
                    }}
                    className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs inline-flex items-center gap-1"
                  >
                    <Gauge className="h-3.5 w-3.5" />
                    {benchmarkingId === m.id ? "Benchmarking..." : "Run Benchmark"}
                  </button>
                  {benchmarks[m.id] && (
                    <span className="text-xs font-mono text-emerald-500">
                      {benchmarks[m.id].avg_inference_ms} ms ({benchmarks[m.id].estimated_fps} FPS · {benchmarks[m.id].execution_provider})
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {compat && (
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 text-xs flex flex-wrap gap-4">
          <span>Python: <strong>{String(compat.python_version)}</strong></span>
          <span>OS: <strong>{String(compat.os_name)} ({String(compat.architecture)})</strong></span>
          <span>OpenCV: <strong>{String(compat.opencv_version)}</strong></span>
          <span>ONNXRuntime: <strong>{compat.onnxruntime_available ? "Ready" : "Not installed"}</strong></span>
          <span>Free Disk: <strong>{String(compat.free_disk_gb)} GB</strong></span>
        </div>
      )}

      {/* Search, Download & Import Deep Learning Models */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-semibold">Deep Learning ONNX Hub — Search, Download &amp; Import</h2>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              Search HuggingFace ONNX repositories, download from HTTPS, or import a local `.onnx` file.
            </p>
          </div>
          <button
            type="button"
            disabled={importing}
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-medium inline-flex items-center gap-1.5"
          >
            <Upload className="h-3.5 w-3.5" />
            {importing ? "Importing..." : "Import Local .onnx"}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".onnx,.xml,.prototxt,.caffemodel"
            className="hidden"
            onChange={(e) => {
              void handleImport(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>

        <form onSubmit={handleDownload} className="flex flex-wrap gap-2">
          <input
            type="url"
            value={downloadUrl}
            onChange={(e) => setDownloadUrl(e.target.value)}
            placeholder="https://huggingface.co/.../resolve/main/model.onnx"
            className="flex-1 min-w-[260px] rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={!downloadUrl.trim() || downloading}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-xs font-semibold inline-flex items-center gap-1.5"
          >
            <Download className="h-3.5 w-3.5" />
            {downloading ? "Downloading..." : "Download to models/custom/"}
          </button>
        </form>

        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search HuggingFace ONNX models (e.g. face detection, scrfd, yolov8-face)..."
            className="flex-1 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={searching}
            className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold inline-flex items-center gap-1.5"
          >
            <Search className="h-3.5 w-3.5" />
            {searching ? "Searching..." : "Search Hub"}
          </button>
        </form>

        {statusMsg && <div className="text-xs text-emerald-500">{statusMsg}</div>}
        {errorMsg && <div className="text-xs text-rose-500">{errorMsg}</div>}

        {searchResults.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
            {searchResults.map((item) => (
              <div
                key={item.id}
                className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2"
              >
                <div className="min-w-0">
                  <div className="font-semibold truncate">{item.id}</div>
                  <div className="text-slate-600 dark:text-slate-400">
                    {item.downloads.toLocaleString()} downloads · {item.likes} likes
                  </div>
                </div>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center gap-1 shrink-0"
                >
                  Open <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            ))}
          </div>
        )}

        {customModels && customModels.length > 0 && (
          <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-2">
            <h3 className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase">
              Imported Custom Models (models/custom/)
            </h3>
            <div className="flex flex-wrap gap-2 text-xs">
              {customModels.map((cm) => (
                <span
                  key={cm.filename}
                  className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 font-mono inline-flex items-center gap-2"
                >
                  {cm.filename} ({(cm.size_bytes / 1024 / 1024).toFixed(2)} MB)
                  <button
                    type="button"
                    onClick={async () => {
                      await api.deleteCustomModel(cm.filename);
                      await queryClient.invalidateQueries({ queryKey: ["custom-models"] });
                    }}
                    className="hover:text-rose-500"
                    title="Remove model"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
