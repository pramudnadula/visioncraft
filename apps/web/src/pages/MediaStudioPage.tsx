import { useState } from "react";
import { Crop, Download, Maximize2, Minimize2, RefreshCw } from "lucide-react";
import { DropZone } from "../components/DropZone";
import { ImageCropModal } from "../components/ImageCropModal";
import { api } from "../lib/api";
import type { ImageMetadataInfo, ImageOperationResponse, MediaAsset } from "../types/api";

type ToolTab = "crop" | "resize" | "convert" | "compress";

export function MediaStudioPage() {
  const [media, setMedia] = useState<MediaAsset | null>(null);
  const [metadata, setMetadata] = useState<ImageMetadataInfo | null>(null);
  const [tab, setTab] = useState<ToolTab>("compress");
  const [showCropModal, setShowCropModal] = useState(false);
  const [result, setResult] = useState<ImageOperationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Resize state
  const [resizeW, setResizeW] = useState<string>("1280");
  const [resizeH, setResizeH] = useState<string>("");
  const [preserveAspect, setPreserveAspect] = useState(true);

  // Convert state
  const [targetFormat, setTargetFormat] = useState("webp");
  const [convQuality, setConvQuality] = useState(90);
  const [bgColor, setBgColor] = useState("#ffffff");

  // Compress state
  const [compressMode, setCompressMode] = useState<"target" | "quality">("target");
  const [targetKb, setTargetKb] = useState(200);
  const [compQuality, setCompQuality] = useState(75);
  const [compFormat, setCompFormat] = useState("jpg");

  const handleUpload = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const uploaded = await api.uploadMedia(file);
      setMedia(uploaded);
      const meta = await api.imageMetadata(uploaded.id);
      setMetadata(meta);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const runResize = async () => {
    if (!media) return;
    setError(null);
    setBusy(true);
    try {
      setResult(
        await api.resizeImage({
          media_id: media.id,
          width: resizeW ? Number(resizeW) : null,
          height: resizeH ? Number(resizeH) : null,
          preserve_aspect: preserveAspect,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Resize failed");
    } finally {
      setBusy(false);
    }
  };

  const runConvert = async () => {
    if (!media) return;
    setError(null);
    setBusy(true);
    try {
      setResult(
        await api.convertImage({
          media_id: media.id,
          target_format: targetFormat,
          quality: convQuality,
          background_color: bgColor,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Conversion failed");
    } finally {
      setBusy(false);
    }
  };

  const runCompress = async () => {
    if (!media) return;
    setError(null);
    setBusy(true);
    try {
      setResult(
        await api.compressImage({
          media_id: media.id,
          target_kb: compressMode === "target" ? targetKb : null,
          quality: compQuality,
          target_format: compFormat,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compression failed");
    } finally {
      setBusy(false);
    }
  };

  const formatBytes = (bytes: number) =>
    bytes >= 1024 * 1024
      ? `${(bytes / (1024 * 1024)).toFixed(2)} MB`
      : `${(bytes / 1024).toFixed(1)} KB`;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Media Editing Studio</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Crop, resize, convert formats, and compress images to exact target sizes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <DropZone
            accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
            label={busy ? "Processing..." : "Drop an image to edit, convert, or compress"}
            onFilesSelected={handleUpload}
            disabled={busy}
          />

          {metadata && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 text-xs space-y-1">
              <div className="font-semibold text-sm">{media?.original_filename}</div>
              <div className="text-slate-600 dark:text-slate-400">
                {metadata.format} · {metadata.width}×{metadata.height} · {formatBytes(metadata.size_bytes)}
              </div>
              <div className="text-slate-600 dark:text-slate-400">
                Alpha: {metadata.has_alpha ? "Yes" : "No"} · Animated:{" "}
                {metadata.is_animated ? `Yes (${metadata.frame_count} frames)` : "No"}
              </div>
            </div>
          )}

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-4">
            <div className="grid grid-cols-2 gap-2 text-xs">
              {(
                [
                  { id: "crop", label: "Crop", icon: Crop },
                  { id: "resize", label: "Resize", icon: Maximize2 },
                  { id: "convert", label: "Convert", icon: RefreshCw },
                  { id: "compress", label: "Compress", icon: Minimize2 },
                ] as const
              ).map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setTab(id)}
                  className={`py-2 px-3 rounded-lg font-medium inline-flex items-center justify-center gap-1.5 ${
                    tab === id
                      ? "bg-indigo-600 text-white"
                      : "border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" /> {label}
                </button>
              ))}
            </div>

            {tab === "crop" && (
              <div className="space-y-3 text-xs">
                <p className="text-slate-600 dark:text-slate-400">
                  Open the interactive Crop &amp; Transform modal with aspect presets, rotation, and flip.
                </p>
                <button
                  type="button"
                  disabled={!media}
                  onClick={() => setShowCropModal(true)}
                  className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-semibold"
                >
                  Open Cropper
                </button>
              </div>
            )}

            {tab === "resize" && (
              <div className="space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-2">
                  <label>
                    <span className="block text-slate-600 dark:text-slate-400 mb-1">Width (px)</span>
                    <input
                      type="number"
                      value={resizeW}
                      onChange={(e) => setResizeW(e.target.value)}
                      className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-1.5"
                    />
                  </label>
                  <label>
                    <span className="block text-slate-600 dark:text-slate-400 mb-1">Height (px)</span>
                    <input
                      type="number"
                      value={resizeH}
                      placeholder="Auto"
                      onChange={(e) => setResizeH(e.target.value)}
                      className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-1.5"
                    />
                  </label>
                </div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={preserveAspect}
                    onChange={(e) => setPreserveAspect(e.target.checked)}
                  />
                  <span>Preserve aspect ratio</span>
                </label>
                <button
                  type="button"
                  disabled={!media || busy}
                  onClick={runResize}
                  className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-semibold"
                >
                  Resize Image
                </button>
              </div>
            )}

            {tab === "convert" && (
              <div className="space-y-3 text-xs">
                <div>
                  <label className="block text-slate-600 dark:text-slate-400 mb-1">Target Format</label>
                  <select
                    value={targetFormat}
                    onChange={(e) => setTargetFormat(e.target.value)}
                    className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-1.5"
                  >
                    {["webp", "jpeg", "png", "bmp", "tiff", "gif"].map((f) => (
                      <option key={f} value={f} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                        {f.toUpperCase()}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-slate-600 dark:text-slate-400 mb-1">
                    Quality ({convQuality}%)
                  </label>
                  <input
                    type="range"
                    min={10}
                    max={100}
                    value={convQuality}
                    onChange={(e) => setConvQuality(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-slate-600 dark:text-slate-400 mb-1">
                    Transparency Background (for JPEG/BMP)
                  </label>
                  <input
                    type="color"
                    value={bgColor}
                    onChange={(e) => setBgColor(e.target.value)}
                    className="h-8 w-16 rounded border border-slate-300 dark:border-slate-700"
                  />
                </div>
                <button
                  type="button"
                  disabled={!media || busy}
                  onClick={runConvert}
                  className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-semibold"
                >
                  Convert Format
                </button>
              </div>
            )}

            {tab === "compress" && (
              <div className="space-y-3 text-xs">
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setCompressMode("target")}
                    className={`flex-1 py-1.5 rounded border ${
                      compressMode === "target" ? "border-indigo-500 text-indigo-400" : "border-slate-700"
                    }`}
                  >
                    Target KB
                  </button>
                  <button
                    type="button"
                    onClick={() => setCompressMode("quality")}
                    className={`flex-1 py-1.5 rounded border ${
                      compressMode === "quality" ? "border-indigo-500 text-indigo-400" : "border-slate-700"
                    }`}
                  >
                    Quality %
                  </button>
                </div>

                {compressMode === "target" ? (
                  <div>
                    <label className="block text-slate-600 dark:text-slate-400 mb-1">Target Size (KB)</label>
                    <div className="flex flex-wrap gap-1.5 mb-2">
                      {[50, 100, 200, 500, 1024].map((kb) => (
                        <button
                          key={kb}
                          type="button"
                          onClick={() => setTargetKb(kb)}
                          className={`px-2 py-0.5 rounded border text-[11px] ${
                            targetKb === kb ? "border-indigo-500 text-indigo-400" : "border-slate-700"
                          }`}
                        >
                          {kb >= 1024 ? "1 MB" : `${kb} KB`}
                        </button>
                      ))}
                    </div>
                    <input
                      type="number"
                      min={10}
                      max={50000}
                      value={targetKb}
                      onChange={(e) => setTargetKb(Number(e.target.value))}
                      className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-1.5"
                    />
                  </div>
                ) : (
                  <div>
                    <label className="block text-slate-600 dark:text-slate-400 mb-1">
                      Quality ({compQuality}%)
                    </label>
                    <input
                      type="range"
                      min={10}
                      max={95}
                      value={compQuality}
                      onChange={(e) => setCompQuality(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-slate-600 dark:text-slate-400 mb-1">Output Format</label>
                  <select
                    value={compFormat}
                    onChange={(e) => setCompFormat(e.target.value)}
                    className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-transparent px-2.5 py-1.5"
                  >
                    <option value="jpg" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">JPEG</option>
                    <option value="webp" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">WEBP</option>
                    <option value="png" className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">PNG</option>
                  </select>
                </div>

                <button
                  type="button"
                  disabled={!media || busy}
                  onClick={runCompress}
                  className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-semibold"
                >
                  Compress Image
                </button>
              </div>
            )}
          </div>

          {error && <div className="p-3 rounded-lg bg-rose-500/15 text-rose-500 text-xs">{error}</div>}
        </div>

        <div className="lg:col-span-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-4">
          {!media ? (
            <div className="h-80 flex items-center justify-center text-sm text-slate-500">
              Upload an image to start editing.
            </div>
          ) : (
            <>
              <img
                src={result ? result.download_url : api.mediaFileUrl(media.id)}
                alt={media.original_filename}
                className="max-h-[460px] mx-auto rounded-lg object-contain"
              />
              {result && (
                <div className="p-4 rounded-lg border border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="space-y-1">
                    <div>
                      <strong>{result.output_format}</strong> · {result.width}×{result.height} ·{" "}
                      Original: <strong>{formatBytes(result.original_size_bytes)}</strong> → New:{" "}
                      <strong>{formatBytes(result.output_size_bytes)}</strong> (
                      <span className="text-emerald-500">{result.reduction_percent}% reduction</span>)
                    </div>
                    {result.note && <div className="text-amber-500">{result.note}</div>}
                  </div>
                  <a
                    href={result.download_url}
                    download={result.output_filename}
                    className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold inline-flex items-center gap-1.5"
                  >
                    <Download className="h-3.5 w-3.5" /> Download
                  </a>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {showCropModal && media && (
        <ImageCropModal
          media={media}
          onClose={() => setShowCropModal(false)}
          onApplied={(res) => setResult(res)}
        />
      )}
    </div>
  );
}
