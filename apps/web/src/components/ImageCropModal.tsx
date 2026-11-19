import { useState } from "react";
import { FlipHorizontal, FlipVertical, RotateCw, X } from "lucide-react";
import { api } from "../lib/api";
import type { ImageOperationResponse, MediaAsset } from "../types/api";

interface ImageCropModalProps {
  media: MediaAsset;
  onClose: () => void;
  onApplied: (result: ImageOperationResponse) => void;
}

const ASPECT_RATIOS: Array<{ label: string; ratio: number | null }> = [
  { label: "Free", ratio: null },
  { label: "1:1", ratio: 1 },
  { label: "4:3", ratio: 4 / 3 },
  { label: "3:2", ratio: 3 / 2 },
  { label: "16:9", ratio: 16 / 9 },
  { label: "9:16", ratio: 9 / 16 },
];

export function ImageCropModal({ media, onClose, onApplied }: ImageCropModalProps) {
  const maxW = media.width ?? 800;
  const maxH = media.height ?? 600;

  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  const [width, setWidth] = useState(maxW);
  const [height, setHeight] = useState(maxH);
  const [rotateDeg, setRotateDeg] = useState(0);
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyRatio = (ratio: number | null) => {
    if (!ratio) return;
    const newW = Math.min(maxW, Math.round(maxH * ratio));
    const newH = Math.min(maxH, Math.round(newW / ratio));
    setX(Math.max(0, Math.round((maxW - newW) / 2)));
    setY(Math.max(0, Math.round((maxH - newH) / 2)));
    setWidth(newW);
    setHeight(newH);
  };

  const handleApply = async () => {
    setError(null);
    setSubmitting(true);
    try {
      const res = await api.cropImage({
        media_id: media.id,
        x,
        y,
        width,
        height,
        rotate_deg: rotateDeg,
        flip_h: flipH,
        flip_v: flipV,
      });
      onApplied(res);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Crop failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Crop &amp; Transform Image</h3>
          <button type="button" onClick={onClose} className="p-1 rounded hover:bg-slate-800">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="bg-black/80 rounded-lg p-2 flex items-center justify-center max-h-72 overflow-hidden">
          <img
            src={api.mediaFileUrl(media.id)}
            alt={media.original_filename}
            className="max-h-64 object-contain transition-transform"
            style={{
              transform: `rotate(${rotateDeg}deg) scaleX(${flipH ? -1 : 1}) scaleY(${flipV ? -1 : 1})`,
            }}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          {ASPECT_RATIOS.map((ar) => (
            <button
              key={ar.label}
              type="button"
              onClick={() => applyRatio(ar.ratio)}
              className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 text-xs hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {ar.label}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setRotateDeg((d) => (d + 90) % 360)}
            className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 text-xs inline-flex items-center gap-1"
          >
            <RotateCw className="h-3.5 w-3.5" /> Rotate 90°
          </button>
          <button
            type="button"
            onClick={() => setFlipH((v) => !v)}
            className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 text-xs inline-flex items-center gap-1"
          >
            <FlipHorizontal className="h-3.5 w-3.5" /> Flip H
          </button>
          <button
            type="button"
            onClick={() => setFlipV((v) => !v)}
            className="px-2.5 py-1 rounded border border-slate-300 dark:border-slate-700 text-xs inline-flex items-center gap-1"
          >
            <FlipVertical className="h-3.5 w-3.5" /> Flip V
          </button>
        </div>

        <div className="grid grid-cols-4 gap-3 text-xs">
          <label>
            <span className="block text-slate-600 dark:text-slate-400 mb-1">X</span>
            <input type="number" min={0} value={x} onChange={(e) => setX(Number(e.target.value))} className="w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-2 py-1.5" />
          </label>
          <label>
            <span className="block text-slate-600 dark:text-slate-400 mb-1">Y</span>
            <input type="number" min={0} value={y} onChange={(e) => setY(Number(e.target.value))} className="w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-2 py-1.5" />
          </label>
          <label>
            <span className="block text-slate-600 dark:text-slate-400 mb-1">Width</span>
            <input type="number" min={1} value={width} onChange={(e) => setWidth(Number(e.target.value))} className="w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-2 py-1.5" />
          </label>
          <label>
            <span className="block text-slate-600 dark:text-slate-400 mb-1">Height</span>
            <input type="number" min={1} value={height} onChange={(e) => setHeight(Number(e.target.value))} className="w-full rounded border border-slate-300 dark:border-slate-700 bg-transparent px-2 py-1.5" />
          </label>
        </div>

        {error && <div className="text-xs text-rose-500">{error}</div>}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={() => {
              setX(0);
              setY(0);
              setWidth(maxW);
              setHeight(maxH);
              setRotateDeg(0);
              setFlipH(false);
              setFlipV(false);
            }}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-xs"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={handleApply}
            disabled={submitting}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold"
          >
            {submitting ? "Applying..." : "Apply Crop"}
          </button>
        </div>
      </div>
    </div>
  );
}
