import type { FaceMeshResult } from "../types/api";

/** Per-face tracking id, head pose and expression indicators (observable movement only). */
export function FaceMeshResults({
  result,
  downloadUrl,
}: {
  result: FaceMeshResult | null | undefined;
  downloadUrl?: string;
}) {
  if (!result) return null;
  const isVideo = result.frames_analyzed !== undefined;
  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 space-y-3 text-xs">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Face mesh results</h3>
        {downloadUrl && (
          <a href={downloadUrl} download className="text-indigo-500 hover:underline">
            Download mesh JSON
          </a>
        )}
      </div>
      {isVideo ? (
        <div className="flex flex-wrap gap-4">
          <span>Frames analysed: <strong>{result.frames_analyzed}</strong></span>
          <span>Frames with faces: <strong>{result.frames_with_faces}</strong></span>
          <span>Face tracks: <strong>{result.unique_face_tracks}</strong></span>
          <span>Most faces at once: <strong>{result.max_faces_in_frame}</strong></span>
        </div>
      ) : result.faces.length === 0 ? (
        <p className="text-slate-600 dark:text-slate-400">No face was found for the mesh.</p>
      ) : (
        <div className="space-y-2">
          {result.faces.map((f) => (
            <div key={f.face_index} className="p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 space-y-1">
              <div className="font-semibold">
                Face #{f.tracking_id ?? f.face_index + 1}{" "}
                <span className="font-normal text-slate-500">· {f.landmark_count} landmarks</span>
              </div>
              {f.head_pose && (
                <div className="text-slate-600 dark:text-slate-400">
                  Head pose — yaw {f.head_pose.yaw_deg}°, pitch {f.head_pose.pitch_deg}°, roll {f.head_pose.roll_deg}°
                </div>
              )}
              {f.expression_indicators.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {f.expression_indicators.slice(0, 4).map((c) => (
                    <span key={c.name} className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                      {c.name} {c.score.toFixed(2)}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <p className="text-[11px] text-slate-500">
        {result.disclaimer ??
          "Movement indicators describe observable facial movement only, not emotion, intent or health."}
      </p>
    </div>
  );
}
