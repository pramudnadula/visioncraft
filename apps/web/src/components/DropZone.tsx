import { useRef, useState } from "react";
import { FolderOpen, UploadCloud } from "lucide-react";

interface DropZoneProps {
  accept: string;
  multiple?: boolean;
  allowFolder?: boolean;
  label?: string;
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export function DropZone({
  accept,
  multiple = false,
  allowFolder = false,
  label = "Drag and drop media here, or click to browse",
  onFilesSelected,
  disabled = false,
}: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    onFilesSelected(Array.from(fileList));
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors ${
        dragging
          ? "border-indigo-500 bg-indigo-500/10"
          : "border-slate-300 dark:border-slate-700 bg-white/50 dark:bg-slate-900/40"
      } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
    >
      <UploadCloud className="h-8 w-8 mx-auto text-indigo-500 mb-2" />
      <p className="text-sm font-medium">{label}</p>
      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
        Accepted: {accept}
      </p>

      <div className="mt-4 flex items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium"
        >
          Select {multiple ? "Files" : "File"}
        </button>

        {allowFolder && (
          <button
            type="button"
            onClick={() => folderRef.current?.click()}
            className="px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-sm font-medium inline-flex items-center gap-2"
          >
            <FolderOpen className="h-4 w-4" />
            Import Folder
          </button>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {allowFolder && (
        <input
          ref={folderRef}
          type="file"
          multiple
          // @ts-expect-error webkitdirectory is supported in Chromium/WebKit
          webkitdirectory=""
          className="hidden"
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
      )}
    </div>
  );
}
