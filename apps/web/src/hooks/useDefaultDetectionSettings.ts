import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { loadLocalPreferences } from "../lib/storage";
import type { BlurMode } from "../types/api";

/**
 * Shared hook initializing model, blur mode, and confidence threshold from saved Settings
 * (localStorage + backend `/settings`), while allowing per-page overrides without clobbering.
 */
export function useDefaultDetectionSettings() {
  const initial = loadLocalPreferences();
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });

  const [modelId, setModelIdState] = useState<string>(initial.defaultModelId);
  const [blurMode, setBlurModeState] = useState<BlurMode>(initial.defaultBlurMode);
  const [confidence, setConfidenceState] = useState<number>(initial.confidence);

  const dirtyRef = useRef({ model: false, blur: false, confidence: false });

  useEffect(() => {
    if (!settings) return;
    if (!dirtyRef.current.model) {
      setModelIdState(settings.default_model_id);
    }
    if (!dirtyRef.current.blur) {
      setBlurModeState(settings.default_blur_mode);
    }
    if (!dirtyRef.current.confidence) {
      setConfidenceState(settings.default_confidence_threshold);
    }
  }, [settings]);

  const setModelId = (value: string) => {
    dirtyRef.current.model = true;
    setModelIdState(value);
  };
  const setBlurMode = (value: BlurMode) => {
    dirtyRef.current.blur = true;
    setBlurModeState(value);
  };
  const setConfidence = (value: number) => {
    dirtyRef.current.confidence = true;
    setConfidenceState(value);
  };

  return { settings, modelId, setModelId, blurMode, setBlurMode, confidence, setConfidence };
}
