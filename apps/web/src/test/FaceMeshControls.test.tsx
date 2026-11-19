import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FaceMeshControls } from "../components/FaceMeshControls";
import { DEFAULT_FACE_MESH } from "../lib/faceMesh";
import type { FaceMeshSettings } from "../types/api";

function mockStatus(available: boolean) {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            available,
            model: "face_landmarker.task",
            license: "Apache-2.0",
            landmarks: 478,
            offline: true,
            checksum_verified: true,
            unavailable_reason: available ? null : "Place face_landmarker.task in models/.",
            disclaimer: "",
          }),
          { status: 200 }
        )
    )
  );
}

function Harness({ onSpy }: { onSpy?: (s: FaceMeshSettings) => void }) {
  const [value, setValue] = useState<FaceMeshSettings>(DEFAULT_FACE_MESH);
  return (
    <QueryClientProvider client={new QueryClient()}>
      <FaceMeshControls
        value={value}
        advanced
        onChange={(next) => {
          setValue(next);
          onSpy?.(next);
        }}
      />
    </QueryClientProvider>
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("FaceMeshControls", () => {
  it("shows style controls only after the mesh is enabled and applies changes", async () => {
    mockStatus(true);
    const spy = vi.fn();
    render(<Harness onSpy={spy} />);

    expect(screen.queryByText("Mesh color")).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Enable face mesh"));
    expect(await screen.findByText("Mesh color")).toBeInTheDocument();
    expect(spy).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: true }));

    fireEvent.click(screen.getByRole("button", { name: "Landmark dots" }));
    expect(spy).toHaveBeenLastCalledWith(
      expect.objectContaining({ show_landmarks: true, show_tessellation: false })
    );

    fireEvent.click(screen.getByLabelText("Lips"));
    expect(spy).toHaveBeenLastCalledWith(expect.objectContaining({ show_lips: true }));
  });

  it("disables the mesh and explains why when the model is unavailable", async () => {
    mockStatus(false);
    render(<Harness />);
    expect(await screen.findByText("Place face_landmarker.task in models/.")).toBeInTheDocument();
    expect(screen.getByLabelText("Enable face mesh")).toBeDisabled();
  });
});
