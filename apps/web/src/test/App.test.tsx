import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { DropZone } from "../components/DropZone";

describe("FaceVision Web App", () => {
  it("renders navigation and dashboard", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/health")) {
          return new Response(
            JSON.stringify({ status: "ok", app: "FaceVision", version: "2.1.0", execution_mode: "web_local_agent" }),
            { status: 200 }
          );
        }
        if (url.includes("/stats")) {
          return new Response(
            JSON.stringify({
              total_media: 3,
              total_jobs: 2,
              completed_jobs: 2,
              active_jobs: 0,
              failed_jobs: 0,
              total_faces_detected: 5,
              execution_mode: "web_local_agent",
              cloud_processing_enabled: false,
            }),
            { status: 200 }
          );
        }
        return new Response(JSON.stringify([]), { status: 200 });
      })
    );

    render(<App />);
    expect(screen.getByText("FaceVision")).toBeInTheDocument();
    expect(screen.getByText("Image Detection")).toBeInTheDocument();
    expect(screen.getByText("Video Detection")).toBeInTheDocument();
    expect(screen.getAllByText("Live Webcam").length).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("5")).toBeInTheDocument();
  });

  it("invokes onFilesSelected when files are chosen in DropZone", () => {
    const onSelect = vi.fn();
    const { container } = render(
      <DropZone accept="image/png" multiple allowFolder onFilesSelected={onSelect} />
    );
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["png"], "test.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(onSelect).toHaveBeenCalledWith([file]);
  });
});
