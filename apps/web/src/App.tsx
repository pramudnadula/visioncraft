import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { BatchPage } from "./pages/BatchPage";
import { DashboardPage } from "./pages/DashboardPage";
import { HistoryPage } from "./pages/HistoryPage";
import { ImageStudioPage } from "./pages/ImageStudioPage";
import { MediaStudioPage } from "./pages/MediaStudioPage";
import { ModelsPage } from "./pages/ModelsPage";
import { ReportsPage } from "./pages/ReportsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { VideoStudioPage } from "./pages/VideoStudioPage";
import { VisionStudioPage } from "./pages/VisionStudioPage";
import { WebcamPage } from "./pages/WebcamPage";

const queryClient = new QueryClient();

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/images" element={<ImageStudioPage />} />
            <Route path="/videos" element={<VideoStudioPage />} />
            <Route path="/webcam" element={<WebcamPage />} />
            <Route path="/vision" element={<VisionStudioPage />} />
            <Route path="/media-studio" element={<MediaStudioPage />} />
            <Route path="/batch" element={<BatchPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
