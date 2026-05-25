import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { MouseSpotlight } from "@/components/animations/MouseSpotlight";
import { NoiseOverlay } from "@/components/animations/NoiseOverlay";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { ScrollProgress } from "@/components/layout/ScrollProgress";

const Home = lazy(() => import("@/pages/Home").then((m) => ({ default: m.Home })));
const Demo = lazy(() => import("@/pages/Demo").then((m) => ({ default: m.Demo })));
const Models = lazy(() => import("@/pages/Models").then((m) => ({ default: m.Models })));
const About = lazy(() => import("@/pages/About").then((m) => ({ default: m.About })));
const NotFound = lazy(() => import("@/pages/NotFound").then((m) => ({ default: m.NotFound })));

function PageLoader() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center" role="status" aria-live="polite">
      <div className="size-12 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    </div>
  );
}

export function App() {
  return (
    <div className="relative flex min-h-screen flex-col">
      <NoiseOverlay opacity={0.03} />
      <MouseSpotlight />
      <ScrollProgress />
      <Navbar />
      <main className="relative flex-1">
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/demo" element={<Demo />} />
            <Route path="/models" element={<Models />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
