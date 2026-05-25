import { useEffect, useRef } from "react";

import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";

interface AuroraBackgroundProps {
  className?: string;
  intensity?: number;
}

/**
 * Animated mesh-gradient aurora background.
 * Three slowly-orbiting radial gradients blurred together.
 * Pure CSS animation — no JS per frame.
 */
export function AuroraBackground({ className, intensity = 0.6 }: AuroraBackgroundProps) {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;

    let raf = 0;
    const start = performance.now();

    const loop = () => {
      const t = (performance.now() - start) / 1000;
      const x1 = 50 + Math.sin(t * 0.18) * 30;
      const y1 = 30 + Math.cos(t * 0.22) * 20;
      const x2 = 70 + Math.sin(t * 0.15 + 2) * 25;
      const y2 = 60 + Math.cos(t * 0.19 + 1) * 25;
      const x3 = 30 + Math.cos(t * 0.13 + 4) * 28;
      const y3 = 70 + Math.sin(t * 0.21 + 3) * 22;
      el.style.setProperty("--aurora-x1", `${x1}%`);
      el.style.setProperty("--aurora-y1", `${y1}%`);
      el.style.setProperty("--aurora-x2", `${x2}%`);
      el.style.setProperty("--aurora-y2", `${y2}%`);
      el.style.setProperty("--aurora-x3", `${x3}%`);
      el.style.setProperty("--aurora-y3", `${y3}%`);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [reduced]);

  return (
    <div
      ref={ref}
      aria-hidden
      className={cn("pointer-events-none absolute inset-0 -z-10 overflow-hidden", className)}
      style={
        {
          "--aurora-x1": "30%",
          "--aurora-y1": "30%",
          "--aurora-x2": "70%",
          "--aurora-y2": "60%",
          "--aurora-x3": "50%",
          "--aurora-y3": "80%",
          opacity: intensity,
        } as React.CSSProperties
      }
    >
      <div
        className="absolute inset-0 transition-[background] duration-[1500ms]"
        style={{
          background: `
            radial-gradient(ellipse 50% 40% at var(--aurora-x1) var(--aurora-y1), hsl(199 89% 58% / 0.55), transparent 70%),
            radial-gradient(ellipse 45% 35% at var(--aurora-x2) var(--aurora-y2), hsl(280 80% 60% / 0.50), transparent 70%),
            radial-gradient(ellipse 40% 30% at var(--aurora-x3) var(--aurora-y3), hsl(320 75% 60% / 0.40), transparent 70%)
          `,
          filter: "blur(60px) saturate(1.2)",
        }}
      />
    </div>
  );
}
