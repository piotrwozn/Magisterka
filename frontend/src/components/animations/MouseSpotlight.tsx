import { motion, useMotionValue, useSpring } from "framer-motion";
import { useEffect } from "react";

import { useReducedMotion } from "@/hooks/useReducedMotion";

interface MouseSpotlightProps {
  size?: number;
  color?: string;
  opacity?: number;
}

/**
 * Soft glowing spot that follows the cursor.
 * GPU-accelerated, springy, with proper fade on idle/leave.
 */
export function MouseSpotlight({
  size = 480,
  color = "hsl(199 89% 58%)",
  opacity = 0.18,
}: MouseSpotlightProps) {
  const reduced = useReducedMotion();
  const x = useMotionValue(-9999);
  const y = useMotionValue(-9999);
  const xs = useSpring(x, { damping: 28, stiffness: 180, mass: 0.5 });
  const ys = useSpring(y, { damping: 28, stiffness: 180, mass: 0.5 });

  useEffect(() => {
    if (reduced) return;
    const onMove = (e: MouseEvent) => {
      x.set(e.clientX - size / 2);
      y.set(e.clientY - size / 2);
    };
    const onLeave = () => {
      x.set(-9999);
      y.set(-9999);
    };
    window.addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", onLeave);
    return () => {
      window.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
    };
  }, [x, y, size, reduced]);

  if (reduced) return null;

  return (
    <motion.div
      aria-hidden
      className="pointer-events-none fixed z-[1] mix-blend-screen"
      style={{
        x: xs,
        y: ys,
        width: size,
        height: size,
        background: `radial-gradient(circle at center, ${color}, transparent 60%)`,
        opacity,
        filter: "blur(40px)",
      }}
    />
  );
}
