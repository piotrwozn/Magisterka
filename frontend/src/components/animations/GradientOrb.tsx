import { motion } from "framer-motion";
import type { CSSProperties } from "react";

import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";

interface GradientOrbProps {
  color?: string;
  size?: number;
  className?: string;
  style?: CSSProperties;
  duration?: number;
}

export function GradientOrb({
  color = "hsl(var(--primary))",
  size = 400,
  className,
  style,
  duration = 18,
}: GradientOrbProps) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      aria-hidden
      className={cn("pointer-events-none absolute rounded-full blur-3xl opacity-40", className)}
      style={{
        width: size,
        height: size,
        background: `radial-gradient(circle at center, ${color} 0%, transparent 70%)`,
        ...style,
      }}
      animate={
        reduced
          ? undefined
          : {
              x: [0, 20, -20, 0],
              y: [0, -15, 15, 0],
              scale: [1, 1.1, 0.95, 1],
            }
      }
      transition={{ duration, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}
