import { motion } from "framer-motion";

import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  value: number;
  color?: string;
  size?: number;
  className?: string;
}

export function ConfidenceGauge({
  value,
  color = "hsl(var(--primary))",
  size = 140,
  className,
}: ConfidenceGaugeProps) {
  const reduced = useReducedMotion();
  const radius = size / 2 - 12;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(1, value));

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth="10"
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: reduced ? circumference * (1 - pct) : circumference * (1 - pct) }}
          transition={{ duration: reduced ? 0 : 1.4, ease: [0.22, 1, 0.36, 1] }}
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.4 }}
          className="text-3xl font-bold tabular-nums"
        >
          {Math.round(pct * 100)}%
        </motion.span>
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
          confidence
        </span>
      </div>
    </div>
  );
}
