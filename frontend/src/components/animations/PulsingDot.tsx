import { cn } from "@/lib/utils";

interface PulsingDotProps {
  color?: string;
  size?: number;
  className?: string;
}

export function PulsingDot({ color = "currentColor", size = 8, className }: PulsingDotProps) {
  return (
    <span className={cn("relative inline-flex", className)} style={{ width: size, height: size }}>
      <span
        className="absolute inset-0 animate-ping rounded-full opacity-75"
        style={{ backgroundColor: color }}
      />
      <span
        className="relative inline-flex rounded-full"
        style={{ width: size, height: size, backgroundColor: color }}
      />
    </span>
  );
}
