import { CountUp } from "@/components/animations/CountUp";

interface MetricValueProps {
  /** undefined / null = result not available yet (shown as "—"). */
  value: number | undefined | null;
  decimals?: number;
  suffix?: string;
  duration?: number;
}

/** Animated metric, or "—" while the model / ensemble has no results yet. */
export function MetricValue({ value, decimals = 0, suffix = "", duration = 2 }: MetricValueProps) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return <span className="text-muted-foreground">—</span>;
  }
  return <CountUp to={value} decimals={decimals} suffix={suffix} duration={duration} />;
}
