import type { TooltipProps } from "recharts";
import type { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

interface ChartTooltipProps extends Omit<TooltipProps<ValueType, NameType>, "formatter"> {
  formatter?: (value: number | string) => string;
  suffix?: string;
}

/**
 * Theme-aware tooltip for Recharts.
 * Uses Tailwind classes that respond to dark/light mode.
 */
export function ChartTooltip({ active, payload, label, formatter, suffix = "" }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-lg border border-border bg-popover/95 px-3 py-2 text-xs text-popover-foreground shadow-lg backdrop-blur-md">
      {label !== undefined && label !== "" && (
        <p className="mb-1 font-medium">{label}</p>
      )}
      <ul className="space-y-0.5">
        {payload.map((entry, idx) => {
          const raw = entry.value;
          const display =
            formatter && typeof raw === "number"
              ? formatter(raw)
              : typeof raw === "number"
                ? raw.toLocaleString()
                : String(raw ?? "");
          return (
            <li key={idx} className="flex items-center gap-2 tabular-nums">
              {entry.color && (
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ background: entry.color }}
                  aria-hidden
                />
              )}
              {entry.name && <span className="text-muted-foreground">{entry.name}:</span>}
              <span className="font-semibold">
                {display}
                {suffix}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
