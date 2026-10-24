import { ESI_LEVELS, type EsiLevelId } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface ESILevelBadgeProps {
  /** Class index from the API: 0–4 = ESI 1–5. */
  category: EsiLevelId | number;
  lang?: "pl" | "en";
  size?: "sm" | "md" | "lg";
  showWait?: boolean;
  className?: string;
}

/** ESI level chip in its TOPSOR colour, e.g. "ESI 2 · Pomarańczowy · 10 min". */
export function ESILevelBadge({
  category,
  lang = "pl",
  size = "md",
  showWait = false,
  className,
}: ESILevelBadgeProps) {
  const lvl = ESI_LEVELS[category];
  if (!lvl) return null;

  const sizes = {
    sm: "px-2 py-0.5 text-xs gap-1",
    md: "px-3 py-1 text-sm gap-1.5",
    lg: "px-4 py-1.5 text-base gap-2",
  };

  const dotSizes = { sm: "size-1.5", md: "size-2", lg: "size-2.5" };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full font-semibold ring-1 ring-inset",
        sizes[size],
        className,
      )}
      style={{
        backgroundColor: `${lvl.color}1f`,
        color: lvl.color,
        boxShadow: `inset 0 0 0 1px ${lvl.color}40`,
      }}
    >
      <span
        className={cn("rounded-full", dotSizes[size])}
        style={{ backgroundColor: lvl.color, boxShadow: `0 0 12px ${lvl.color}` }}
      />
      <span>ESI {lvl.level}</span>
      <span className="font-normal opacity-80">· {lang === "pl" ? lvl.colorPl : lvl.colorEn}</span>
      {showWait && (
        <span className="text-[10px] font-normal opacity-70">· {lvl.maxWait}</span>
      )}
    </div>
  );
}
