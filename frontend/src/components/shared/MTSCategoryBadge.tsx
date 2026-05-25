import { MTS_CATEGORIES, type MTSCategoryId } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface MTSCategoryBadgeProps {
  category: MTSCategoryId | number;
  lang?: "pl" | "en";
  size?: "sm" | "md" | "lg";
  showWait?: boolean;
  className?: string;
}

export function MTSCategoryBadge({
  category,
  lang = "pl",
  size = "md",
  showWait = false,
  className,
}: MTSCategoryBadgeProps) {
  const cat = MTS_CATEGORIES[category];
  if (!cat) return null;

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
        backgroundColor: `${cat.color}1f`,
        color: cat.color,
        boxShadow: `inset 0 0 0 1px ${cat.color}40`,
      }}
    >
      <span
        className={cn("rounded-full", dotSizes[size])}
        style={{ backgroundColor: cat.color, boxShadow: `0 0 12px ${cat.color}` }}
      />
      <span>{lang === "pl" ? cat.labelPl : cat.labelEn}</span>
      {showWait && (
        <span className="text-[10px] font-normal opacity-70">· {cat.maxWait}</span>
      )}
    </div>
  );
}
