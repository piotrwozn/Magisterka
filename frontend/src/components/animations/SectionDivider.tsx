import { motion } from "framer-motion";

import { cn } from "@/lib/utils";

interface SectionDividerProps {
  className?: string;
  variant?: "line" | "dotted" | "gradient";
}

/**
 * Animated gradient divider with sliding shimmer.
 * Drawn-in line effect on scroll-into-view.
 */
export function SectionDivider({ className, variant = "gradient" }: SectionDividerProps) {
  return (
    <div className={cn("relative flex items-center justify-center py-2", className)} aria-hidden>
      <motion.div
        initial={{ scaleX: 0, opacity: 0 }}
        whileInView={{ scaleX: 1, opacity: 1 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1] }}
        className={cn(
          "h-px w-full origin-center",
          variant === "gradient" &&
            "bg-gradient-to-r from-transparent via-primary/60 to-transparent",
          variant === "line" && "bg-border",
          variant === "dotted" &&
            "border-t border-dashed border-border bg-transparent",
        )}
      />
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        whileInView={{ scale: 1, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.6, duration: 0.5 }}
        className="absolute size-2 rounded-full bg-primary shadow-[0_0_20px_4px_hsl(var(--primary)/0.6)]"
      />
    </div>
  );
}
