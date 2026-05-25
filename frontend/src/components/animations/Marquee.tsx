import { type ReactNode } from "react";

import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";

interface MarqueeProps {
  children: ReactNode;
  speed?: number;
  className?: string;
  pauseOnHover?: boolean;
  reverse?: boolean;
}

/**
 * Infinite horizontal marquee — pure CSS keyframes for buttery 60fps.
 * Renders content twice so the loop seam is invisible.
 */
export function Marquee({
  children,
  speed = 30,
  className,
  pauseOnHover = true,
  reverse = false,
}: MarqueeProps) {
  const reduced = useReducedMotion();

  return (
    <div
      className={cn(
        "group relative flex overflow-hidden",
        "[--marquee-gap:3rem]",
        "[mask-image:linear-gradient(to_right,transparent,black_8%,black_92%,transparent)]",
        className,
      )}
    >
      <div
        className={cn(
          "flex shrink-0 items-center justify-around gap-[--marquee-gap] pr-[--marquee-gap]",
          !reduced && "animate-[marquee_var(--marquee-duration)_linear_infinite]",
          pauseOnHover && "group-hover:[animation-play-state:paused]",
        )}
        style={
          {
            "--marquee-duration": `${speed}s`,
            animationDirection: reverse ? "reverse" : "normal",
          } as React.CSSProperties
        }
      >
        {children}
      </div>
      <div
        aria-hidden
        className={cn(
          "flex shrink-0 items-center justify-around gap-[--marquee-gap] pr-[--marquee-gap]",
          !reduced && "animate-[marquee_var(--marquee-duration)_linear_infinite]",
          pauseOnHover && "group-hover:[animation-play-state:paused]",
        )}
        style={
          {
            "--marquee-duration": `${speed}s`,
            animationDirection: reverse ? "reverse" : "normal",
          } as React.CSSProperties
        }
      >
        {children}
      </div>
    </div>
  );
}
