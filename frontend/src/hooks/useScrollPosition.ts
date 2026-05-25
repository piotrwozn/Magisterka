import { useEffect, useState } from "react";

interface ScrollState {
  y: number;
  progress: number;
  direction: "up" | "down" | null;
}

/**
 * Tracks scroll position with rAF throttling.
 * Returns y, progress (0-1), and direction.
 */
export function useScrollPosition(): ScrollState {
  const [state, setState] = useState<ScrollState>({ y: 0, progress: 0, direction: null });

  useEffect(() => {
    let lastY = window.scrollY;
    let ticking = false;

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const y = window.scrollY;
          const max = document.documentElement.scrollHeight - window.innerHeight;
          const progress = max > 0 ? Math.min(1, Math.max(0, y / max)) : 0;
          const direction: "up" | "down" | null = y > lastY ? "down" : y < lastY ? "up" : null;
          lastY = y;
          setState({ y, progress, direction });
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return state;
}
