import { useEffect, useState } from "react";

import { useReducedMotion } from "@/hooks/useReducedMotion";

interface TypewriterTextProps {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
  showCursor?: boolean;
  onComplete?: () => void;
}

export function TypewriterText({
  text,
  speed = 35,
  delay = 0,
  className,
  showCursor = true,
  onComplete,
}: TypewriterTextProps) {
  const reduced = useReducedMotion();
  const [displayed, setDisplayed] = useState(reduced ? text : "");
  const [done, setDone] = useState(reduced);

  useEffect(() => {
    if (reduced) {
      setDisplayed(text);
      setDone(true);
      onComplete?.();
      return;
    }

    setDisplayed("");
    setDone(false);

    let idx = 0;
    const timeoutId = window.setTimeout(() => {
      const intervalId = window.setInterval(() => {
        idx++;
        setDisplayed(text.slice(0, idx));
        if (idx >= text.length) {
          window.clearInterval(intervalId);
          setDone(true);
          onComplete?.();
        }
      }, speed);

      return () => window.clearInterval(intervalId);
    }, delay);

    return () => window.clearTimeout(timeoutId);
  }, [text, speed, delay, reduced, onComplete]);

  return (
    <span className={className}>
      {displayed}
      {showCursor && !done && (
        <span className="ml-0.5 inline-block h-[0.9em] w-[2px] -translate-y-[2px] animate-pulse bg-primary align-middle" />
      )}
    </span>
  );
}
