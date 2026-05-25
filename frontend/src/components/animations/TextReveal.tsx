import { motion, useReducedMotion as useFmReducedMotion } from "framer-motion";
import { Fragment } from "react";

import { cn } from "@/lib/utils";

interface TextRevealProps {
  text: string;
  className?: string;
  delay?: number;
  staggerWords?: number;
  splitBy?: "word" | "char";
  as?: keyof JSX.IntrinsicElements;
}

/**
 * Cinematic stagger reveal — words or characters.
 * Combines y-translate + blur fade for that filmic feel.
 */
export function TextReveal({
  text,
  className,
  delay = 0,
  staggerWords = 0.045,
  splitBy = "word",
  as: Component = "span",
}: TextRevealProps) {
  const reduced = useFmReducedMotion();
  const tokens = splitBy === "word" ? text.split(" ") : text.split("");
  const sep = splitBy === "word" ? "\u00A0" : "";

  if (reduced) {
    const Wrapper = Component as React.ElementType;
    return <Wrapper className={className}>{text}</Wrapper>;
  }

  const Wrapper = Component as React.ElementType;

  return (
    <Wrapper className={cn("inline-block", className)} aria-label={text}>
      {tokens.map((token, i) => (
        <Fragment key={i}>
          <motion.span
            aria-hidden
            className="inline-block whitespace-pre [will-change:transform,opacity,filter]"
            initial={{ opacity: 0, y: "0.6em", filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{
              duration: 0.7,
              delay: delay + i * staggerWords,
              ease: [0.16, 1, 0.3, 1],
            }}
          >
            {token}
          </motion.span>
          {sep && <span aria-hidden>{sep}</span>}
        </Fragment>
      ))}
    </Wrapper>
  );
}
