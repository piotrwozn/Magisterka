import { motion, useReducedMotion as useFmReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

interface FadeInOnScrollProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  y?: number;
  className?: string;
  once?: boolean;
  amount?: number;
}

export function FadeInOnScroll({
  children,
  delay = 0,
  duration = 0.6,
  y = 24,
  className,
  once = true,
  amount = 0.2,
}: FadeInOnScrollProps) {
  const reduced = useFmReducedMotion();

  const variants: Variants = {
    hidden: { opacity: 0, y: reduced ? 0 : y },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: reduced ? 0 : duration, delay, ease: [0.22, 1, 0.36, 1] },
    },
  };

  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="visible"
      viewport={{ once, amount }}
      variants={variants}
    >
      {children}
    </motion.div>
  );
}
