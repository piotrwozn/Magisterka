import { motion } from "framer-motion";

import { useScrollPosition } from "@/hooks/useScrollPosition";

export function ScrollProgress() {
  const { progress } = useScrollPosition();
  return (
    <motion.div
      className="fixed inset-x-0 top-0 z-[100] h-[2px] origin-left bg-gradient-to-r from-primary via-fuchsia-500 to-purple-500"
      style={{ scaleX: progress }}
      aria-hidden
    />
  );
}
