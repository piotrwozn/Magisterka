import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import { TIMELINE_EVENTS } from "@/lib/constants";

export function JourneyTimeline() {
  const { t } = useTranslation("sections");

  return (
    <Card className="relative max-h-[70vh] overflow-y-auto bg-card/70 p-6 backdrop-blur-md">
      <div className="relative">
        <div className="pointer-events-none absolute left-4 top-1 bottom-1 w-px -translate-x-1/2 bg-gradient-to-b from-primary via-purple-500 to-emerald-500" />
        <ul className="space-y-3">
          {TIMELINE_EVENTS.map((event, idx) => (
            <motion.li
              key={`${event.date}-${idx}`}
              initial={{ opacity: 0, x: -12 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: idx * 0.04 }}
              className="relative flex items-center gap-4 pl-10"
            >
              <span className="absolute left-4 top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))] ring-4 ring-background" />
              <span className="w-16 shrink-0 font-mono text-[10px] uppercase tracking-wider text-primary">
                {event.date}
              </span>
              <span className="text-xs leading-relaxed">{t(event.titleKey)}</span>
            </motion.li>
          ))}
        </ul>
      </div>
    </Card>
  );
}
