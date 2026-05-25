import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { TIMELINE_EVENTS } from "@/lib/constants";

export function Timeline() {
  const { t } = useTranslation("sections");

  return (
    <section id="timeline" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Timeline"
            title={t("timeline.title")}
            subtitle={t("timeline.subtitle")}
          />
        </FadeInOnScroll>

        <div className="relative mx-auto max-w-3xl">
          <div className="absolute left-4 top-0 h-full w-px bg-gradient-to-b from-primary via-purple-500 to-emerald-500 md:left-1/2 md:-translate-x-1/2" />

          <div className="space-y-8">
            {TIMELINE_EVENTS.map((event, idx) => {
              const isLeft = idx % 2 === 0;
              return (
                <motion.div
                  key={`${event.date}-${idx}`}
                  initial={{ opacity: 0, x: isLeft ? -30 : 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: idx * 0.05 }}
                  className={`relative flex gap-4 md:gap-8 ${isLeft ? "md:flex-row" : "md:flex-row-reverse"}`}
                >
                  <div className="relative z-10 shrink-0">
                    <motion.div
                      whileHover={{ scale: 1.3 }}
                      className="relative ml-0.5 mt-3 size-7 rounded-full border-2 border-primary bg-background md:ml-[calc(50%-14px)]"
                    >
                      <div className="absolute inset-1 rounded-full bg-primary opacity-60" />
                    </motion.div>
                  </div>

                  <Card
                    className={`flex-1 p-4 md:p-5 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-xl md:max-w-[calc(50%-2rem)] ${isLeft ? "md:mr-auto" : "md:ml-auto"}`}
                  >
                    <p className="font-mono text-xs text-primary">{event.date}</p>
                    <h3 className="mt-1 text-sm font-semibold leading-snug">
                      {t(event.titleKey)}
                    </h3>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
