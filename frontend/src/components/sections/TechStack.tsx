import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { TiltCard } from "@/components/animations/TiltCard";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { TECH_STACK } from "@/lib/constants";

export function TechStack() {
  const { t } = useTranslation("sections");

  const groups: { key: keyof typeof TECH_STACK; titleKey: string }[] = [
    { key: "frontend", titleKey: "techStack.frontend" },
    { key: "backend", titleKey: "techStack.backend" },
    { key: "ml", titleKey: "techStack.ml" },
    { key: "llm", titleKey: "techStack.llm" },
  ];

  return (
    <section id="tech-stack" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Stack"
            title={t("techStack.title")}
            subtitle={t("techStack.subtitle")}
          />
        </FadeInOnScroll>

        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          {groups.map((group, gi) => (
            <FadeInOnScroll key={group.key} delay={gi * 0.1}>
              <TiltCard intensity={5} className="group h-full">
                <Card className="h-full p-6 transition-all hover:border-primary/40 hover:shadow-xl hover:shadow-primary/5">
                  <h3 className="mb-5 text-xs font-bold uppercase tracking-[0.2em] text-primary">
                    {t(group.titleKey)}
                  </h3>
                  <ul className="space-y-3">
                    {(TECH_STACK[group.key] ?? []).map((tech, ti) => (
                      <motion.li
                        key={tech.name}
                        initial={{ opacity: 0, x: -10 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: gi * 0.1 + ti * 0.05, duration: 0.4 }}
                        className="group/item flex items-center gap-3 text-sm transition-all"
                      >
                        <span
                          className="relative size-1.5 shrink-0 rounded-full transition-all group-hover/item:scale-150"
                          style={{
                            backgroundColor: tech.color,
                            boxShadow: `0 0 12px ${tech.color}80`,
                          }}
                          aria-hidden
                        />
                        <span className="text-foreground/85 transition-colors group-hover/item:text-foreground">
                          {tech.name}
                        </span>
                      </motion.li>
                    ))}
                  </ul>
                </Card>
              </TiltCard>
            </FadeInOnScroll>
          ))}
        </div>
      </div>
    </section>
  );
}
