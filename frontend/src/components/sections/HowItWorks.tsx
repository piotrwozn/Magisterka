import { motion } from "framer-motion";
import { Brain, Cpu, Edit3, FileJson, GitMerge, ShieldCheck } from "lucide-react";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";

const STEPS = [
  { key: "step1", icon: Edit3, color: "#06b6d4" },
  { key: "step2", icon: FileJson, color: "#0ea5e9" },
  { key: "step3", icon: Cpu, color: "#3b82f6" },
  { key: "step4", icon: Brain, color: "#8b5cf6" },
  { key: "step5", icon: GitMerge, color: "#a855f7" },
  { key: "step6", icon: ShieldCheck, color: "#10b981" },
];

export function HowItWorks() {
  const { t } = useTranslation("sections");

  return (
    <section id="how-it-works" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Pipeline"
            title={t("howItWorks.title")}
            subtitle={t("howItWorks.subtitle")}
          />
        </FadeInOnScroll>

        <div className="relative mx-auto max-w-3xl">
          <div className="absolute left-7 top-0 hidden h-full w-px bg-gradient-to-b from-primary via-purple-500 to-emerald-500 md:block" />

          <div className="space-y-6">
            {STEPS.map((step, idx) => {
              const Icon = step.icon;
              return (
                <motion.div
                  key={step.key}
                  initial={{ opacity: 0, x: -30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                  className="relative flex gap-5"
                >
                  <div className="relative z-10 hidden md:block">
                    <motion.div
                      whileHover={{ scale: 1.15 }}
                      className="relative flex size-14 items-center justify-center rounded-2xl border-2 border-border bg-background shadow-lg"
                      style={{ boxShadow: `0 0 30px -10px ${step.color}` }}
                    >
                      <Icon className="size-6" style={{ color: step.color }} />
                      <span
                        className="absolute -right-1 -top-1 flex size-5 items-center justify-center rounded-full text-[10px] font-bold text-background"
                        style={{ background: step.color }}
                      >
                        {idx + 1}
                      </span>
                    </motion.div>
                  </div>

                  <Card className="flex-1 p-5 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-xl">
                    <div className="flex items-start gap-4 md:hidden">
                      <Icon className="size-5 shrink-0" style={{ color: step.color }} />
                    </div>
                    <h3 className="text-lg font-bold">{t(`howItWorks.${step.key}.title`)}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                      {t(`howItWorks.${step.key}.description`)}
                    </p>
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
