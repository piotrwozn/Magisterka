import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Activity } from "lucide-react";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { PatientForm } from "@/components/demo/PatientForm";
import { PredictionResult } from "@/components/demo/PredictionResult";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { usePredict } from "@/hooks/usePredict";
import { useDemoStore } from "@/stores/demoStore";

export function DemoSection() {
  const { t } = useTranslation("sections");
  const { result } = useDemoStore();
  const { isPending } = usePredict();

  return (
    <section id="demo" className="relative py-24 md:py-32">
      <div className="container space-y-12">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow={
              <span className="inline-flex items-center gap-1">
                <Sparkles className="size-3" /> Live demo
              </span>
            }
            title={t("demo.title")}
            subtitle={t("demo.subtitle")}
          />
        </FadeInOnScroll>

        <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          <FadeInOnScroll>
            <PatientForm />
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.1}>
            <div className="lg:sticky lg:top-24">
              <AnimatePresence mode="wait">
                {isPending ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="space-y-4"
                  >
                    <Skeleton className="h-32 rounded-xl" />
                    <Skeleton className="h-40 rounded-xl" />
                    <Skeleton className="h-32 rounded-xl" />
                  </motion.div>
                ) : result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <PredictionResult result={result} />
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card className="flex min-h-[400px] flex-col items-center justify-center gap-4 p-8 text-center">
                      <div className="rounded-2xl bg-primary/10 p-4 text-primary">
                        <Activity className="size-8" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold">{t("demo.emptyState.title")}</h3>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {t("demo.emptyState.description")}
                        </p>
                      </div>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </FadeInOnScroll>
        </div>
      </div>
    </section>
  );
}
