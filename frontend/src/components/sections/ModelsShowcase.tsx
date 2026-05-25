import { motion } from "framer-motion";
import { Award, CheckCircle2, Cpu, Hourglass, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { CountUp } from "@/components/animations/CountUp";
import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { TiltCard } from "@/components/animations/TiltCard";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MODELS } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function ModelsShowcase() {
  const { t } = useTranslation("sections");
  const trained = MODELS.filter((m) => m.status === "trained");
  const best = (trained.length > 0 ? trained : MODELS).reduce((a, b) => (a.qwk > b.qwk ? a : b));

  return (
    <section id="models" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Models"
            title={t("models.title")}
            subtitle={t("models.subtitle")}
          />
        </FadeInOnScroll>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {MODELS.map((model, idx) => {
            const isBest = model.id === best.id;
            const isTrained = model.status === "trained";
            const isTraining = model.status === "training";
            return (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.5, delay: idx * 0.08, ease: [0.22, 1, 0.36, 1] }}
              >
                <TiltCard intensity={6} className="group h-full">
                  <Card
                    className={cn(
                      "relative h-full overflow-hidden p-5 transition-all duration-300",
                      "hover:shadow-2xl",
                      isBest
                        ? "border-amber-500/50 shadow-lg shadow-amber-500/10"
                        : "hover:border-primary/40 hover:shadow-primary/5",
                      !isTrained && !isTraining && "opacity-75",
                    )}
                    style={{ transform: "translateZ(0)" }}
                  >
                  <div className="absolute right-3 top-3 flex gap-1.5">
                    {isBest && (
                      <Badge variant="warning" className="gap-1">
                        <Award className="size-3" /> {t("models.bestModel")}
                      </Badge>
                    )}
                    {isTrained && !isBest && (
                      <Badge variant="success" className="gap-1">
                        <CheckCircle2 className="size-3" />
                      </Badge>
                    )}
                    {isTraining && (
                      <Badge variant="outline" className="gap-1 border-primary/40 text-primary">
                        <Loader2 className="size-3 animate-spin" />
                        training
                      </Badge>
                    )}
                  </div>

                  <div
                    className="absolute -right-12 -top-12 size-32 rounded-full opacity-10 blur-2xl transition-all duration-500 group-hover:opacity-25 group-hover:scale-110"
                    style={{ background: model.color }}
                  />

                  <CardContent className="relative space-y-4 p-0">
                    <div className="space-y-1">
                      <h3 className="text-xl font-bold">{model.name}</h3>
                      <p className="text-xs text-muted-foreground">{model.type.replace(/_/g, " ")}</p>
                    </div>

                    <p className="text-xs leading-relaxed text-muted-foreground line-clamp-3">
                      {model.description}
                    </p>

                    <div className="space-y-2.5">
                      <div className="flex items-baseline justify-between">
                        <span className="text-xs text-muted-foreground">
                          {t("models.metrics.qwk")}
                          {!isTrained && (
                            <span className="ml-1 text-[10px] opacity-70">
                              ({isTraining ? "CV" : "est."})
                            </span>
                          )}
                        </span>
                        <span className="text-2xl font-bold tabular-nums">
                          <CountUp to={model.qwk} decimals={4} duration={1.8} />
                        </span>
                      </div>
                      <Progress value={model.qwk * 100} />
                    </div>

                    <div className="grid grid-cols-2 gap-3 border-t border-border pt-3 text-xs">
                      <div>
                        <p className="text-muted-foreground">{t("models.metrics.undertriage")}</p>
                        <p className="mt-0.5 font-mono font-semibold tabular-nums">
                          {(model.undertriage * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">{t("models.metrics.criticalMiss")}</p>
                        <p className="mt-0.5 font-mono font-semibold tabular-nums">
                          {(model.criticalMiss * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between border-t border-border pt-3 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1">
                        <Cpu className="size-3" />
                        {model.device}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Hourglass className="size-3" />
                        {model.trainTimeMin}m
                      </span>
                    </div>
                  </CardContent>
                </Card>
                </TiltCard>
              </motion.div>
            );
          })}
        </div>

        <FadeInOnScroll delay={0.2}>
          <div className="space-y-3 text-center">
            <p className="text-sm text-muted-foreground">{t("models.tunedWith")}</p>
            <p className="text-xs text-muted-foreground/70">
              {t("models.disclaimer")}
            </p>
          </div>
        </FadeInOnScroll>
      </div>
    </section>
  );
}
