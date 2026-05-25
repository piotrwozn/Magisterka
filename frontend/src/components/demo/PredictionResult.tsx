import { motion } from "framer-motion";
import { Brain, Clock } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ConfidenceGauge } from "@/components/demo/ConfidenceGauge";
import { ConflictDetection } from "@/components/demo/ConflictDetection";
import { ModelComparison } from "@/components/demo/ModelComparison";
import { ShapWaterfall } from "@/components/demo/ShapWaterfall";
import { MTSCategoryBadge } from "@/components/shared/MTSCategoryBadge";
import { Card } from "@/components/ui/card";
import { MTS_CATEGORIES, type MTSCategoryId } from "@/lib/constants";
import type { PredictResponse } from "@/lib/types";

interface PredictionResultProps {
  result: PredictResponse;
}

export function PredictionResult({ result }: PredictionResultProps) {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";
  const cat = MTS_CATEGORIES[result.finalCategory as MTSCategoryId];

  return (
    <div className="space-y-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <Card
          className="relative overflow-hidden p-6"
          style={{ borderColor: cat ? `${cat.color}40` : undefined }}
        >
          {cat && (
            <div
              className="absolute -right-20 -top-20 size-64 rounded-full opacity-20 blur-3xl"
              style={{ background: cat.color }}
            />
          )}

          <div className="relative grid items-center gap-6 sm:grid-cols-[1fr_auto]">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">
                {t("demo.result.category")}
              </p>
              {cat && <MTSCategoryBadge category={cat.id} lang={lang} size="lg" showWait />}
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Clock className="size-3" />
                {t("demo.result.processingTime")}: {result.processingTimeMs}ms
              </p>
            </div>

            <ConfidenceGauge
              value={result.confidence}
              color={cat?.color ?? "hsl(var(--primary))"}
            />
          </div>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.4 }}
      >
        <ShapWaterfall values={result.shapTop5} />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25, duration: 0.4 }}
      >
        <ModelComparison predictions={result.modelPredictions} />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.4 }}
      >
        <Card className="p-5">
          <div className="flex items-start gap-3">
            <Brain className="mt-0.5 size-5 shrink-0 text-purple-500" />
            <div className="flex-1 space-y-2">
              <h4 className="text-sm font-bold">{t("demo.result.medgemmaTitle")}</h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {result.medgemma.reasoning}
              </p>
              {result.medgemma.riskFlags.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {result.medgemma.riskFlags.map((flag) => (
                    <span
                      key={flag}
                      className="rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive"
                    >
                      {flag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45, duration: 0.4 }}
      >
        <ConflictDetection conflict={result.conflict} />
      </motion.div>
    </div>
  );
}
