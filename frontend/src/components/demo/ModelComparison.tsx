import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";

import { MTSCategoryBadge } from "@/components/shared/MTSCategoryBadge";
import { Card } from "@/components/ui/card";
import type { MTSCategoryId } from "@/lib/constants";
import type { ModelPrediction } from "@/lib/types";

interface ModelComparisonProps {
  predictions: ModelPrediction[];
}

export function ModelComparison({ predictions }: ModelComparisonProps) {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";

  return (
    <Card className="p-5">
      <h4 className="mb-4 text-sm font-bold">{t("demo.result.modelsTitle")}</h4>
      <ul className="space-y-2.5">
        {predictions.map((p, idx) => (
          <motion.li
            key={p.modelName}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.08, duration: 0.3 }}
            className="flex items-center justify-between gap-3 rounded-md border border-border/60 p-2.5"
          >
            <div className="flex min-w-0 items-center gap-2">
              <span className="font-mono text-xs font-semibold capitalize">{p.modelName}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {(p.confidence * 100).toFixed(0)}%
              </span>
              <MTSCategoryBadge
                category={p.category as MTSCategoryId}
                lang={lang}
                size="sm"
              />
            </div>
          </motion.li>
        ))}
      </ul>
    </Card>
  );
}
