import { AnimatePresence, motion } from "framer-motion";
import { Activity, Brain, Dice5, Loader2, Send } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ConfidenceGauge } from "@/components/demo/ConfidenceGauge";
import { ConflictDetection } from "@/components/demo/ConflictDetection";
import { ModelComparison } from "@/components/demo/ModelComparison";
import { ShapWaterfall } from "@/components/demo/ShapWaterfall";
import { VitalsInput } from "@/components/demo/VitalsInput";
import { ClinicalNoteInput } from "@/components/demo/ClinicalNoteInput";
import { MTSCategoryBadge } from "@/components/shared/MTSCategoryBadge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { usePredict } from "@/hooks/usePredict";
import { MTS_CATEGORIES, type MTSCategoryId } from "@/lib/constants";
import { EXAMPLE_PATIENTS } from "@/lib/mockData";
import { useDemoStore } from "@/stores/demoStore";

export function JourneyDemo() {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";
  const { vitals, clinicalNote, result, setResult, reset, loadExample } = useDemoStore();
  const { mutate, isPending } = usePredict();

  const handleSubmit = () => mutate({ vitals, clinicalNote }, { onSuccess: setResult });
  const handleRandom = () => {
    const ex = EXAMPLE_PATIENTS[Math.floor(Math.random() * EXAMPLE_PATIENTS.length)]!;
    loadExample({ ...ex.vitals }, ex.clinicalNote);
  };

  const cat = result ? MTS_CATEGORIES[result.finalCategory as MTSCategoryId] : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
      {/* ── LEWA — formularz ── */}
      <Card className="bg-card/70 p-5 backdrop-blur-md">
        <h4 className="mb-3 text-xs font-bold uppercase tracking-[0.2em] text-primary">
          {t("demo.form.vitalsTitle")}
        </h4>
        <VitalsInput />

        <h4 className="mb-2 mt-4 text-xs font-bold uppercase tracking-[0.2em] text-primary">
          {t("demo.form.noteTitle")}
        </h4>
        <ClinicalNoteInput />

        <div className="mt-3 flex flex-wrap gap-1.5">
          {EXAMPLE_PATIENTS.slice(0, 4).map((ex) => (
            <Button
              key={ex.nameKey}
              variant="outline"
              size="sm"
              onClick={() => loadExample({ ...ex.vitals }, ex.clinicalNote)}
              className="h-7 text-[10px]"
            >
              {t(ex.nameKey)}
            </Button>
          ))}
        </div>

        <div className="mt-4 flex gap-2">
          <Button onClick={handleSubmit} disabled={isPending} variant="glow" className="flex-1">
            {isPending ? (
              <>
                <Loader2 className="size-3.5 animate-spin" /> {t("demo.form.loading")}
              </>
            ) : (
              <>
                <Send className="size-3.5" /> {t("demo.form.predict")}
              </>
            )}
          </Button>
          <Button onClick={handleRandom} variant="outline" size="icon" aria-label="Random">
            <Dice5 className="size-4" />
          </Button>
          <Button onClick={reset} variant="ghost" size="sm">
            {t("demo.form.reset")}
          </Button>
        </div>
      </Card>

      {/* ── PRAWA — wyniki ── */}
      <div className="max-h-[70vh] overflow-y-auto pr-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
        <AnimatePresence mode="wait">
          {isPending ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              <Skeleton className="h-32" />
              <Skeleton className="h-24" />
              <Skeleton className="h-24" />
              <Skeleton className="h-32" />
            </motion.div>
          ) : result && cat ? (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-3"
            >
              {/* Kategoria + pewność */}
              <Card
                className="bg-card/70 p-5 backdrop-blur-md"
                style={{ borderColor: `${cat.color}40` }}
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="space-y-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      {t("demo.result.category")}
                    </p>
                    <MTSCategoryBadge category={cat.id} lang={lang} size="lg" showWait />
                    <p className="text-[10px] text-muted-foreground">
                      {t("demo.result.processingTime")}: {result.processingTimeMs}ms
                    </p>
                  </div>
                  <ConfidenceGauge value={result.confidence} color={cat.color} size={110} />
                </div>
              </Card>

              {/* Wykrywanie konfliktu + eskalacja */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <ConflictDetection conflict={result.conflict} />
              </motion.div>

              {/* Predykcje per model */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="[&>div]:bg-card/70 [&>div]:backdrop-blur-md"
              >
                <ModelComparison predictions={result.modelPredictions} />
              </motion.div>

              {/* SHAP top 5 */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="[&>div]:bg-card/70 [&>div]:backdrop-blur-md"
              >
                <ShapWaterfall values={result.shapTop5} />
              </motion.div>

              {/* MedGemma z risk flags */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="bg-card/70 p-4 backdrop-blur-md">
                  <div className="flex items-start gap-3">
                    <Brain className="mt-0.5 size-4 shrink-0 text-purple-400" />
                    <div className="flex-1 space-y-1.5">
                      <p className="text-xs font-semibold text-purple-400">
                        {t("demo.result.medgemmaTitle")}
                      </p>
                      <p className="text-sm leading-relaxed">{result.medgemma.reasoning}</p>
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
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Card className="flex h-full min-h-[280px] flex-col items-center justify-center bg-card/70 p-6 text-center backdrop-blur-md">
                <Activity className="mb-3 size-8 text-primary opacity-50" />
                <h3 className="text-sm font-bold">{t("demo.emptyState.title")}</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  {t("demo.emptyState.description")}
                </p>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
