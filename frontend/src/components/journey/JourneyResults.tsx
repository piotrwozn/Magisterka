import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertOctagon, Award, Info, ShieldCheck, Sparkles, TrendingUp, X } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { ModelDetailModal } from "@/components/journey/ModelDetailModal";
import { MetricValue } from "@/components/shared/MetricValue";
import { Card } from "@/components/ui/card";
import { ESI_LEVELS, MODELS, RESULTS, type Model } from "@/lib/constants";
import { formatMetric } from "@/lib/utils";

// ── Ensemble description (backend/app/models/ensemble.py) ────────────────

const ENSEMBLE_DETAILS = {
  pl: [
    "Ważona średnia prawdopodobieństw ESI 1–5 z 6 modeli (domyślnie równe wagi).",
    "Wagi i reguła decyzyjna dobierane WYŁĄCZNIE na zbiorze walidacyjnym, raport na teście.",
    "Reguła „cost”: decyzja Bayesa z macierzą kosztów — niedotriażowanie 3× droższe.",
    "Wagi wkleja się do backend/models/ensemble_weights.json.",
    "Reguły bezpieczeństwa ESI mogą tylko podnieść pilność (punkt A → ESI 1, B/D → min. ESI 2).",
  ],
  en: [
    "Weighted average of ESI 1–5 probabilities from 6 models (equal weights by default).",
    "Weights and the decision rule are chosen ONLY on the validation split, reported on test.",
    "“cost” rule: Bayes decision with a cost matrix — undertriage costs 3× more.",
    "Weights are pasted into backend/models/ensemble_weights.json.",
    "ESI safety rules can only raise the urgency (point A → ESI 1, B/D → at least ESI 2).",
  ],
};

export function JourneyResults() {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";

  const [ensembleOpen, setEnsembleOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);

  const kpis = [
    {
      Icon: TrendingUp,
      label: t("results.kpis.qwk"),
      value: <MetricValue value={RESULTS?.qwk} decimals={4} />,
      desc: t("results.kpis.qwkDesc"),
      color: "hsl(199 89% 58%)",
    },
    {
      Icon: AlertOctagon,
      label: t("results.kpis.undertriage"),
      value: <MetricValue value={RESULTS?.undertriage} decimals={1} suffix="%" />,
      desc: t("results.kpis.undertriageDesc"),
      color: "hsl(40 96% 56%)",
    },
    {
      Icon: ShieldCheck,
      label: t("results.kpis.criticalMiss"),
      value: <MetricValue value={RESULTS?.criticalMiss} decimals={2} suffix="%" />,
      desc: t("results.kpis.criticalMissDesc"),
      color: "hsl(160 70% 50%)",
    },
    {
      Icon: Activity,
      label: t("results.kpis.overtriage"),
      value: <MetricValue value={RESULTS?.overtriage} decimals={1} suffix="%" />,
      desc: t("results.kpis.overtriageDesc"),
      color: "hsl(280 80% 60%)",
    },
  ];

  const recall = RESULTS?.recallPerLevel;

  return (
    <div className="space-y-4">
      {/* ── KPI grid ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {kpis.map(({ Icon, label, value, desc, color }) => (
          <Card key={label} className="relative overflow-hidden bg-card/65 p-4 backdrop-blur-md">
            <div className="absolute -right-8 -top-8 size-24 rounded-full opacity-15 blur-2xl" style={{ background: color }} />
            <div className="relative space-y-1.5">
              <Icon className="size-4" style={{ color }} />
              <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">{label}</p>
              <p className="text-2xl font-bold tabular-nums">{value}</p>
              <p className="text-[10px] leading-tight text-muted-foreground/70">{desc}</p>
            </div>
          </Card>
        ))}
      </div>
      {!RESULTS && (
        <p className="text-center text-[11px] text-muted-foreground">{t("results.pending")}</p>
      )}

      {/* ── Models + ensemble ── */}
      <Card className="relative overflow-hidden border-amber-500/40 bg-card/70 p-5 backdrop-blur-md">
        <div className="absolute -right-12 -top-12 size-40 rounded-full bg-amber-500/15 blur-3xl" />
        <div className="relative">
          <div className="mb-4 flex items-center gap-3">
            <div className="rounded-xl bg-amber-500/15 p-2.5 text-amber-400">
              <Sparkles className="size-4" />
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-amber-400">{t("results.headline")}</p>
              <p className="text-xs text-muted-foreground">
                {lang === "en" ? "Click any card to learn more" : "Kliknij kafelek, żeby dowiedzieć się więcej"}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 sm:grid-cols-7">
            {MODELS.map((model) => (
              <button
                key={model.id}
                onClick={() => setSelectedModel(model)}
                className="group relative rounded-lg border border-border/40 bg-muted/30 p-3 text-center transition-all duration-200 hover:border-primary/40 hover:bg-muted/50"
              >
                <p className="text-[10px] font-medium leading-tight text-muted-foreground">{model.name}</p>
                <p className="mt-1 font-mono text-sm font-bold tabular-nums">{formatMetric(model.qwk, 4)}</p>
                <Info className="mx-auto mt-1 size-2.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-60" />
              </button>
            ))}
            <button
              onClick={() => setEnsembleOpen(true)}
              className="group relative rounded-lg border border-amber-500/60 bg-amber-500/10 p-3 text-center transition-all duration-200 hover:bg-amber-500/15 hover:shadow-[0_0_20px_hsl(45_100%_50%/0.3)]"
            >
              <p className="text-[10px] font-medium leading-tight text-amber-400">Ensemble</p>
              <p className="mt-1 font-mono text-sm font-bold tabular-nums text-amber-400">{formatMetric(RESULTS?.qwk, 4)}</p>
              <Info className="mx-auto mt-1 size-2.5 text-amber-400 opacity-0 transition-opacity group-hover:opacity-60" />
            </button>
          </div>
        </div>
      </Card>

      {/* ── Recall per ESI level (ensemble) ── */}
      <Card className="bg-card/65 p-4 backdrop-blur-md">
        <div className="mb-3 flex items-center gap-2">
          <Award className="size-4 text-primary" />
          <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-primary">
            {lang === "en" ? "Ensemble recall per ESI level" : "Recall ensembla per poziom ESI"}
          </h4>
        </div>
        <div className="space-y-1.5">
          {ESI_LEVELS.map((lvl) => {
            const value = recall?.[lvl.id];
            return (
              <div key={lvl.id}>
                <div className="mb-0.5 flex justify-between text-[10px]">
                  <span style={{ color: lvl.color }}>
                    ESI {lvl.level} · {lang === "pl" ? lvl.colorPl : lvl.colorEn}
                  </span>
                  <span className="font-mono tabular-nums">{formatMetric(value, 1, "%")}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full" style={{ width: `${value ?? 0}%`, background: lvl.color }} />
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* ── Ensemble detail modal ── */}
      <AnimatePresence>
        {ensembleOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm"
            onClick={() => setEnsembleOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.92, y: 20 }}
              transition={{ type: "spring", damping: 24, stiffness: 280 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md"
            >
              <Card className="relative overflow-hidden border-2 border-amber-500/40 bg-background/95 p-6 backdrop-blur-xl">
                <div className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full bg-amber-500/10 blur-3xl" />
                <div className="relative mb-4 flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-bold">Ensemble</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{t("results.subtitle")}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-2xl font-bold text-amber-400">{formatMetric(RESULTS?.qwk, 4)}</p>
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">QWK</p>
                  </div>
                  <button
                    onClick={() => setEnsembleOpen(false)}
                    className="absolute right-0 top-0 rounded-full p-1 text-muted-foreground hover:bg-accent"
                  >
                    <X className="size-4" />
                  </button>
                </div>
                <ul className="space-y-2">
                  {ENSEMBLE_DETAILS[lang].map((d) => (
                    <li key={d} className="flex gap-2 text-sm text-muted-foreground">
                      <span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary/60" />
                      {d}
                    </li>
                  ))}
                </ul>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Single model modal ── */}
      <ModelDetailModal model={selectedModel} onClose={() => setSelectedModel(null)} />
    </div>
  );
}
