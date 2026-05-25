import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertOctagon, Award, Info, Layers, ShieldCheck, Sparkles, TrendingUp, X } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { CountUp } from "@/components/animations/CountUp";
import { ModelDetailModal } from "@/components/journey/ModelDetailModal";
import { Card } from "@/components/ui/card";
import { MODELS, type Model } from "@/lib/constants";

// ── Detail data for ensemble methods ─────────────────────────────────────

interface EnsembleDetail {
  id: string;
  name: string;
  qwk: number;
  color: string;
  highlight?: boolean;
  summary: { pl: string; en: string };
  details: { pl: string[]; en: string[] };
}

const ENSEMBLE_ITEMS: EnsembleDetail[] = [
  {
    id: "catboost",   name: "CatBoost",   qwk: 0.8729, color: "#FFCC00",
    summary: { pl: "Najlepszy pojedynczy model", en: "Best single model" },
    details: {
      pl: ["Oblivious symmetric trees z CUDA natywnie","154 triiale Optuna, najniższy critical miss 0.055%","Red recall 99.1% — najlepszy spośród wszystkich modeli"],
      en: ["Oblivious symmetric trees with native CUDA","154 Optuna trials, lowest critical miss 0.055%","Red recall 99.1% — best among all models"],
    },
  },
  {
    id: "xgboost",    name: "XGBoost",    qwk: 0.8691, color: "#00BFFF",
    summary: { pl: "QuantileDMatrix + multi-GPU", en: "QuantileDMatrix + multi-GPU" },
    details: {
      pl: ["lossguide grow policy, max_depth=14, 3500 estimatorów","150 triali Optuna, best trial #83","5 foldów CV równolegle na 5× RTX 5090"],
      en: ["lossguide grow policy, max_depth=14, 3,500 estimators","150 Optuna trials, best trial #83","5 CV folds in parallel on 5× RTX 5090"],
    },
  },
  {
    id: "lightgbm",   name: "LightGBM",   qwk: 0.8719, color: "#7CFC00",
    summary: { pl: "Custom weights, najniższy undertriage", en: "Custom weights, lowest undertriage" },
    details: {
      pl: ["Leaf-wise splitting, custom class weights z Optuny","Retrenowany — critical miss 0.41% → 0.04% (10× lepiej)","Undertriage 2.38% — najniższy ze wszystkich modeli"],
      en: ["Leaf-wise splitting, custom class weights from Optuna","Retrained — critical miss 0.41% → 0.04% (10× better)","Undertriage 2.38% — lowest across all models"],
    },
  },
  {
    id: "random_forest", name: "RF",      qwk: 0.840, color: "#FF69B4",
    summary: { pl: "Bagging, diversity booster", en: "Bagging, diversity booster" },
    details: {
      pl: ["cuML GPU, 900 drzew, custom class weights (retrenowany)","Wnoszący różnorodność — korelacja z boosterami tylko 0.97","Red recall 99.3%, Yellow 54.3%, podbił stacking Yellow o +3pp"],
      en: ["cuML GPU, 900 trees, custom class weights (retrained)","Diversity booster — correlation with boosters only 0.97","Red recall 99.3%, Yellow 54.3%, lifted stacking Yellow by +3pp"],
    },
  },
  {
    id: "avg",        name: "Średnia",     qwk: 0.870, color: "#A855F7",
    summary: { pl: "Prosta średnia 4 modeli", en: "Simple average of 4 models" },
    details: {
      pl: ["(p_CB + p_XGB + p_LGB + p_RF) / 4 dla każdej klasy","Wagi równe — nieoptymalne dla słabszego RF","Używane jako baseline do porównania"],
      en: ["(p_CB + p_XGB + p_LGB + p_RF) / 4 per class","Equal weights — suboptimal for weaker RF","Used as comparison baseline"],
    },
  },
  {
    id: "stacking",   name: "Stacking", qwk: 0.8797, color: "#F59E0B", highlight: true,
    summary: { pl: "LogReg meta — 4 modele", en: "LogReg meta — 4 models" },
    details: {
      pl: [
        "Meta-learner LogisticRegression na probability 4 modeli (20 features)",
        "QWK 0.8797 — +0.007 vs CatBoost solo",
        "Yellow recall 60.2% — +3pp dzięki dodaniu RF",
        "Critical miss 0.09%, undertriage 3.81%",
        "All-4 agreement 88.0% — RF obniżył z 90.9% (dodaje diversity)",
        "RF↔boosters korelacja tylko 0.97 — kluczowa różnorodność",
      ],
      en: [
        "LogisticRegression meta-learner on 4-model probability vectors (20 features)",
        "QWK 0.8797 — +0.007 vs CatBoost solo",
        "Yellow recall 60.2% — +3pp thanks to RF addition",
        "Critical miss 0.09%, undertriage 3.81%",
        "All-4 agreement 88.0% — RF lowered from 90.9% (adds diversity)",
        "RF↔boosters correlation only 0.97 — key diversity signal",
      ],
    },
  },
];

export function JourneyResults() {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";

  const [selectedEnsemble, setSelectedEnsemble] = useState<EnsembleDetail | null>(null);
  const [selectedModel, setSelectedModel]       = useState<Model | null>(null);

  const kpis = [
    {
      Icon: TrendingUp,
      label: t("results.kpis.qwk"),
      value: <CountUp to={0.8798} decimals={4} duration={2} />,
      desc:  t("results.kpis.qwkDesc"),
      color: "hsl(199 89% 58%)",
    },
    {
      Icon: AlertOctagon,
      label: t("results.kpis.undertriage"),
      value: <CountUp to={3.8} decimals={1} suffix="%" duration={2} />,
      desc:  t("results.kpis.undertriageDesc"),
      color: "hsl(40 96% 56%)",
    },
    {
      Icon: ShieldCheck,
      label: t("results.kpis.criticalMiss"),
      value: <CountUp to={0.08} decimals={2} suffix="%" duration={2} />,
      desc:  t("results.kpis.criticalMissDesc"),
      color: "hsl(160 70% 50%)",
    },
    {
      Icon: Activity,
      label: t("results.kpis.overtriage"),
      value: <CountUp to={11.1} decimals={1} suffix="%" duration={2} />,
      desc:  t("results.kpis.overtriageDesc"),
      color: "hsl(280 80% 60%)",
    },
  ];

  const handleMiniClick = (item: EnsembleDetail) => {
    // Single models → ModelDetailModal; ensemble methods → EnsembleDetailModal
    const model = MODELS.find((m) => m.id === item.id);
    if (model && model.status === "trained") {
      setSelectedModel(model);
    } else {
      setSelectedEnsemble(item);
    }
  };

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

      {/* ── Ensemble showcase ── */}
      <Card className="relative overflow-hidden border-amber-500/40 bg-card/70 p-5 backdrop-blur-md">
        <div className="absolute -right-12 -top-12 size-40 rounded-full bg-amber-500/15 blur-3xl" />
        <div className="relative">
          {/* Header */}
          <div className="mb-4 flex items-center gap-3">
            <div className="rounded-xl bg-amber-500/15 p-2.5 text-amber-400">
              <Sparkles className="size-4" />
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-amber-400">Stacking ensemble</p>
              <p className="text-xs text-muted-foreground">
                {lang === "en"
                  ? "Click any card to learn more"
                  : "Kliknij kafelek żeby dowiedzieć się więcej"}
              </p>
            </div>
          </div>

          {/* 6 cards — 3+3 grid, no nesting in md:grid-cols-[auto_1fr] */}
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {ENSEMBLE_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => handleMiniClick(item)}
                className={`group relative rounded-lg border p-3 text-center transition-all duration-200
                  ${item.highlight
                    ? "border-amber-500/60 bg-amber-500/10 hover:bg-amber-500/15 hover:shadow-[0_0_20px_hsl(45_100%_50%/0.3)]"
                    : "border-border/40 bg-muted/30 hover:border-primary/40 hover:bg-muted/50"
                  }`}
              >
                {item.highlight && (
                  <span className="absolute -right-1 -top-1 flex size-4 items-center justify-center rounded-full bg-amber-500 text-[8px] font-bold text-black">⭐</span>
                )}
                <p className={`text-[10px] font-medium leading-tight ${item.highlight ? "text-amber-400" : "text-muted-foreground"}`}>
                  {item.name}
                </p>
                <p
                  className="mt-1 font-mono text-sm font-bold tabular-nums"
                  style={{ color: item.highlight ? item.color : undefined }}
                >
                  {item.qwk.toFixed(4)}
                </p>
                <Info className={`mx-auto mt-1 size-2.5 opacity-0 transition-opacity group-hover:opacity-60 ${item.highlight ? "text-amber-400" : "text-muted-foreground"}`} />
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* ── Agreement ── */}
      <Card className="bg-card/65 p-4 backdrop-blur-md">
        <div className="mb-3 flex items-center gap-2">
          <Layers className="size-4 text-primary" />
          <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-primary">
            {lang === "en" ? "Inter-model agreement" : "Zgodność między modelami"}
          </h4>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          {[
            { value: "90.9%", label: lang === "en" ? "All 3 agree" : "Pełna zgodność", color: "hsl(160 70% 50%)" },
            { value: "6.7%",  label: lang === "en" ? "Minor difference" : "Niewielka różnica", color: "hsl(40 96% 56%)" },
            { value: "2.4%",  label: lang === "en" ? "Conflict ≥2 grades" : "Konflikt ≥2 stopnie", color: "hsl(0 84% 60%)" },
          ].map((s) => (
            <div key={s.label}>
              <p className="text-xl font-bold" style={{ color: s.color }}>{s.value}</p>
              <p className="text-[10px] text-muted-foreground">{s.label}</p>
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-[10px] text-muted-foreground">
          {[
            { a: "CatBoost", b: "XGBoost",  r: 0.993 },
            { a: "XGBoost",  b: "LightGBM", r: 0.978 },
            { a: "CatBoost", b: "LightGBM", r: 0.974 },
          ].map(({ a, b, r }) => (
            <div key={a+b} className="rounded-md border border-border/30 bg-muted/30 px-1.5 py-1 text-center font-mono text-[9px]">
              {a} ↔ {b}<br />r = {r.toFixed(3)}
            </div>
          ))}
        </div>
      </Card>

      {/* ── Per-class recall (stacking) ── */}
      <Card className="bg-card/65 p-4 backdrop-blur-md">
        <div className="mb-3 flex items-center gap-2">
          <Award className="size-4 text-primary" />
          <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-primary">
            {lang === "en" ? "Stacking recall per MTS class" : "Recall stackingu per kategoria MTS"}
          </h4>
        </div>
        <div className="space-y-1.5">
          {[
            { label: "Red",    value: 97.9, color: "#DC2626" },
            { label: "Orange", value: 72.1, color: "#EA580C" },
            { label: "Yellow", value: 57.4, color: "#EAB308" },
            { label: "Green",  value: 53.5, color: "#16A34A" },
            { label: "Blue",   value: 60.1, color: "#2563EB" },
          ].map((c) => (
            <div key={c.label}>
              <div className="mb-0.5 flex justify-between text-[10px]">
                <span style={{ color: c.color }}>{c.label}</span>
                <span className="font-mono tabular-nums">{c.value.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full" style={{ width: `${c.value}%`, background: c.color }} />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* ── Ensemble detail modal ── */}
      <AnimatePresence>
        {selectedEnsemble && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
            onClick={() => setSelectedEnsemble(null)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.92, y: 20 }}
              transition={{ type: "spring", damping: 24, stiffness: 280 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-md"
            >
              <Card
                className="relative overflow-hidden border-2 bg-background/95 p-6 backdrop-blur-xl"
                style={{ borderColor: `${selectedEnsemble.color}50` }}
              >
                <div
                  className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full opacity-10 blur-3xl"
                  style={{ background: selectedEnsemble.color }}
                />
                <div className="relative mb-4 flex items-start justify-between gap-4">
                  <div>
                    <div className="mb-1.5 flex items-center gap-2">
                      <span className="size-3 rounded-full" style={{ background: selectedEnsemble.color, boxShadow: `0 0 8px ${selectedEnsemble.color}` }} />
                      <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                        {selectedEnsemble.highlight ? "Best result" : "Ensemble strategy"}
                      </span>
                    </div>
                    <h2 className="text-xl font-bold">{selectedEnsemble.name}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{selectedEnsemble.summary[lang]}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-2xl font-bold" style={{ color: selectedEnsemble.color }}>
                      {selectedEnsemble.qwk.toFixed(4)}
                    </p>
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">QWK</p>
                  </div>
                  <button
                    onClick={() => setSelectedEnsemble(null)}
                    className="absolute right-0 top-0 rounded-full p-1 text-muted-foreground hover:bg-accent"
                  >
                    <X className="size-4" />
                  </button>
                </div>
                <ul className="space-y-2">
                  {selectedEnsemble.details[lang].map((d, i) => (
                    <li key={i} className="flex gap-2 text-sm text-muted-foreground">
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

      {/* ── Single model modal (reuses existing) ── */}
      <ModelDetailModal model={selectedModel} onClose={() => setSelectedModel(null)} />
    </div>
  );
}
