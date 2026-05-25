import { motion } from "framer-motion";
import { TrendingDown, TrendingUp } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import type { ShapValue } from "@/lib/types";

// Maps technical feature names → human-readable label (PL/EN aware)
const FEATURE_LABELS: Record<string, { pl: string; en: string }> = {
  triage_vital_sbp:        { pl: "Ciśnienie skurczowe",     en: "Systolic BP" },
  triage_vital_dbp:        { pl: "Ciśnienie rozkurczowe",   en: "Diastolic BP" },
  triage_vital_hr:         { pl: "Tętno",                   en: "Heart rate" },
  triage_vital_o2:         { pl: "Saturacja (SpO₂)",        en: "SpO₂ saturation" },
  triage_vital_temp:       { pl: "Temperatura ciała",       en: "Body temperature" },
  triage_vital_rr:         { pl: "Liczba oddechów",         en: "Respiratory rate" },
  age_group:               { pl: "Kategoria wiekowa",       en: "Age group" },
  age:                     { pl: "Wiek pacjenta",           en: "Patient age" },
  shock_index:             { pl: "Indeks wstrząsu (HR/SBP)", en: "Shock index (HR/SBP)" },
  mews:                    { pl: "Skala MEWS",              en: "MEWS score" },
  qsofa:                   { pl: "Skala qSOFA",             en: "qSOFA score" },
  sirs:                    { pl: "Kryteria SIRS",           en: "SIRS criteria" },
  arrival_hour:            { pl: "Godzina przybycia",       en: "Arrival hour" },
  arrival_mode:            { pl: "Sposób przybycia",        en: "Arrival mode" },
  triage_acuity:           { pl: "Wskazanie triażu",        en: "Triage acuity" },
  chief_complaint:         { pl: "Główna dolegliwość",      en: "Chief complaint" },
  pain_score:              { pl: "Ocena bólu (0–10)",       en: "Pain score (0–10)" },
  sbp_hr_ratio:            { pl: "Stosunek SBP/HR",         en: "SBP/HR ratio" },
  vital_temp_x_age:        { pl: "Temp × wiek (interakcja)", en: "Temp × age (interaction)" },
};

function featureLabel(name: string, lang: string): { label: string; desc: string } {
  // Check exact match
  const entry = FEATURE_LABELS[name.toLowerCase()];
  if (entry) {
    return { label: name, desc: lang === "en" ? entry.en : entry.pl };
  }
  // cc_ prefix = chief complaint binary flag
  if (name.startsWith("cc_")) {
    const complaint = name.replace("cc_", "").replace(/_/g, " ");
    return {
      label: name,
      desc: lang === "en" ? `Chief complaint: ${complaint}` : `Dolegliwość: ${complaint}`,
    };
  }
  // engineered_ prefix
  if (name.startsWith("eng_") || name.startsWith("engineered_")) {
    return {
      label: name,
      desc: lang === "en" ? "Engineered feature" : "Cecha inżynierowana",
    };
  }
  return { label: name, desc: "" };
}

interface ShapWaterfallProps {
  values: ShapValue[];
}

export function ShapWaterfall({ values }: ShapWaterfallProps) {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2);
  const maxAbs = Math.max(...values.map((v) => Math.abs(v.value)), 0.001);

  return (
    <Card className="p-5">
      <h4 className="mb-4 text-sm font-bold">{t("demo.result.shapTitle")}</h4>
      <div className="space-y-2.5">
        {values.map((v, idx) => {
          const isPos = v.direction === "positive";
          const widthPct = (Math.abs(v.value) / maxAbs) * 100;
          const { label, desc } = featureLabel(v.feature, lang);
          return (
            <motion.div
              key={v.feature}
              initial={{ opacity: 0, x: isPos ? 10 : -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1, duration: 0.4 }}
              className="space-y-1"
            >
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 font-mono">
                  {isPos ? (
                    <TrendingUp className="size-3 shrink-0 text-amber-500" />
                  ) : (
                    <TrendingDown className="size-3 shrink-0 text-emerald-500" />
                  )}
                  <span>{label}</span>
                  {desc && (
                    <span className="text-muted-foreground font-sans font-normal">
                      ({desc})
                    </span>
                  )}
                </span>
                <span
                  className={`ml-2 shrink-0 font-mono tabular-nums ${isPos ? "text-amber-500" : "text-emerald-500"}`}
                >
                  {isPos ? "+" : ""}
                  {v.value.toFixed(3)}
                </span>
              </div>
              <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${widthPct}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.1, ease: [0.22, 1, 0.36, 1] }}
                  className={`h-full rounded-full ${
                    isPos
                      ? "bg-gradient-to-r from-amber-500/60 to-amber-500"
                      : "bg-gradient-to-r from-emerald-500/60 to-emerald-500"
                  }`}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </Card>
  );
}
