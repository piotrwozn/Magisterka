import { AnimatePresence, motion } from "framer-motion";
import { X, Layers, Database, Calendar, Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { ChartTooltip } from "@/components/shared/ChartTooltip";
import { Card } from "@/components/ui/card";
import { CLASS_DISTRIBUTION } from "@/lib/constants";

export type DataPanelId = "features" | "classes" | "split";

interface DataPanelModalProps {
  panel: DataPanelId | null;
  onClose: () => void;
}

const FEATURES_GROUPS = [
  {
    group: { pl: "Parametry życiowe (triażowe)", en: "Triage vitals" },
    count: 7,
    examples: ["SBP", "DBP", "HR", "SpO₂", "Temp", "RR", "Pain score"],
  },
  {
    group: { pl: "Dane demograficzne", en: "Demographics" },
    count: 8,
    examples: ["Wiek", "Płeć", "Rasa", "Metoda przybycia", "Pora dnia", "Dzień tygodnia", "Sezon", "Rok"],
  },
  {
    group: { pl: "Chief complaints (binarnie)", en: "Chief complaints (binary)" },
    count: 200,
    examples: ["cc_chest_pain", "cc_dyspnea", "cc_abdominal_pain", "cc_syncope", "…"],
  },
  {
    group: { pl: "Cechy inżynierowane", en: "Engineered features" },
    count: 116,
    examples: ["Shock index (HR/SBP)", "MEWS", "qSOFA", "SIRS score", "Temp × wiek", "SBP × wiek", "Missing flags"],
  },
];

const MTS_DETAILS = [
  { name: "Red (Natychmiastowy)", value: 68.6, color: "#DC2626", wait: "0 min", desc: "Stan zagrożenia życia. Resuscytacja lub natychmiastowa interwencja.", visits: "~307 k" },
  { name: "Orange (Pilny)", value: 13.2, color: "#EA580C", wait: "10 min", desc: "Bardzo wysoki priorytet. Intensywny ból, krytyczne parametry.", visits: "~59 k" },
  { name: "Yellow (Pilny)", value: 12.2, color: "#EAB308", wait: "60 min", desc: "Pilny, ale stabilny. Wymaga oceny w ciągu godziny.", visits: "~54 k" },
  { name: "Green (Mniej pilny)", value: 4.1, color: "#16A34A", wait: "120 min", desc: "Stan nieostry. Może poczekać do 2 godzin.", visits: "~18 k" },
  { name: "Blue (Nieostry)", value: 1.9, color: "#2563EB", wait: "240 min", desc: "Stan przewlekły lub drobna dolegliwość. Tryb ambulatoryjny.", visits: "~8.5 k" },
];

const SPLIT_DETAILS = [
  {
    label: "Train", pct: 80, color: "hsl(199 89% 58%)",
    rows: "446 480",
    desc: { pl: "Dane do treningu modeli i tuningowy CV. Najstarsze chronologicznie.", en: "Training data for models and CV tuning. Oldest chronologically." },
  },
  {
    label: "Val", pct: 10, color: "hsl(280 80% 60%)",
    rows: "55 747",
    desc: { pl: "Walidacja w trakcie treningu (early stopping). Nie dotknięte przez tuning Optuny.", en: "Validation during training (early stopping). Not touched by Optuna tuning." },
  },
  {
    label: "Test", pct: 10, color: "hsl(160 70% 50%)",
    rows: "55 802",
    desc: { pl: "Holdout set — 10% najnowszych wizyt. Jedyny prawdziwy test generalizacji. Użyty jednokrotnie.", en: "Holdout set — 10% most recent visits. True generalization test. Used once." },
  },
];

export function DataPanelModal({ panel, onClose }: DataPanelModalProps) {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";

  return (
    <AnimatePresence>
      {panel && (
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 24 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 24 }}
            transition={{ type: "spring", damping: 24, stiffness: 280 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-xl max-h-[85vh] overflow-y-auto"
          >
            {panel === "features" && <FeaturesDetail t={t} onClose={onClose} />}
            {panel === "classes"  && <ClassesDetail  t={t} lang={lang} onClose={onClose} />}
            {panel === "split"    && <SplitDetail    t={t} lang={lang} onClose={onClose} />}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ── Header shared ── */
function ModalHeader({ icon, title, subtitle, onClose }: {
  icon: React.ReactNode; title: string; subtitle: string; onClose: () => void;
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 text-primary">{icon}</div>
        <div>
          <h2 className="text-xl font-bold">{title}</h2>
          <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>
        </div>
      </div>
      <button
        onClick={onClose}
        className="shrink-0 rounded-full p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
      >
        <X className="size-4" />
      </button>
    </div>
  );
}

/* ── Features panel ── */
function FeaturesDetail({ t, onClose }: { t: ReturnType<typeof useTranslation>["t"]; onClose: () => void }) {
  const total = FEATURES_GROUPS.reduce((s, g) => s + g.count, 0);
  return (
    <Card className="overflow-hidden border-2 border-primary/30 bg-background/95 p-6 backdrop-blur-xl">
      <div className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full bg-primary/10 blur-3xl" />
      <ModalHeader icon={<Layers className="size-5" />} title={t("data.features.title")} subtitle={`${total} cech łącznie`} onClose={onClose} />
      <div className="space-y-3">
        {FEATURES_GROUPS.map((g) => (
          <Card key={g.count} className="bg-card/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-semibold">{g.group["pl"]}</span>
              <span className="font-mono text-lg font-bold text-primary">{g.count}</span>
            </div>
            <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-muted">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(g.count / total) * 100}%` }}
                transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                className="h-full rounded-full bg-primary"
              />
            </div>
            <p className="font-mono text-[10px] text-muted-foreground">{g.examples.join(" · ")}</p>
          </Card>
        ))}
      </div>
      <Card className="mt-3 bg-amber-500/10 border-amber-500/30 p-4">
        <div className="flex gap-2">
          <Info className="size-4 shrink-0 text-amber-400 mt-0.5" />
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("data.features.engineeredDesc")}
          </p>
        </div>
      </Card>
    </Card>
  );
}

/* ── Classes panel ── */
function ClassesDetail({ t, onClose }: { t: ReturnType<typeof useTranslation>["t"]; lang: "pl" | "en"; onClose: () => void }) {
  return (
    <Card className="overflow-hidden border-2 border-primary/30 bg-background/95 p-6 backdrop-blur-xl">
      <div className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full bg-primary/10 blur-3xl" />
      <ModalHeader icon={<Database className="size-5" />} title={t("data.classes.title")} subtitle="558 029 wizyt · Yale-New Haven Hospital" onClose={onClose} />
      <div className="mb-4 h-44">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={CLASS_DISTRIBUTION as unknown as { name: string; value: number; color: string }[]}
              dataKey="value"
              innerRadius={44}
              outerRadius={70}
              paddingAngle={3}
              animationDuration={1100}
            >
              {CLASS_DISTRIBUTION.map((e) => (
                <Cell key={e.name} fill={e.color} stroke="transparent" />
              ))}
            </Pie>
            <Tooltip
              cursor={{ fill: "transparent" }}
              content={<ChartTooltip suffix="%" formatter={(v) => (v as number).toFixed(1)} />}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-2">
        {MTS_DETAILS.map((c) => (
          <Card key={c.name} className="bg-card/60 p-3" style={{ borderColor: `${c.color}30` }}>
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="size-2.5 shrink-0 rounded-full" style={{ background: c.color, boxShadow: `0 0 8px ${c.color}` }} />
                <div>
                  <p className="text-xs font-bold">{c.name}</p>
                  <p className="text-[10px] text-muted-foreground">{c.desc}</p>
                </div>
              </div>
              <div className="shrink-0 text-right">
                <p className="font-mono text-sm font-bold" style={{ color: c.color }}>{c.value}%</p>
                <p className="font-mono text-[10px] text-muted-foreground">{c.visits}</p>
                <p className="text-[10px] text-muted-foreground">max {c.wait}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>
      <p className="mt-3 text-center text-[10px] text-muted-foreground">{t("data.classes.imbalanced")}</p>
    </Card>
  );
}

/* ── Split panel ── */
function SplitDetail({ t, lang, onClose }: { t: ReturnType<typeof useTranslation>["t"]; lang: "pl" | "en"; onClose: () => void }) {
  return (
    <Card className="overflow-hidden border-2 border-primary/30 bg-background/95 p-6 backdrop-blur-xl">
      <div className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full bg-primary/10 blur-3xl" />
      <ModalHeader icon={<Calendar className="size-5" />} title={t("data.split.title")} subtitle="Podział temporalny — bez wycieku z przyszłości" onClose={onClose} />
      <div className="space-y-4">
        {SPLIT_DETAILS.map((s) => (
          <Card key={s.label} className="bg-card/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="size-2.5 rounded-full" style={{ background: s.color }} />
                <span className="font-bold">{s.label}</span>
              </div>
              <div className="text-right">
                <span className="font-mono text-xl font-bold">{s.pct}%</span>
                <span className="ml-2 font-mono text-xs text-muted-foreground">{s.rows} wierszy</span>
              </div>
            </div>
            <div className="mb-3 h-2 overflow-hidden rounded-full bg-muted">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${s.pct}%` }}
                transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                className="h-full rounded-full"
                style={{ background: s.color }}
              />
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">{s.desc[lang]}</p>
          </Card>
        ))}
      </div>
      <Card className="mt-3 bg-emerald-500/10 border-emerald-500/30 p-4">
        <div className="flex gap-2">
          <Info className="size-4 shrink-0 text-emerald-400 mt-0.5" />
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("data.split.description")}
          </p>
        </div>
      </Card>
    </Card>
  );
}
