import { Database, Layers, Calendar } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { CountUp } from "@/components/animations/CountUp";
import { DataPanelModal, type DataPanelId } from "@/components/journey/DataPanelModal";
import { ChartTooltip } from "@/components/shared/ChartTooltip";
import { Card } from "@/components/ui/card";
import { CLASS_DISTRIBUTION } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function JourneyDataPanels() {
  const { t } = useTranslation("sections");
  const [open, setOpen] = useState<DataPanelId | null>(null);

  const panelClass = "group relative cursor-pointer bg-card/65 p-5 backdrop-blur-md transition-all duration-300 hover:border-primary/50 hover:shadow-[0_0_24px_hsl(var(--primary)/0.12)] hover:-translate-y-0.5";

  return (
    <>
      <div className="grid gap-3 lg:grid-cols-3">
        {/* ── Cechy ── */}
        <Card className={panelClass} onClick={() => setOpen("features")}>
          <ClickHint />
          <Layers className="mb-3 size-5 text-primary" />
          <h3 className="text-sm font-bold">{t("data.features.title")}</h3>
          <dl className="mt-3 space-y-2 text-sm">
            <Row label={t("data.features.original")}>
              <CountUp to={974} duration={1.4} />
            </Row>
            <Row label={t("data.features.engineered")}>
              +<CountUp to={116} duration={1.6} />
            </Row>
            <div className="my-1 h-px bg-border/60" />
            <Row label={t("data.features.total")} bold>
              <CountUp to={1090} duration={1.8} />
            </Row>
          </dl>
          <p className="mt-3 text-[10px] leading-snug text-muted-foreground">
            {t("data.features.engineeredDesc")}
          </p>
        </Card>

        {/* ── Dystrybucja klas ── */}
        <Card className={panelClass} onClick={() => setOpen("classes")}>
          <ClickHint />
          <Database className="mb-3 size-5 text-primary" />
          <h3 className="text-sm font-bold">{t("data.classes.title")}</h3>
          <div className="-mx-2 mt-1 h-36">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={CLASS_DISTRIBUTION as unknown as { name: string; value: number; color: string }[]}
                  dataKey="value"
                  innerRadius={36}
                  outerRadius={58}
                  paddingAngle={3}
                  animationDuration={1100}
                  animationBegin={150}
                >
                  {CLASS_DISTRIBUTION.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip
                  cursor={{ fill: "transparent" }}
                  content={<ChartTooltip suffix="%" formatter={(v) => (v as number).toFixed(1)} />}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <ul className="mt-2 grid grid-cols-5 gap-1 text-[9px]">
            {CLASS_DISTRIBUTION.map((c) => (
              <li key={c.name} className="text-center">
                <span
                  className="inline-block size-1.5 rounded-full"
                  style={{ background: c.color, boxShadow: `0 0 6px ${c.color}80` }}
                />
                <p className="mt-0.5 font-mono tabular-nums">{c.value.toFixed(1)}%</p>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[10px] text-muted-foreground">{t("data.classes.imbalanced")}</p>
        </Card>

        {/* ── Podział ── */}
        <Card className={panelClass} onClick={() => setOpen("split")}>
          <ClickHint />
          <Calendar className="mb-3 size-5 text-primary" />
          <h3 className="text-sm font-bold">{t("data.split.title")}</h3>
          <div className="mt-4 space-y-2.5">
            <SplitBar label="train" pct={80} color="hsl(199 89% 58%)" />
            <SplitBar label="val"   pct={10} color="hsl(280 80% 60%)" />
            <SplitBar label="test"  pct={10} color="hsl(160 70% 50%)" />
          </div>
          <p className="mt-3 text-[10px] leading-snug text-muted-foreground">
            {t("data.split.description")}
          </p>
        </Card>
      </div>

      <DataPanelModal panel={open} onClose={() => setOpen(null)} />
    </>
  );
}

function ClickHint() {
  return (
    <div className="absolute inset-x-0 bottom-0 translate-y-full rounded-b-lg bg-primary/90 py-1 text-center text-[10px] font-semibold text-primary-foreground transition-transform duration-200 group-hover:translate-y-0">
      Kliknij po szczegóły
    </div>
  );
}

function Row({ label, bold, children }: { label: string; bold?: boolean; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={cn(bold ? "text-lg font-bold tabular-nums" : "tabular-nums")}>{children}</dd>
    </div>
  );
}

function SplitBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}
