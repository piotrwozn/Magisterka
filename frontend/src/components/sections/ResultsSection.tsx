import { Activity, AlertOctagon, ShieldCheck, TrendingUp } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { MetricValue } from "@/components/shared/MetricValue";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { ESI_LEVELS, MODELS, RESULTS } from "@/lib/constants";

export function ResultsSection() {
  const { t } = useTranslation("sections");
  const scored = MODELS.filter((m) => m.qwk !== undefined);
  const confusion = RESULTS?.confusionMatrix;
  const importance = RESULTS?.featureImportance;

  return (
    <section id="results" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Results"
            title={t("results.title")}
            subtitle={t("results.subtitle")}
          />
        </FadeInOnScroll>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            icon={TrendingUp}
            label={t("results.kpis.qwk")}
            value={<MetricValue value={RESULTS?.qwk} decimals={4} />}
            description={t("results.kpis.qwkDesc")}
            color="hsl(199 89% 58%)"
          />
          <KpiCard
            icon={AlertOctagon}
            label={t("results.kpis.undertriage")}
            value={<MetricValue value={RESULTS?.undertriage} decimals={1} suffix="%" />}
            description={t("results.kpis.undertriageDesc")}
            color="hsl(40 96% 56%)"
          />
          <KpiCard
            icon={ShieldCheck}
            label={t("results.kpis.criticalMiss")}
            value={<MetricValue value={RESULTS?.criticalMiss} decimals={2} suffix="%" />}
            description={t("results.kpis.criticalMissDesc")}
            color="hsl(160 70% 50%)"
          />
          <KpiCard
            icon={Activity}
            label={t("results.kpis.overtriage")}
            value={<MetricValue value={RESULTS?.overtriage} decimals={1} suffix="%" />}
            description={t("results.kpis.overtriageDesc")}
            color="hsl(280 80% 60%)"
          />
        </div>

        <div className="grid gap-5 lg:grid-cols-3">
          <FadeInOnScroll>
            <Card className="p-6 lg:col-span-2">
              <h3 className="mb-4 text-lg font-bold">{t("results.comparisonTitle")}</h3>
              {scored.length === 0 ? (
                <PendingNote text={t("results.pending")} />
              ) : (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={scored.map((m) => ({ name: m.name, qwk: m.qwk, color: m.color }))}
                    margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
                    <YAxis
                      domain={[0.7, 0.9]}
                      tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v: number) => v.toFixed(4)}
                    />
                    <Bar dataKey="qwk" radius={[8, 8, 0, 0]} animationDuration={1400}>
                      {scored.map((m) => (
                        <Cell key={m.id} fill={m.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              )}
            </Card>
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.1}>
            <Card className="p-6">
              <h3 className="mb-4 text-lg font-bold">{t("results.featureImportanceTitle")}</h3>
              {!importance || importance.length === 0 ? (
                <PendingNote text={t("results.pending")} />
              ) : (
              <div className="space-y-2.5">
                {importance.map((f, i) => (
                  <div key={f.feature} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-mono">{f.feature}</span>
                      <span className="tabular-nums text-muted-foreground">
                        {(f.value * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-primary to-purple-500"
                        style={{
                          width: `${(f.value / importance[0]!.value) * 100}%`,
                          animation: `slide-up 0.8s ${i * 0.08}s ease-out backwards`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              )}
            </Card>
          </FadeInOnScroll>
        </div>

        <FadeInOnScroll delay={0.15}>
          <Card className="overflow-hidden p-6">
            <h3 className="mb-5 text-lg font-bold">{t("results.confusionTitle")}</h3>
            {!confusion ? (
              <PendingNote text={t("results.pending")} />
            ) : (
            <div className="overflow-x-auto">
              <table className="mx-auto border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-xs text-muted-foreground"></th>
                    {ESI_LEVELS.map((c) => (
                      <th key={c.id} className="p-2 text-xs">
                        <span style={{ color: c.color }}>ESI {c.level}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusion.map((row, i) => {
                    const rowTotal = row.reduce((a, b) => a + b, 0);
                    return (
                      <tr key={i}>
                        <th className="p-2 text-right text-xs" style={{ color: ESI_LEVELS[i]!.color }}>
                          ESI {ESI_LEVELS[i]!.level}
                        </th>
                        {row.map((v, j) => {
                          const pct = rowTotal > 0 ? (v / rowTotal) * 100 : 0;
                          const isDiag = i === j;
                          return (
                            <td
                              key={j}
                              className="border border-border/40 p-2 text-center font-mono text-xs tabular-nums transition-all hover:scale-105"
                              style={{
                                backgroundColor: isDiag
                                  ? `${ESI_LEVELS[i]!.color}${Math.floor((pct / 100) * 200)
                                      .toString(16)
                                      .padStart(2, "0")}`
                                  : `hsl(var(--muted) / ${Math.min(pct / 50, 1)})`,
                              }}
                              title={`${v} (${pct.toFixed(1)}%)`}
                            >
                              {pct >= 1 ? `${pct.toFixed(1)}%` : v > 0 ? v : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            )}
            {confusion && (
              <p className="mt-4 text-xs text-muted-foreground">
                Wiersze = prawdziwy poziom ESI, kolumny = predykcja; diagonala = poprawne klasyfikacje
                ({RESULTS?.source}).
              </p>
            )}
          </Card>
        </FadeInOnScroll>
      </div>
    </section>
  );
}

function PendingNote({ text }: { text: string }) {
  return (
    <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-border/60 p-6 text-center text-sm text-muted-foreground">
      {text}
    </div>
  );
}

function KpiCard({
  icon: Icon,
  label,
  value,
  description,
  color,
}: {
  icon: typeof Activity;
  label: string;
  value: React.ReactNode;
  description: string;
  color: string;
}) {
  return (
    <FadeInOnScroll>
      <Card className="group relative overflow-hidden p-6 transition-all hover:-translate-y-1 hover:shadow-2xl">
        <div
          className="absolute -right-10 -top-10 size-32 rounded-full opacity-10 blur-2xl transition-all group-hover:opacity-25"
          style={{ background: color }}
        />
        <div className="relative flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {label}
            </p>
            <p className="text-3xl font-bold tabular-nums">{value}</p>
            <p className="text-[11px] text-muted-foreground">{description}</p>
          </div>
          <div className="rounded-lg p-2" style={{ backgroundColor: `${color}20`, color }}>
            <Icon className="size-5" />
          </div>
        </div>
      </Card>
    </FadeInOnScroll>
  );
}
