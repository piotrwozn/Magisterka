import { Database, Layers, Calendar } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { CountUp } from "@/components/animations/CountUp";
import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { CLASS_DISTRIBUTION } from "@/lib/constants";

export function DataSection() {
  const { t } = useTranslation("sections");

  return (
    <section id="data" className="relative py-24 md:py-32">
      <div className="container space-y-16">
        <FadeInOnScroll>
          <SectionHeader eyebrow="Data" title={t("data.title")} subtitle={t("data.subtitle")} />
        </FadeInOnScroll>

        <FadeInOnScroll delay={0.1}>
          <p className="mx-auto max-w-3xl text-center text-base leading-relaxed text-muted-foreground md:text-lg">
            {t("data.description")}
          </p>
        </FadeInOnScroll>

        <div className="grid gap-6 lg:grid-cols-3">
          <FadeInOnScroll delay={0.15}>
            <Card className="p-7 transition-all hover:border-primary/40 hover:shadow-xl hover:shadow-primary/5">
              <Layers className="mb-4 size-7 text-primary" />
              <h3 className="text-lg font-bold">{t("data.features.title")}</h3>
              <dl className="mt-5 space-y-3">
                <Row label={t("data.features.original")}>
                  <CountUp to={293} duration={1.6} />
                </Row>
                <Row label={t("data.features.engineered")}>
                  +<CountUp to={43} duration={1.8} />
                </Row>
                <div className="my-2 h-px bg-border" />
                <Row label={t("data.features.total")} bold>
                  <CountUp to={336} duration={2} />
                </Row>
              </dl>
              <p className="mt-4 text-xs text-muted-foreground">
                {t("data.features.engineeredDesc")}
              </p>
            </Card>
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.25}>
            <Card className="p-7 transition-all hover:border-primary/40 hover:shadow-xl hover:shadow-primary/5">
              <Database className="mb-4 size-7 text-primary" />
              <h3 className="text-lg font-bold">{t("data.classes.title")}</h3>
              <div className="mt-3 h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={CLASS_DISTRIBUTION as unknown as { name: string; value: number; color: string }[]}
                      dataKey="value"
                      innerRadius={42}
                      outerRadius={68}
                      paddingAngle={3}
                      animationDuration={1100}
                      animationBegin={200}
                    >
                      {CLASS_DISTRIBUTION.map((entry) => (
                        <Cell key={entry.name} fill={entry.color} stroke="transparent" />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(v: number) => `${v}%`}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground">{t("data.classes.imbalanced")}</p>
            </Card>
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.35}>
            <Card className="p-7 transition-all hover:border-primary/40 hover:shadow-xl hover:shadow-primary/5">
              <Calendar className="mb-4 size-7 text-primary" />
              <h3 className="text-lg font-bold">{t("data.split.title")}</h3>
              <div className="mt-5 space-y-2.5">
                <SplitBar label="train" pct={80} color="hsl(199 89% 58%)" />
                <SplitBar label="val" pct={10} color="hsl(280 80% 60%)" />
                <SplitBar label="test" pct={10} color="hsl(160 70% 50%)" />
              </div>
              <p className="mt-4 text-xs text-muted-foreground">{t("data.split.description")}</p>
            </Card>
          </FadeInOnScroll>
        </div>
      </div>
    </section>
  );
}

function Row({
  label,
  bold,
  children,
}: {
  label: string;
  bold?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className={bold ? "text-2xl font-bold tabular-nums" : "text-base tabular-nums"}>
        {children}
      </dd>
    </div>
  );
}

function SplitBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
