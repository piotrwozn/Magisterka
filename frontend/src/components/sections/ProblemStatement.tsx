import { motion } from "framer-motion";
import { AlertTriangle, Clock, Users, Activity } from "lucide-react";
import { useTranslation } from "react-i18next";

import { CountUp } from "@/components/animations/CountUp";
import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { MTSCategoryBadge } from "@/components/shared/MTSCategoryBadge";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Card } from "@/components/ui/card";
import { MTS_CATEGORIES } from "@/lib/constants";

export function ProblemStatement() {
  const { t, i18n } = useTranslation("sections");
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2) as "pl" | "en";

  return (
    <section id="problem" className="relative py-24 md:py-32">
      <div className="container space-y-16">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Problem"
            title={t("problem.title")}
            subtitle={t("problem.subtitle")}
          />
        </FadeInOnScroll>

        <div className="grid items-center gap-12 lg:grid-cols-2">
          <FadeInOnScroll>
            <div className="space-y-5 text-base leading-relaxed text-muted-foreground">
              <p>{t("problem.paragraph1")}</p>
              <p>{t("problem.paragraph2")}</p>
            </div>
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.15}>
            <div className="grid grid-cols-2 gap-3">
              <StatCard
                icon={Users}
                value={<CountUp to={8.3} decimals={1} suffix="M" duration={2} />}
                label={t("problem.stats.visits")}
              />
              <StatCard
                icon={Clock}
                value={<CountUp to={47} suffix=" min" duration={2} />}
                label={t("problem.stats.waitTime")}
              />
              <StatCard
                icon={AlertTriangle}
                value={
                  <span>
                    <CountUp to={3} duration={1.5} />×
                  </span>
                }
                label={t("problem.stats.undertriageCost")}
              />
              <StatCard
                icon={Activity}
                value={<CountUp to={0.1} decimals={1} suffix="%" duration={2} />}
                label={t("problem.stats.criticalMiss")}
              />
            </div>
          </FadeInOnScroll>
        </div>

        <FadeInOnScroll delay={0.2}>
          <Card className="overflow-hidden border-border/60 p-8 md:p-10">
            <h3 className="mb-6 text-xl font-bold">Manchester Triage System</h3>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {MTS_CATEGORIES.map((cat, i) => (
                <motion.div
                  key={cat.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  className="group relative overflow-hidden rounded-xl border border-border/60 p-5 transition-all hover:border-current"
                  style={{ color: cat.color }}
                >
                  <div
                    className="absolute inset-0 -z-10 opacity-0 transition-opacity group-hover:opacity-10"
                    style={{ background: `radial-gradient(circle at top right, ${cat.color}, transparent 70%)` }}
                  />
                  <MTSCategoryBadge category={cat.id} lang={lang} size="sm" />
                  <p className="mt-3 text-3xl font-bold tabular-nums text-foreground">
                    {cat.maxWait}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {lang === "pl"
                      ? t(`mts.${cat.code}Desc`, { ns: "common" })
                      : t(`mts.${cat.code}Desc`, { ns: "common" })}
                  </p>
                </motion.div>
              ))}
            </div>
          </Card>
        </FadeInOnScroll>
      </div>
    </section>
  );
}

function StatCard({
  icon: Icon,
  value,
  label,
}: {
  icon: typeof Users;
  value: React.ReactNode;
  label: string;
}) {
  return (
    <Card className="group flex flex-col gap-2 p-5 transition-all hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5">
      <Icon className="size-5 text-primary opacity-70 transition-all group-hover:opacity-100" />
      <p className="text-2xl font-bold tabular-nums">{value}</p>
      <p className="text-xs leading-snug text-muted-foreground">{label}</p>
    </Card>
  );
}
