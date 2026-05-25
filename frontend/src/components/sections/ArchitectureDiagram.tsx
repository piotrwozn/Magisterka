import { motion } from "framer-motion";
import { ArrowDown, Brain, Cpu, FileJson, Network, ShieldCheck } from "lucide-react";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { PulsingDot } from "@/components/animations/PulsingDot";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export function ArchitectureDiagram() {
  const { t } = useTranslation("sections");

  const layers = [
    {
      key: "layer0",
      icon: FileJson,
      latency: t("architecture.layer0.latency"),
      color: "#06b6d4",
    },
    {
      key: "layer1a",
      icon: Cpu,
      latency: t("architecture.layer1a.latency"),
      color: "#3b82f6",
      parallel: true,
    },
    {
      key: "layer1b",
      icon: Brain,
      latency: t("architecture.layer1b.latency"),
      color: "#8b5cf6",
      parallel: true,
    },
    {
      key: "layer2",
      icon: Network,
      latency: t("architecture.layer2.latency"),
      color: "#a855f7",
    },
  ];

  return (
    <section id="architecture" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="Architecture"
            title={t("architecture.title")}
            subtitle={t("architecture.subtitle")}
          />
        </FadeInOnScroll>

        <div className="mx-auto max-w-4xl space-y-5">
          <Layer
            data={layers[0]!}
            title={t("architecture.layer0.title")}
            model={t("architecture.layer0.model")}
            description={t("architecture.layer0.description")}
          />

          <FlowConnector />

          <div className="grid gap-5 md:grid-cols-2">
            <Layer
              data={layers[1]!}
              title={t("architecture.layer1a.title")}
              model={t("architecture.layer1a.model")}
              description={t("architecture.layer1a.description")}
            />
            <Layer
              data={layers[2]!}
              title={t("architecture.layer1b.title")}
              model={t("architecture.layer1b.model")}
              description={t("architecture.layer1b.description")}
            />
          </div>

          <FlowConnector />

          <Layer
            data={layers[3]!}
            title={t("architecture.layer2.title")}
            model={t("architecture.layer2.model")}
            description={t("architecture.layer2.description")}
          />
        </div>

        <FadeInOnScroll delay={0.15}>
          <Card className="mx-auto max-w-4xl border-emerald-500/30 bg-emerald-500/5 p-7">
            <div className="flex items-start gap-4">
              <div className="rounded-lg bg-emerald-500/20 p-2 text-emerald-500">
                <ShieldCheck className="size-5" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold">{t("architecture.safetyRules.title")}</h3>
                <ul className="mt-3 grid gap-2 sm:grid-cols-3">
                  <li className="text-sm text-muted-foreground">
                    <span className="mr-1.5 font-mono text-emerald-500">01</span>{" "}
                    {t("architecture.safetyRules.rule1")}
                  </li>
                  <li className="text-sm text-muted-foreground">
                    <span className="mr-1.5 font-mono text-emerald-500">02</span>{" "}
                    {t("architecture.safetyRules.rule2")}
                  </li>
                  <li className="text-sm text-muted-foreground">
                    <span className="mr-1.5 font-mono text-emerald-500">03</span>{" "}
                    {t("architecture.safetyRules.rule3")}
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </FadeInOnScroll>
      </div>
    </section>
  );
}

function Layer({
  data,
  title,
  model,
  description,
}: {
  data: { key: string; icon: typeof Brain; latency: string; color: string };
  title: string;
  model: string;
  description: string;
}) {
  const Icon = data.icon;
  return (
    <FadeInOnScroll>
      <Card className="group relative overflow-hidden p-6 transition-all hover:border-primary/40 hover:shadow-xl">
        <div
          className="absolute -right-16 -top-16 size-48 rounded-full opacity-10 blur-2xl transition-all duration-500 group-hover:opacity-20"
          style={{ background: data.color }}
        />
        <div className="relative flex items-start gap-4">
          <div
            className="flex size-12 shrink-0 items-center justify-center rounded-xl"
            style={{ backgroundColor: `${data.color}26`, color: data.color }}
          >
            <Icon className="size-6" />
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-lg font-bold">{title}</h3>
              <Badge variant="outline" className="gap-1.5 text-xs">
                <PulsingDot color={data.color} size={6} />
                <span className="font-mono">{data.latency}</span>
              </Badge>
            </div>
            <p className="font-mono text-xs uppercase tracking-wider" style={{ color: data.color }}>
              {model}
            </p>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
      </Card>
    </FadeInOnScroll>
  );
}

function FlowConnector() {
  return (
    <div className="flex justify-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.5 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
        className="flex flex-col items-center gap-1.5"
      >
        <ArrowDown className="size-4 text-muted-foreground" />
        <div className="h-8 w-px bg-gradient-to-b from-primary/60 to-transparent" />
      </motion.div>
    </div>
  );
}
