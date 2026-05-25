import { motion } from "framer-motion";
import { ArrowRight, Sparkles, Zap } from "lucide-react";
import { useTranslation } from "react-i18next";

import { CountUp } from "@/components/animations/CountUp";
import { MagneticButton } from "@/components/animations/MagneticButton";
import { TextReveal } from "@/components/animations/TextReveal";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

function scrollToAnchor(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const top = window.scrollY + el.getBoundingClientRect().top - 80;
  window.scrollTo({ top, behavior: "smooth" });
}

export function JourneyHero() {
  const { t } = useTranslation("sections");

  return (
    <div className="flex w-full max-w-4xl flex-col items-center text-center">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <Badge
          variant="outline"
          className="gap-1.5 border-primary/30 bg-primary/5 px-3 py-1 backdrop-blur-md"
        >
          <Sparkles className="size-3 text-primary" />
          <span className="font-medium">{t("hero.badge")}</span>
        </Badge>
      </motion.div>

      <h1 className="mt-6 text-balance text-4xl font-bold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
        <TextReveal text={t("hero.title")} className="block" />
        <TextReveal
          text={t("hero.titleAccent")}
          className="gradient-text block bg-[length:200%_auto] animate-gradient-shift"
          delay={0.3}
          staggerWords={0.06}
        />
      </h1>

      <motion.p
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7, delay: 0.7 }}
        className="mt-6 max-w-2xl text-balance text-base leading-relaxed text-muted-foreground sm:text-lg md:text-xl"
      >
        {t("hero.subtitle")}
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 1 }}
        className="mt-10 grid grid-cols-3 gap-6 rounded-2xl border border-border/60 bg-card/60 px-6 py-5 backdrop-blur-md sm:gap-12 sm:px-10"
      >
        <Metric
          label={t("hero.metric1")}
          value={<CountUp to={0.8718} decimals={4} duration={2.2} />}
        />
        <Metric
          label={t("hero.metric2")}
          value={<CountUp to={0.06} decimals={2} suffix="%" duration={2.2} />}
        />
        <Metric
          label={t("hero.metric3")}
          value={<CountUp to={350} suffix=" ms" duration={2.2} />}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 1.2 }}
        className="mt-8 flex flex-wrap items-center justify-center gap-3"
      >
        <MagneticButton>
          <Button
            size="xl"
            variant="glow"
            onClick={() => scrollToAnchor("demo")}
            className="group"
          >
            {t("hero.ctaPrimary")}
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
          </Button>
        </MagneticButton>
        <Button
          size="xl"
          variant="outline"
          className="backdrop-blur-md"
          onClick={() => scrollToAnchor("results")}
        >
          {t("hero.ctaSecondary")}
        </Button>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 1.5 }}
        className="mt-6 flex items-center gap-2 text-xs text-muted-foreground"
      >
        <Zap className="size-3 text-emerald-500" />
        <span>Działa w pełni lokalnie · bez przesyłu danych pacjentów</span>
      </motion.div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className="text-xl font-bold tabular-nums sm:text-2xl md:text-3xl">{value}</p>
    </div>
  );
}
