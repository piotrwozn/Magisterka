import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Sparkles, Zap } from "lucide-react";
import { Suspense, lazy, useRef } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { AuroraBackground } from "@/components/animations/AuroraBackground";
import { CountUp } from "@/components/animations/CountUp";
import { MagneticButton } from "@/components/animations/MagneticButton";
import { Marquee } from "@/components/animations/Marquee";
import { TextReveal } from "@/components/animations/TextReveal";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const NeuralNetworkScene = lazy(() =>
  import("@/components/animations/NeuralNetworkScene").then((m) => ({
    default: m.NeuralNetworkScene,
  })),
);

const MARQUEE_ITEMS = [
  "QWK 0.8718",
  "·",
  "Manchester Triage System",
  "·",
  "8× RTX 5090",
  "·",
  "Optuna 5-fold CV",
  "·",
  "CatBoost · XGBoost · LightGBM · cuML RF · EBM",
  "·",
  "MedGemma 27B",
  "·",
  "Llama 3.2 3B",
  "·",
  "100% on-premise",
];

export function Hero() {
  const { t } = useTranslation("sections");
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const heroY = useTransform(scrollYProgress, [0, 1], ["0%", "20%"]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <section
      ref={ref}
      id="hero"
      className="relative flex min-h-[100svh] items-center overflow-hidden"
    >
      <AuroraBackground intensity={0.5} />
      <div className="absolute inset-0 grid-bg opacity-30" />

      <div className="absolute inset-y-0 right-0 hidden w-1/2 opacity-90 lg:block">
        <Suspense fallback={null}>
          <NeuralNetworkScene />
        </Suspense>
      </div>

      <motion.div
        style={{ y: heroY, opacity: heroOpacity }}
        className="container relative z-10 grid items-center gap-12 py-20 lg:grid-cols-2"
      >
        <div className="space-y-7">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Badge
              variant="outline"
              className="gap-1.5 border-primary/30 bg-primary/5 px-3 py-1 backdrop-blur-md"
            >
              <Sparkles className="size-3 text-primary" />
              <span className="font-medium">{t("hero.badge")}</span>
            </Badge>
          </motion.div>

          <h1 className="text-balance text-4xl font-bold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            <TextReveal text={t("hero.title")} className="block" />
            <TextReveal
              text={t("hero.titleAccent")}
              className="gradient-text block bg-[length:200%_auto] animate-gradient-shift"
              delay={0.4}
              staggerWords={0.06}
            />
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1 }}
            className="max-w-xl text-balance text-base leading-relaxed text-muted-foreground sm:text-lg md:text-xl"
          >
            {t("hero.subtitle")}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.3 }}
            className="grid grid-cols-3 gap-4 border-y border-border/60 py-5"
          >
            <Metric
              label={t("hero.metric1")}
              value={<CountUp to={0.8718} decimals={4} duration={2.4} />}
            />
            <Metric
              label={t("hero.metric2")}
              value={<CountUp to={0.06} decimals={2} suffix="%" duration={2.4} />}
            />
            <Metric
              label={t("hero.metric3")}
              value={<CountUp to={350} suffix=" ms" duration={2.4} />}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.5 }}
            className="flex flex-wrap items-center gap-3"
          >
            <MagneticButton>
              <Button asChild size="xl" variant="glow">
                <Link to="/demo" className="group">
                  {t("hero.ctaPrimary")}
                  <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
            </MagneticButton>
            <Button asChild size="xl" variant="outline" className="backdrop-blur-md">
              <a href="#results">{t("hero.ctaSecondary")}</a>
            </Button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.8 }}
            className="flex items-center gap-2 text-xs text-muted-foreground"
          >
            <Zap className="size-3 text-emerald-500" />
            <span>Działa w pełni lokalnie · bez przesyłu danych pacjentów</span>
          </motion.div>
        </div>

        <div className="hidden lg:block" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 2 }}
        className="absolute bottom-0 left-0 right-0 z-10 border-t border-border/40 bg-background/40 py-3 backdrop-blur-sm"
      >
        <Marquee speed={45} className="text-xs uppercase tracking-[0.25em] text-muted-foreground/70">
          {MARQUEE_ITEMS.map((item, i) => (
            <span key={i} className="font-mono">
              {item}
            </span>
          ))}
        </Marquee>
      </motion.div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className="text-xl font-bold tabular-nums sm:text-2xl">{value}</p>
    </div>
  );
}
