import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { lazy, Suspense, useRef } from "react";

import type { ShapeName } from "@/components/animations/MorphingParticles";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { cn } from "@/lib/utils";

const MorphingParticles = lazy(() =>
  import("@/components/animations/MorphingParticles").then((m) => ({
    default: m.MorphingParticles,
  })),
);

export interface Stage {
  eyebrow: string;
  title: string;
  body: string;
  side: "left" | "right";
  shape: ShapeName;
  stat?: { value: string; label: string };
  accentColor?: string;
}

interface ScrollStageProps {
  stages: Stage[];
  stageHeight?: string;
}

export function ScrollStage({ stages, stageHeight = "100vh" }: ScrollStageProps) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end end"],
  });

  const shapes = stages.map((s) => s.shape);

  return (
    <section
      ref={ref}
      aria-label="Scrollytelling narrative"
      className="relative isolate"
      style={{ height: `calc(${stageHeight} * ${stages.length})` }}
    >
      <div className="sticky top-0 flex h-screen items-center justify-center overflow-hidden">
        <div className="absolute inset-0 grid-bg opacity-25" />
        <StageBackdrop progress={scrollYProgress} stages={stages} />

        <div className="absolute inset-0 -z-0 flex items-center justify-center">
          <div className="h-[110vh] w-[110vh] max-w-[100vw]">
            {!reduced && (
              <Suspense fallback={null}>
                <MorphingParticles progress={scrollYProgress} shapes={shapes} />
              </Suspense>
            )}
          </div>
        </div>

        <div className="container relative z-10 h-full">
          {stages.map((stage, i) => (
            <StageText
              key={i}
              stage={stage}
              index={i}
              total={stages.length}
              progress={scrollYProgress}
            />
          ))}
        </div>

        <ProgressIndicator total={stages.length} progress={scrollYProgress} />
      </div>
    </section>
  );
}

function StageText({
  stage,
  index,
  total,
  progress,
}: {
  stage: Stage;
  index: number;
  total: number;
  progress: MotionValue<number>;
}) {
  const slice = 1 / total;
  const start = index * slice;
  const end = start + slice;
  const fadeInEnd = start + slice * 0.15;
  const fadeOutStart = end - slice * 0.15;

  const opacity = useTransform(
    progress,
    [start, fadeInEnd, fadeOutStart, end],
    index === 0
      ? [1, 1, 0, 0]
      : index === total - 1
        ? [0, 1, 1, 1]
        : [0, 1, 1, 0],
  );

  const xOffset = stage.side === "left" ? -50 : 50;
  const x = useTransform(progress, [start, fadeInEnd, fadeOutStart, end], [
    xOffset,
    0,
    0,
    -xOffset,
  ]);

  return (
    <motion.div
      style={{ opacity, x }}
      className={cn(
        "absolute inset-y-0 flex w-full max-w-md flex-col justify-center",
        stage.side === "left"
          ? "left-6 items-start text-left lg:left-12"
          : "right-6 items-end text-right lg:right-12",
      )}
    >
      <span className="mb-3 inline-flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-primary">
        <span className="size-1 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]" />
        {stage.eyebrow}
      </span>

      <h3 className="text-balance text-3xl font-bold leading-tight tracking-tight sm:text-4xl md:text-[2.75rem]">
        {stage.title}
      </h3>

      <p className="mt-4 text-pretty text-base leading-relaxed text-muted-foreground md:text-lg">
        {stage.body}
      </p>

      {stage.stat && (
        <div className="mt-7 inline-flex items-baseline gap-3">
          <span className="text-5xl font-bold tabular-nums gradient-text md:text-6xl">
            {stage.stat.value}
          </span>
          <span className="text-xs uppercase tracking-widest text-muted-foreground">
            {stage.stat.label}
          </span>
        </div>
      )}
    </motion.div>
  );
}

function StageBackdrop({
  progress,
  stages,
}: {
  progress: MotionValue<number>;
  stages: Stage[];
}) {
  const hue = useTransform(progress, [0, 1], [199, 320]);
  const bg = useTransform(hue, (h) => `hsl(${h} 89% 58% / 0.1)`);

  return (
    <motion.div
      aria-hidden
      className="pointer-events-none absolute inset-0 -z-10"
      style={{ backgroundColor: bg }}
    >
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_center,transparent_30%,hsl(var(--background))_85%)]" />
      <div
        className="absolute left-1/2 top-1/2 size-[80vh] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-40 blur-3xl"
        style={{ background: "radial-gradient(circle, hsl(var(--primary)), transparent 60%)" }}
      />
      {/* aria-hidden visual cue — total stages count for ARIA */}
      <span className="sr-only">{stages.length} stages</span>
    </motion.div>
  );
}

function ProgressIndicator({
  total,
  progress,
}: {
  total: number;
  progress: MotionValue<number>;
}) {
  const dots = Array.from({ length: total });
  return (
    <div className="pointer-events-none absolute right-4 top-1/2 z-20 hidden -translate-y-1/2 flex-col gap-2.5 lg:flex">
      {dots.map((_, i) => (
        <Dot key={i} index={i} total={total} progress={progress} />
      ))}
    </div>
  );
}

function Dot({
  index,
  total,
  progress,
}: {
  index: number;
  total: number;
  progress: MotionValue<number>;
}) {
  const opacity = useTransform(progress, (p) => {
    const stage = p * (total - 1);
    return Math.abs(stage - index) < 0.4 ? 1 : 0.22;
  });
  const scale = useTransform(progress, (p) => {
    const stage = p * (total - 1);
    return Math.abs(stage - index) < 0.4 ? 1.4 : 1;
  });
  return (
    <motion.div
      style={{ opacity, scale }}
      className="size-1.5 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]"
    />
  );
}
