import { AnimatePresence, motion, useScroll, useTransform, useMotionValueEvent, type MotionValue } from "framer-motion";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { ShapeName } from "@/components/animations/MorphingParticles";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { useJourneyStore } from "@/stores/journeyStore";
import { cn } from "@/lib/utils";

const MorphingParticles = lazy(() =>
  import("@/components/animations/MorphingParticles").then((m) => ({
    default: m.MorphingParticles,
  })),
);

export interface JourneyStage {
  eyebrow?: string;
  title: string;
  body?: string;
  shape: ShapeName;
  side?: "left" | "right";
  stat?: { value: string; label: string };
  content?: ReactNode;
  heightFactor?: number;
}

export interface JourneyChapter {
  id: string;
  number: string;
  name: string;
  stages: JourneyStage[];
}

interface FlatStage extends JourneyStage {
  chapterIndex: number;
  globalIndex: number;
  start: number;
  end: number;
  chapter: JourneyChapter;
}

interface CinematicJourneyProps {
  chapters: JourneyChapter[];
}

export function CinematicJourney({ chapters }: CinematicJourneyProps) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  useJourneyStore(); // subscribe to re-render on autoPlay toggle (getState() used in effects)

  const { flat, totalHeight } = useMemo(() => {
    const flat: FlatStage[] = [];
    let cursor = 0;
    let total = 0;
    chapters.forEach((c, ci) => {
      c.stages.forEach((s) => {
        const h = s.heightFactor ?? 1;
        flat.push({
          ...s,
          chapter: c,
          chapterIndex: ci,
          globalIndex: flat.length,
          start: cursor,
          end: cursor + h,
        });
        cursor += h;
        total += h;
      });
    });
    flat.forEach((s) => {
      s.start /= total;
      s.end /= total;
    });
    return { flat, totalHeight: total };
  }, [chapters]);

  const shapes = flat.map((s) => s.shape);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end end"],
  });

  const [activeChapter, setActiveChapter] = useState(0);
  const [activeStage, setActiveStage] = useState(0);

  // Ref for stable autoplay closure
  const activeStageRef = useRef(activeStage);
  useEffect(() => { activeStageRef.current = activeStage; }, [activeStage]);

  useMotionValueEvent(scrollYProgress, "change", (p) => {
    let idx = 0;
    for (let i = 0; i < flat.length; i++) {
      if (p >= flat[i]!.start) idx = i;
      else break;
    }
    setActiveStage(idx);
    setActiveChapter(flat[idx]!.chapterIndex);
  });

  // ── Scroll to exact stage — targets fadeIn point so text is immediately visible ──
  const scrollToStage = useCallback((idx: number) => {
    if (!ref.current) return;
    const stage = flat[Math.max(0, Math.min(idx, flat.length - 1))];
    if (!stage) return;
    const range = stage.end - stage.start;
    // Scroll past fadeIn (0.18) so text is fully visible on arrival
    const progress = stage.start + range * 0.22;
    const sectionTop = ref.current.getBoundingClientRect().top + window.scrollY;
    const sectionScrollRange = ref.current.offsetHeight - window.innerHeight;
    window.scrollTo({ top: sectionTop + progress * sectionScrollRange, behavior: "smooth" });
  }, [flat]);

  const scrollToChapter = useCallback((chapterIndex: number) => {
    const idx = flat.findIndex((s) => s.chapterIndex === chapterIndex);
    if (idx >= 0) scrollToStage(idx);
  }, [flat, scrollToStage]);

  // Keep latest scrollToStage + flat.length in refs so interval never goes stale
  const scrollToStageRef = useRef(scrollToStage);
  useEffect(() => { scrollToStageRef.current = scrollToStage; }, [scrollToStage]);
  const flatLengthRef = useRef(flat.length);
  useEffect(() => { flatLengthRef.current = flat.length; }, [flat.length]);

  // ── Arrow key navigation ─────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      if (e.key === "ArrowDown") { e.preventDefault(); scrollToStageRef.current(activeStageRef.current + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); scrollToStageRef.current(activeStageRef.current - 1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Auto-play: stable interval with empty deps, reads store via getState() ──
  useEffect(() => {
    const timer = setInterval(() => {
      if (!useJourneyStore.getState().autoPlay) return;
      const next = activeStageRef.current + 1;
      if (next >= flatLengthRef.current) {
        useJourneyStore.getState().setAutoPlay(false);
        return;
      }
      scrollToStageRef.current(next);
    }, 3000);
    return () => clearInterval(timer);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Stop autoplay on manual scroll ──────────────────────────────
  useEffect(() => {
    const stop = () => { if (useJourneyStore.getState().autoPlay) useJourneyStore.getState().setAutoPlay(false); };
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    return () => { window.removeEventListener("wheel", stop); window.removeEventListener("touchmove", stop); };
  }, []);

  return (
    <section
      ref={ref}
      aria-label="Cinematic journey"
      className="relative isolate"
      style={{ height: `${totalHeight * 100}vh` }}
    >
      {/* Invisible scroll anchors per chapter */}
      {chapters.map((c, ci) => {
        const stageStart = flat.find((s) => s.chapterIndex === ci);
        const topPct = stageStart ? stageStart.start * 100 : 0;
        return (
          <div
            key={c.id}
            id={c.id}
            aria-hidden
            className="pointer-events-none absolute left-0 right-0"
            style={{ top: `${topPct}%`, height: 1, scrollMarginTop: "5rem" }}
          />
        );
      })}

      <div className="sticky top-0 flex h-screen items-center justify-center overflow-hidden">
        <BackdropAurora progress={scrollYProgress} />

        {!reduced && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-[110vh] w-[110vh] max-w-[100vw]">
              <Suspense fallback={null}>
                <MorphingParticles progress={scrollYProgress} shapes={shapes} />
              </Suspense>
            </div>
          </div>
        )}

        <ChapterIndicator
          chapters={chapters}
          activeChapter={activeChapter}
          progress={scrollYProgress}
          stagesCount={flat.length}
        />

        <div className="container relative z-10 h-full">
          {flat.map((stage) => (
            <StageContent
              key={stage.globalIndex}
              stage={stage}
              progress={scrollYProgress}
              isActive={activeStage === stage.globalIndex}
            />
          ))}
        </div>

        <ChapterProgressDots
          chapters={chapters}
          activeChapter={activeChapter}
          onChapterClick={scrollToChapter}
        />
      </div>
    </section>
  );
}

/* ─────────── Chapter indicator (sticky top bar) ─────────── */

function ChapterIndicator({
  chapters,
  activeChapter,
  progress,
  stagesCount,
}: {
  chapters: JourneyChapter[];
  activeChapter: number;
  progress: MotionValue<number>;
  stagesCount: number;
}) {
  const widthPct = useTransform(progress, (p) => `${p * 100}%`);
  const current = chapters[activeChapter];
  if (!current) return null;

  return (
    <div className="pointer-events-none absolute inset-x-0 top-16 z-30 sm:top-20">
      <div className="container relative flex items-center justify-between gap-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={current.id}
            initial={{ y: 14, opacity: 0, filter: "blur(6px)" }}
            animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
            exit={{ y: -14, opacity: 0, filter: "blur(6px)" }}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            className="flex items-baseline gap-3 rounded-full border border-border/60 bg-background/70 px-4 py-1.5 backdrop-blur-md"
          >
            <span className="font-mono text-xs font-medium tracking-[0.3em] text-primary">
              {current.number}
            </span>
            <span className="text-sm font-semibold tracking-tight sm:text-base">
              {current.name}
            </span>
          </motion.div>
        </AnimatePresence>

        <div className="hidden rounded-full border border-border/60 bg-background/70 px-3 py-1 font-mono text-[10px] tracking-widest text-muted-foreground backdrop-blur-md sm:block">
          {activeChapter + 1} / {chapters.length}
        </div>
      </div>

      <div className="container relative mt-3 h-px overflow-hidden bg-border/30">
        <motion.div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary via-fuchsia-500 to-purple-500"
          style={{ width: widthPct }}
        />
      </div>

      <span className="sr-only">
        Stage {Math.min(stagesCount, activeChapter + 1)} of {stagesCount}
      </span>
    </div>
  );
}

/* ─────────── Chapter dots (right side, clickable) ─────────── */

function ChapterProgressDots({
  chapters,
  activeChapter,
  onChapterClick,
}: {
  chapters: JourneyChapter[];
  activeChapter: number;
  onChapterClick: (index: number) => void;
}) {
  return (
    <div className="absolute right-4 top-1/2 z-20 hidden -translate-y-1/2 flex-col gap-3 lg:flex">
      {chapters.map((c, i) => (
        <button
          key={c.id}
          onClick={() => onChapterClick(i)}
          aria-label={`${c.number} ${c.name}`}
          className="group relative flex items-center justify-end outline-none"
        >
          {/* Tooltip */}
          <span className="pointer-events-none absolute right-full mr-3 whitespace-nowrap rounded-full border border-border/60 bg-background/85 px-2.5 py-1 text-[10px] font-medium tracking-wide backdrop-blur-md opacity-0 transition-all duration-200 group-hover:opacity-100 group-focus-visible:opacity-100">
            <span className="font-mono text-primary">{c.number}</span>
            {" "}{c.name}
          </span>

          {/* Dot */}
          <span
            className={cn(
              "block size-2 rounded-full transition-all duration-500",
              i === activeChapter
                ? "scale-150 bg-primary shadow-[0_0_14px_hsl(var(--primary))]"
                : i < activeChapter
                  ? "bg-primary/60"
                  : "bg-muted-foreground/30",
              "group-hover:scale-[1.8] group-hover:bg-primary group-hover:shadow-[0_0_10px_hsl(var(--primary)/0.7)]",
            )}
          />
        </button>
      ))}
    </div>
  );
}

/* ─────────── Stage content (text + optional embed) ─────────── */

function StageContent({
  stage,
  progress,
  isActive,
}: {
  stage: FlatStage;
  progress: MotionValue<number>;
  isActive: boolean;
}) {
  const range = stage.end - stage.start;
  const fadeIn = stage.start + range * 0.18;
  const fadeOut = stage.end - range * 0.18;

  const isFirst = stage.globalIndex === 0;
  const opacity = useTransform(
    progress,
    isFirst
      ? [stage.start, fadeOut, stage.end]
      : [stage.start, fadeIn, fadeOut, stage.end],
    isFirst ? [1, 1, 0] : [0, 1, 1, 0],
  );

  const xOffset = stage.side === "left" ? -40 : 40;
  const x = useTransform(
    progress,
    [stage.start, fadeIn, fadeOut, stage.end],
    [xOffset, 0, 0, -xOffset],
  );

  const side = stage.side ?? "left";

  if (stage.content) {
    return (
      <motion.div
        style={{ opacity }}
        aria-hidden={!isActive}
        className={cn(
          "absolute inset-0 flex items-center justify-center",
          isActive ? "pointer-events-auto" : "pointer-events-none",
        )}
      >
        <div className="w-full max-w-5xl px-6">{stage.content}</div>
      </motion.div>
    );
  }

  return (
    <motion.div
      style={{ opacity, x }}
      aria-hidden={!isActive}
      className={cn(
        "absolute inset-y-0 flex w-full max-w-md flex-col justify-center",
        side === "left"
          ? "left-6 items-start text-left lg:left-12"
          : "right-6 items-end text-right lg:right-12",
        isActive ? "pointer-events-auto" : "pointer-events-none",
      )}
    >
      {/* Glassmorphism backdrop for readability */}
      <div className="rounded-2xl bg-background/55 px-7 py-6 backdrop-blur-sm ring-1 ring-white/[0.06]">
        {stage.eyebrow && (
          <span className="mb-3 inline-flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.3em] text-primary">
            <span className="size-1 rounded-full bg-primary shadow-[0_0_10px_hsl(var(--primary))]" />
            {stage.eyebrow}
          </span>
        )}

        <h3 className="text-balance text-3xl font-bold leading-tight tracking-tight sm:text-4xl md:text-[2.75rem]">
          {stage.title}
        </h3>

        {stage.body && (
          <p className="mt-4 text-pretty text-base leading-relaxed text-muted-foreground md:text-lg">
            {stage.body}
          </p>
        )}

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
      </div>
    </motion.div>
  );
}

/* ─────────── Aurora backdrop that shifts hue per chapter ─────────── */

function BackdropAurora({ progress }: { progress: MotionValue<number> }) {
  const hue = useTransform(progress, [0, 1], [199, 320]);
  const bg = useTransform(hue, (h) => `hsl(${h} 89% 58% / 0.08)`);
  const glowBg = useTransform(
    hue,
    (h) => `radial-gradient(circle, hsl(${h} 89% 58%), transparent 60%)`,
  );

  return (
    <motion.div
      aria-hidden
      className="pointer-events-none absolute inset-0 -z-10"
      style={{ backgroundColor: bg }}
    >
      <div className="absolute inset-0 grid-bg opacity-25" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_center,transparent_25%,hsl(var(--background))_85%)]" />
      <motion.div
        className="absolute left-1/2 top-1/2 size-[80vh] -translate-x-1/2 -translate-y-1/2 rounded-full opacity-40 blur-3xl"
        style={{ background: glowBg }}
      />
    </motion.div>
  );
}
