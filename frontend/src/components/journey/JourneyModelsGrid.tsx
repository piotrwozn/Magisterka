import { motion } from "framer-motion";
import { Award, CheckCircle2, Cpu, Hourglass, Loader2, Lock } from "lucide-react";
import { useState } from "react";

import { CountUp } from "@/components/animations/CountUp";
import { TiltCard } from "@/components/animations/TiltCard";
import { ModelDetailModal } from "@/components/journey/ModelDetailModal";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { MODELS, type Model } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function JourneyModelsGrid() {
  const [selected, setSelected] = useState<Model | null>(null);

  const trained = MODELS.filter((m) => m.status === "trained");
  const best = (trained.length > 0 ? trained : MODELS).reduce((a, b) => (a.qwk > b.qwk ? a : b));

  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {MODELS.map((model, idx) => {
          const isBest = model.id === best.id;
          const isTrained = model.status === "trained";
          const isTraining = model.status === "training";
          const isPlanned = model.status === "planned";
          const isClickable = isTrained;

          return (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: idx * 0.05 }}
            >
              <TiltCard intensity={isClickable ? 8 : 3} className="group h-full">
                <Card
                  onClick={() => isClickable && setSelected(model)}
                  className={cn(
                    "relative h-full overflow-hidden p-4 backdrop-blur-md transition-all duration-300",
                    // Trained: luminous border + pointer
                    isTrained && "cursor-pointer border-2",
                    isBest && "border-amber-500/70 bg-card/75 shadow-[0_0_24px_hsl(45_100%_50%/0.12)]",
                    isTrained && !isBest && "border-emerald-500/50 bg-card/75 shadow-[0_0_20px_hsl(142_76%_36%/0.10)] hover:border-emerald-400/70 hover:shadow-[0_0_30px_hsl(142_76%_36%/0.18)]",
                    // Training: amber pulsing border
                    isTraining && "border-amber-500/40 bg-card/65 hover:border-amber-500/60",
                    // Planned: muted
                    isPlanned && "border-border/40 bg-card/50 opacity-70",
                  )}
                  style={isTrained ? { borderColor: isBest ? `${model.color}70` : undefined } : undefined}
                >
                  {/* Trained glow overlay */}
                  {isTrained && (
                    <div
                      className="pointer-events-none absolute inset-0 opacity-5"
                      style={{ background: `radial-gradient(circle at top right, ${model.color}, transparent 70%)` }}
                    />
                  )}

                  {/* Header row */}
                  <div className="relative mb-3 flex items-start justify-between gap-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className={cn("size-2.5 shrink-0 rounded-full", isPlanned && "opacity-40")}
                        style={{ background: model.color, boxShadow: isTrained ? `0 0 8px ${model.color}` : undefined }}
                      />
                      <h3 className="text-sm font-bold leading-tight">{model.name}</h3>
                    </div>
                    <div className="shrink-0">
                      {isBest && <Award className="size-4 text-amber-500 drop-shadow-[0_0_6px_#f59e0b]" />}
                      {isTrained && !isBest && <CheckCircle2 className="size-4 text-emerald-400" />}
                      {isTraining && <Loader2 className="size-4 animate-spin text-amber-400" />}
                      {isPlanned && <Lock className="size-4 text-muted-foreground/40" />}
                    </div>
                  </div>

                  {/* QWK */}
                  <div className="mb-3 space-y-1.5">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs text-muted-foreground">QWK</span>
                      <span className={cn("text-xl font-bold tabular-nums", isPlanned && "text-muted-foreground")}>
                        {isTrained
                          ? <CountUp to={model.qwk} decimals={4} duration={1.6} />
                          : model.qwk.toFixed(4)
                        }
                        {isTraining || isPlanned ? <span className="ml-0.5 text-[10px] text-muted-foreground">est.</span> : null}
                      </span>
                    </div>
                    <Progress
                      value={model.qwk * 100}
                      className={cn("h-1.5", isPlanned && "opacity-40")}
                    />
                  </div>

                  {/* Extra metrics for trained */}
                  {isTrained && model.accuracy !== undefined && (
                    <div className="mb-3 flex gap-3 text-[11px]">
                      <div className="flex flex-col items-center">
                        <span className="text-muted-foreground">ACC</span>
                        <span className="font-bold">{model.accuracy.toFixed(1)}%</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <span className="text-muted-foreground">AUC</span>
                        <span className="font-bold">{model.aucMacro?.toFixed(3)}</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <span className="text-muted-foreground">UT</span>
                        <span className="font-bold">{model.undertriage.toFixed(1)}%</span>
                      </div>
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex items-center justify-between border-t border-border/40 pt-2 text-[10px] text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Cpu className="size-2.5" />{model.device}
                    </span>
                    <span className="flex items-center gap-1">
                      <Hourglass className="size-2.5" />{model.trainTimeMin}m
                    </span>
                  </div>

                  {/* Click hint */}
                  {isClickable && (
                    <div className="absolute inset-x-0 bottom-0 translate-y-full rounded-b-lg bg-primary/90 py-1 text-center text-[10px] font-semibold text-primary-foreground transition-transform duration-200 group-hover:translate-y-0">
                      Kliknij po szczegóły
                    </div>
                  )}
                </Card>
              </TiltCard>
            </motion.div>
          );
        })}
      </div>

      <ModelDetailModal model={selected} onClose={() => setSelected(null)} />
    </>
  );
}
