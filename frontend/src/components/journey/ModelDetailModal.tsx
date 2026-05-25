import { AnimatePresence, motion } from "framer-motion";
import { X, Cpu, Clock, TrendingUp, AlertTriangle, ShieldAlert, Info } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import type { Model } from "@/lib/constants";

interface ModelDetailModalProps {
  model: Model | null;
  onClose: () => void;
}

function MetricRow({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-border/40 py-2 last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-right font-mono text-sm font-bold tabular-nums">
        {value}
        {sub && <span className="ml-1 text-xs font-normal text-muted-foreground">{sub}</span>}
      </span>
    </div>
  );
}

export function ModelDetailModal({ model, onClose }: ModelDetailModalProps) {
  const { t } = useTranslation("sections");
  const m = (key: string) => t(`models.modal.${key}`);
  const mt = (key: string) => t(`models.modal.metrics.${key}`);
  return (
    <AnimatePresence>
      {model && (
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 24 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 24 }}
            transition={{ type: "spring", damping: 24, stiffness: 280 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl max-h-[85vh] overflow-y-auto"
          >
            <Card
              className="relative overflow-hidden border-2 bg-background/95 p-6 backdrop-blur-xl"
              style={{ borderColor: `${model.color}50` }}
            >
              {/* Glow */}
              <div
                className="pointer-events-none absolute -right-24 -top-24 size-64 rounded-full opacity-15 blur-3xl"
                style={{ background: model.color }}
              />

              {/* Header */}
              <div className="relative mb-5 flex items-start justify-between gap-4">
                <div>
                  <div className="mb-1 flex items-center gap-2">
                    <span
                      className="size-3 rounded-full"
                      style={{ background: model.color, boxShadow: `0 0 10px ${model.color}` }}
                    />
                    <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                      {model.type}
                    </span>
                    <StatusBadge status={model.status} />
                  </div>
                  <h2 className="text-2xl font-bold tracking-tight">{model.name}</h2>
                  <p className="mt-1 text-sm text-muted-foreground">{model.description}</p>
                </div>
                <button
                  onClick={onClose}
                  className="shrink-0 rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <X className="size-4" />
                </button>
              </div>

              <div className="relative grid gap-4 sm:grid-cols-2">
                {/* ── Metryki wydajności ── */}
                <Card className="bg-card/60 p-4">
                  <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-primary">
                    <TrendingUp className="size-3.5" /> {m("performance")}
                  </p>
                  <MetricRow label={mt("qwk")} value={model.qwk.toFixed(4)} />
                  {model.accuracy !== undefined && (
                    <MetricRow label={mt("accuracy")} value={`${model.accuracy.toFixed(2)}%`} />
                  )}
                  {model.f1Macro !== undefined && (
                    <MetricRow label={mt("f1macro")} value={model.f1Macro.toFixed(4)} />
                  )}
                  {model.aucMacro !== undefined && (
                    <MetricRow label={mt("aucmacro")} value={model.aucMacro.toFixed(4)} />
                  )}
                  {model.cohenKappa !== undefined && (
                    <MetricRow label={mt("cohenkappa")} value={model.cohenKappa.toFixed(4)} />
                  )}
                </Card>

                {/* ── Metryki medyczne ── */}
                <Card className="bg-card/60 p-4">
                  <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-amber-500">
                    <AlertTriangle className="size-3.5" /> {m("safety")}
                  </p>
                  <MetricRow label={mt("undertriage")} value={`${model.undertriage.toFixed(2)}%`} />
                  <MetricRow label={mt("criticalmiss")} value={`${model.criticalMiss.toFixed(3)}%`} />
                  <div className="mt-3 border-t border-border/40 pt-3">
                    <p className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      <Cpu className="size-3" /> {m("hardware")}
                    </p>
                    <MetricRow label={mt("device")} value={model.device} />
                    <MetricRow label={mt("traintime")} value={`~${model.trainTimeMin} min`} />
                  </div>
                </Card>

                {/* ── Recall per klasa MTS ── */}
                {model.classRecalls && model.classRecalls.length > 0 && (
                  <Card className="bg-card/60 p-4 sm:col-span-2">
                    <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      <ShieldAlert className="size-3.5" /> {m("classRecalls")}
                    </p>
                    <div className="space-y-2.5">
                      {model.classRecalls.map((r) => (
                        <div key={r.label}>
                          <div className="mb-1 flex justify-between text-xs">
                            <span className="font-medium" style={{ color: r.color }}>{r.label}</span>
                            <span className="font-mono tabular-nums">{r.value.toFixed(1)}%</span>
                          </div>
                          <div className="relative h-2 overflow-hidden rounded-full bg-muted">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${r.value}%` }}
                              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                              className="h-full rounded-full"
                              style={{ background: r.color }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {/* ── Hiperparametry ── */}
                {model.params && Object.keys(model.params).length > 0 && (
                  <Card className="bg-card/60 p-4 sm:col-span-2">
                    <p className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      <Clock className="size-3.5" /> {m("params")}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(model.params).map(([k, v]) => (
                        <span
                          key={k}
                          className="rounded-md border border-border/60 bg-muted/50 px-2.5 py-1 font-mono text-xs"
                        >
                          <span className="text-muted-foreground">{k}=</span>
                          <span className="font-semibold">{typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}</span>
                        </span>
                      ))}
                    </div>
                  </Card>
                )}

                {/* ── Notatki ── */}
                {model.notes && (
                  <Card className="bg-card/60 p-4 sm:col-span-2">
                    <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      <Info className="size-3.5" /> {m("notes")}
                    </p>
                    <p className="text-sm leading-relaxed text-muted-foreground">{model.notes}</p>
                  </Card>
                )}
              </div>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function StatusBadge({ status }: { status: Model["status"] }) {
  const { t } = useTranslation("sections");
  if (status === "trained") {
    return (
      <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
        {t("models.modal.status.trained")}
      </span>
    );
  }
  if (status === "training") {
    return (
      <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-semibold text-amber-400">
        {t("models.modal.status.training")}
      </span>
    );
  }
  return (
    <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground">
      {t("models.modal.status.planned")}
    </span>
  );
}
