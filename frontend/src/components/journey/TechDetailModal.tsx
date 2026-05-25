import { AnimatePresence, motion } from "framer-motion";
import { X, Lightbulb, Wrench, MapPin } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import type { TechItem } from "@/lib/constants";

interface TechDetailModalProps {
  tech: TechItem | null;
  category: string;
  onClose: () => void;
}

const CATEGORY_LABELS: Record<string, { pl: string; en: string }> = {
  frontend:  { pl: "Frontend",           en: "Frontend" },
  backend:   { pl: "Backend",            en: "Backend" },
  ml:        { pl: "Machine Learning",   en: "Machine Learning" },
  llm:       { pl: "Modele językowe",    en: "Language Models" },
};

export function TechDetailModal({ tech, category, onClose }: TechDetailModalProps) {
  const { i18n } = useTranslation();
  const lang = (i18n.resolvedLanguage ?? "pl").slice(0, 2);
  const catLabel = CATEGORY_LABELS[category]?.[lang as "pl" | "en"] ?? category;

  return (
    <AnimatePresence>
      {tech && (
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
            className="w-full max-w-lg"
          >
            <Card
              className="relative overflow-hidden border-2 bg-background/95 p-6 backdrop-blur-xl"
              style={{ borderColor: `${tech.color}50` }}
            >
              {/* Glow */}
              <div
                className="pointer-events-none absolute -right-20 -top-20 size-52 rounded-full opacity-10 blur-3xl"
                style={{ background: tech.color }}
              />

              {/* Header */}
              <div className="relative mb-5 flex items-start justify-between gap-4">
                <div>
                  <div className="mb-1.5 flex items-center gap-2">
                    <span
                      className="size-3 rounded-full"
                      style={{ background: tech.color, boxShadow: `0 0 10px ${tech.color}` }}
                    />
                    <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                      {catLabel}
                      {tech.version && (
                        <span className="ml-1.5 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
                          v{tech.version}
                        </span>
                      )}
                    </span>
                  </div>
                  <h2 className="text-xl font-bold tracking-tight">{tech.name}</h2>
                  <p className="mt-1 text-sm text-muted-foreground">{tech.role}</p>
                </div>
                <button
                  onClick={onClose}
                  className="shrink-0 rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <X className="size-4" />
                </button>
              </div>

              {/* Details */}
              <div className="relative space-y-3">
                <DetailBlock
                  icon={<Lightbulb className="size-4 text-amber-400" />}
                  label={lang === "en" ? "Why" : "Dlaczego"}
                  text={tech.why}
                  accent="amber"
                />
                <DetailBlock
                  icon={<Wrench className="size-4 text-sky-400" />}
                  label={lang === "en" ? "How" : "Jak"}
                  text={tech.how}
                  accent="sky"
                />
                <DetailBlock
                  icon={<MapPin className="size-4 text-emerald-400" />}
                  label={lang === "en" ? "Where" : "Gdzie"}
                  text={tech.where}
                  accent="emerald"
                />
              </div>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function DetailBlock({
  icon,
  label,
  text,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  text: string;
  accent: "amber" | "sky" | "emerald";
}) {
  const border = {
    amber:   "border-amber-500/30 bg-amber-500/5",
    sky:     "border-sky-500/30 bg-sky-500/5",
    emerald: "border-emerald-500/30 bg-emerald-500/5",
  }[accent];

  return (
    <div className={`rounded-xl border p-4 ${border}`}>
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <span className="text-xs font-bold uppercase tracking-widest">{label}</span>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">{text}</p>
    </div>
  );
}
