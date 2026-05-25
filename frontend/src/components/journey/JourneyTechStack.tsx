import { motion } from "framer-motion";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { TechDetailModal } from "@/components/journey/TechDetailModal";
import { Card } from "@/components/ui/card";
import { TECH_STACK, type TechItem } from "@/lib/constants";

const GROUP_KEYS = ["frontend", "backend", "ml", "llm"] as const;

export function JourneyTechStack() {
  const { t } = useTranslation("sections");
  const [selected, setSelected] = useState<{ tech: TechItem; category: string } | null>(null);

  const titleKeys: Record<string, string> = {
    frontend: "techStack.frontend",
    backend:  "techStack.backend",
    ml:       "techStack.ml",
    llm:      "techStack.llm",
  };

  return (
    <>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {GROUP_KEYS.map((groupKey) => (
          <Card key={groupKey} className="bg-card/65 p-4 backdrop-blur-md">
            <h3 className="mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-primary">
              {t(titleKeys[groupKey]!)}
            </h3>
            <ul className="space-y-1.5">
              {TECH_STACK[groupKey]!.map((tech, idx) => (
                <motion.li
                  key={tech.name}
                  initial={{ opacity: 0, x: -6 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <button
                    onClick={() => setSelected({ tech, category: groupKey })}
                    className="group flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left transition-colors hover:bg-accent/60"
                  >
                    <span
                      className="size-1.5 shrink-0 rounded-full transition-all group-hover:scale-125"
                      style={{
                        backgroundColor: tech.color,
                        boxShadow: `0 0 8px ${tech.color}80`,
                      }}
                    />
                    <span className="truncate text-xs text-foreground/85 group-hover:text-foreground">
                      {tech.name}
                    </span>
                  </button>
                </motion.li>
              ))}
            </ul>
          </Card>
        ))}
      </div>

      <TechDetailModal
        tech={selected?.tech ?? null}
        category={selected?.category ?? ""}
        onClose={() => setSelected(null)}
      />
    </>
  );
}
