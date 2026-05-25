import { motion } from "framer-motion";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useDemoStore } from "@/stores/demoStore";
import { VITALS_RANGES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Vitals } from "@/lib/types";

type VitalKey = keyof Vitals;

const FIELDS: { key: VitalKey; labelKey: string }[] = [
  { key: "age", labelKey: "demo.form.age" },
  { key: "temp", labelKey: "demo.form.temp" },
  { key: "hr", labelKey: "demo.form.hr" },
  { key: "sbp", labelKey: "demo.form.sbp" },
  { key: "dbp", labelKey: "demo.form.dbp" },
  { key: "rr", labelKey: "demo.form.rr" },
  { key: "o2", labelKey: "demo.form.o2" },
];

import { useTranslation } from "react-i18next";

export function VitalsInput() {
  const { t } = useTranslation("sections");
  const { vitals, setVitals } = useDemoStore();

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {FIELDS.map((field, idx) => {
        const range = VITALS_RANGES[field.key];
        const val = vitals[field.key];
        const outOfRange =
          (field.key === "sbp" && val < 100) ||
          (field.key === "hr" && (val > 110 || val < 50)) ||
          (field.key === "o2" && val < 95) ||
          (field.key === "temp" && (val >= 38 || val < 35)) ||
          (field.key === "rr" && (val > 22 || val < 12));

        return (
          <motion.div
            key={field.key}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.04 }}
            className="flex flex-col gap-1.5"
          >
            <Label
              htmlFor={field.key}
              className="flex h-[2.25rem] items-end justify-between gap-1 text-xs leading-tight"
            >
              <span className="text-muted-foreground line-clamp-2">{t(field.labelKey)}</span>
              <span className="shrink-0 font-mono text-[10px] text-muted-foreground/60">
                {range.unit}
              </span>
            </Label>
            <Input
              id={field.key}
              type="number"
              value={val}
              onChange={(e) => {
                const next = Number(e.target.value);
                if (!Number.isNaN(next)) setVitals({ [field.key]: next });
              }}
              min={range.min}
              max={range.max}
              step={"step" in range ? range.step : 1}
              className={cn(
                "h-10 font-mono tabular-nums transition-all",
                outOfRange && "border-amber-500/50 bg-amber-500/5 text-amber-500",
              )}
            />
          </motion.div>
        );
      })}
    </div>
  );
}
