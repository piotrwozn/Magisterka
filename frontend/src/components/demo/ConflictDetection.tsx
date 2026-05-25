import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ConflictInfo } from "@/lib/types";

interface ConflictDetectionProps {
  conflict: ConflictInfo;
}

export function ConflictDetection({ conflict }: ConflictDetectionProps) {
  const { t } = useTranslation("sections");

  const config = conflict.alertDoctor
    ? {
        Icon: ShieldAlert,
        color: "text-destructive",
        bg: "bg-destructive/10 border-destructive/30",
        label: t("demo.result.conflictHigh"),
      }
    : conflict.detected
      ? {
          Icon: AlertTriangle,
          color: "text-amber-500",
          bg: "bg-amber-500/10 border-amber-500/30",
          label: t("demo.result.conflictLow"),
        }
      : {
          Icon: CheckCircle2,
          color: "text-emerald-500",
          bg: "bg-emerald-500/10 border-emerald-500/30",
          label: t("demo.result.noConflict"),
        };

  const { Icon } = config;

  return (
    <Card className={cn("p-5", config.bg)}>
      <div className="flex items-start gap-3">
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 220 }}
        >
          <Icon className={cn("size-5", config.color)} />
        </motion.div>
        <div className="flex-1">
          <h4 className={cn("text-sm font-bold", config.color)}>
            {t("demo.result.conflictTitle")}
          </h4>
          <p className="mt-1 text-sm">{config.label}</p>
          {conflict.message && (
            <p className="mt-1 text-xs text-muted-foreground">{conflict.message}</p>
          )}
        </div>
      </div>
    </Card>
  );
}
