import { Dice5, Loader2, Send } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ClinicalNoteInput } from "@/components/demo/ClinicalNoteInput";
import { VitalsInput } from "@/components/demo/VitalsInput";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { usePredict } from "@/hooks/usePredict";
import { EXAMPLE_PATIENTS } from "@/lib/mockData";
import { useDemoStore } from "@/stores/demoStore";

export function PatientForm() {
  const { t } = useTranslation("sections");
  const { vitals, clinicalNote, setResult, reset, loadExample } = useDemoStore();
  const { mutate, isPending } = usePredict();

  const handleSubmit = () => {
    mutate(
      { vitals, clinicalNote },
      {
        onSuccess: (data) => setResult(data),
      },
    );
  };

  const handleRandom = () => {
    const idx = Math.floor(Math.random() * EXAMPLE_PATIENTS.length);
    const ex = EXAMPLE_PATIENTS[idx]!;
    loadExample({ ...ex.vitals }, ex.clinicalNote);
  };

  return (
    <div className="space-y-5">
      <Card className="p-5">
        <h3 className="mb-4 text-sm font-bold uppercase tracking-wider text-primary">
          {t("demo.form.vitalsTitle")}
        </h3>
        <VitalsInput />
      </Card>

      <Card className="p-5">
        <h3 className="mb-3 text-sm font-bold uppercase tracking-wider text-primary">
          {t("demo.form.noteTitle")}
        </h3>
        <ClinicalNoteInput />
      </Card>

      <Card className="p-5">
        <h3 className="mb-3 text-sm font-bold uppercase tracking-wider text-primary">
          {t("demo.form.examples")}
        </h3>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_PATIENTS.map((ex) => (
            <Button
              key={ex.nameKey}
              variant="outline"
              size="sm"
              onClick={() => loadExample({ ...ex.vitals }, ex.clinicalNote)}
            >
              {t(ex.nameKey)}
            </Button>
          ))}
        </div>
      </Card>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={handleSubmit} disabled={isPending} size="lg" variant="glow" className="flex-1">
          {isPending ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              {t("demo.form.loading")}
            </>
          ) : (
            <>
              <Send className="size-4" />
              {t("demo.form.predict")}
            </>
          )}
        </Button>
        <Button onClick={handleRandom} variant="outline" size="lg">
          <Dice5 className="size-4" />
          {t("demo.form.random")}
        </Button>
        <Button onClick={reset} variant="ghost" size="lg">
          {t("demo.form.reset")}
        </Button>
      </div>
    </div>
  );
}
