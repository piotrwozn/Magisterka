import { useTranslation } from "react-i18next";

import { Textarea } from "@/components/ui/textarea";
import { useDemoStore } from "@/stores/demoStore";

export function ClinicalNoteInput() {
  const { t } = useTranslation("sections");
  const { clinicalNote, setClinicalNote } = useDemoStore();

  return (
    <Textarea
      value={clinicalNote}
      onChange={(e) => setClinicalNote(e.target.value)}
      placeholder={t("demo.form.notePlaceholder")}
      rows={4}
      className="min-h-[100px] resize-y leading-relaxed"
      maxLength={500}
    />
  );
}
