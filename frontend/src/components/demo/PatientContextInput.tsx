import { useTranslation } from "react-i18next";

import { Label } from "@/components/ui/label";
import type { ArrivalMode, Sex } from "@/lib/types";
import { useDemoStore } from "@/stores/demoStore";

const selectClass =
  "h-10 w-full rounded-md border border-input bg-background px-3 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

/** Optional request context: sex, arrival mode (ambulance / own transport / other), oxygen at triage. */
export function PatientContextInput() {
  const { t } = useTranslation("sections");
  const { context, setContext } = useDemoStore();

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="sex" className="text-xs text-muted-foreground">{t("demo.form.sex")}</Label>
        <select
          id="sex"
          className={selectClass}
          value={context.sex ?? ""}
          onChange={(e) => setContext({ sex: (e.target.value || undefined) as Sex | undefined })}
        >
          <option value="">{t("demo.form.sexUnknown")}</option>
          <option value="M">{t("demo.form.sexMale")}</option>
          <option value="F">{t("demo.form.sexFemale")}</option>
        </select>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="arrivalMode" className="text-xs text-muted-foreground">{t("demo.form.arrivalMode")}</Label>
        <select
          id="arrivalMode"
          className={selectClass}
          value={context.arrivalMode ?? ""}
          onChange={(e) =>
            setContext({ arrivalMode: (e.target.value || undefined) as ArrivalMode | undefined })
          }
        >
          <option value="">{t("demo.form.arrivalUnknown")}</option>
          <option value="ambulance">{t("demo.form.arrivalAmbulance")}</option>
          <option value="self">{t("demo.form.arrivalSelf")}</option>
          <option value="other">{t("demo.form.arrivalOther")}</option>
        </select>
      </div>
      <label className="flex items-center gap-2 self-end pb-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          className="size-4 accent-primary"
          checked={context.o2Device ?? false}
          onChange={(e) => setContext({ o2Device: e.target.checked ? true : undefined })}
        />
        {t("demo.form.o2Device")}
      </label>
    </div>
  );
}
