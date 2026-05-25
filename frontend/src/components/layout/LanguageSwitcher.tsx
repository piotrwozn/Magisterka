import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Language } from "@/lib/types";

const LANGS: { code: Language; label: string; flag: string }[] = [
  { code: "pl", label: "PL", flag: "🇵🇱" },
  { code: "en", label: "EN", flag: "🇬🇧" },
];

export function LanguageSwitcher() {
  const { i18n } = useTranslation();
  const current = (i18n.resolvedLanguage ?? i18n.language ?? "pl").slice(0, 2) as Language;

  return (
    <div className="flex items-center gap-1 rounded-full border border-border bg-card/50 p-0.5">
      {LANGS.map((lang) => (
        <Button
          key={lang.code}
          variant="ghost"
          size="sm"
          onClick={() => void i18n.changeLanguage(lang.code)}
          aria-pressed={current === lang.code}
          aria-label={`Switch to ${lang.label}`}
          className={cn(
            "h-7 rounded-full px-2.5 text-xs font-semibold transition-all",
            current === lang.code
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <span className="hidden text-[10px] sm:inline">{lang.flag}</span>
          <span>{lang.label}</span>
        </Button>
      ))}
    </div>
  );
}
