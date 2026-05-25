import { Check, Copy, GraduationCap, Mail } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const AUTHOR = {
  name: "inż. Piotr Woźnicki",
  email: "piotrwozn@gmail.com",
};

export function JourneyAbout() {
  const { t } = useTranslation("sections");
  const [copied, setCopied] = useState(false);

  const copyEmail = async () => {
    await navigator.clipboard.writeText(AUTHOR.email);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <Card className="bg-card/75 p-8 text-center backdrop-blur-md">
      <div className="mx-auto mb-6 inline-flex size-14 items-center justify-center rounded-2xl bg-primary/15 text-primary">
        <GraduationCap className="size-6" />
      </div>

      <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
        {t("about.thesis")}
      </p>
      <h3 className="mt-2 text-balance text-xl font-bold leading-snug md:text-2xl">
        {t("about.thesisTitle")}
      </h3>

      <p className="mx-auto mt-6 max-w-md text-sm text-muted-foreground">
        {t("about.disclaimer")}
      </p>

      <blockquote className="mx-auto mt-6 max-w-sm border-l-2 border-primary/50 pl-4 text-left">
        <p className="text-sm italic leading-relaxed text-foreground/80">
          "Build things that matter.<br />Make them reliable enough to trust."
        </p>
        <footer className="mt-1.5 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          — Piotr Woźnicki
        </footer>
      </blockquote>

      <div className="mx-auto mt-6 flex max-w-sm flex-col gap-2">
        <Button asChild variant="outline">
          <a href={`mailto:${AUTHOR.email}`}>
            <Mail className="size-4" />
            {AUTHOR.email}
          </a>
        </Button>
        <Button variant="ghost" onClick={copyEmail} className="text-muted-foreground">
          {copied ? <Check className="size-4 text-emerald-500" /> : <Copy className="size-4" />}
          {copied ? "Skopiowano" : "Skopiuj adres"}
        </Button>
      </div>

      <div className="mt-6 border-t border-border pt-4">
        <p className="text-xs text-muted-foreground">{AUTHOR.name}</p>
      </div>
    </Card>
  );
}
