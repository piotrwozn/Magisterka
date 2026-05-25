import { GraduationCap, Mail, AlertTriangle, Copy, Check } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { FadeInOnScroll } from "@/components/animations/FadeInOnScroll";
import { SectionHeader } from "@/components/shared/SectionHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const AUTHOR = {
  name: "inż. Piotr Woźnicki",
  email: "piotrwozn@gmail.com",
};

export function About() {
  const { t } = useTranslation("sections");
  const [copied, setCopied] = useState(false);

  const copyEmail = async () => {
    await navigator.clipboard.writeText(AUTHOR.email);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <section id="about" className="relative py-24 md:py-32">
      <div className="container space-y-14">
        <FadeInOnScroll>
          <SectionHeader
            eyebrow="About"
            title={t("about.title")}
            subtitle={t("about.subtitle")}
          />
        </FadeInOnScroll>

        <div className="mx-auto grid max-w-5xl gap-5 lg:grid-cols-5">
          <FadeInOnScroll className="lg:col-span-3">
            <Card className="h-full p-8">
              <div className="mb-6 inline-flex size-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <GraduationCap className="size-5" />
              </div>

              <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                {t("about.thesis")}
              </p>
              <h3 className="mt-1.5 text-pretty text-xl font-semibold leading-snug md:text-2xl">
                {t("about.thesisTitle")}
              </h3>

              <dl className="mt-8 border-t border-border pt-6">
                <Detail term={t("about.author")} value={AUTHOR.name} />
              </dl>
            </Card>
          </FadeInOnScroll>

          <FadeInOnScroll delay={0.1} className="lg:col-span-2">
            <Card className="flex h-full flex-col justify-between p-8">
              <div>
                <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                  {t("about.contact")}
                </p>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  W sprawach związanych z pracą, kodem lub współpracą badawczą — proszę o kontakt
                  mailowy.
                </p>
              </div>

              <div className="mt-6 space-y-2">
                <Button asChild variant="outline" className="w-full justify-start">
                  <a href={`mailto:${AUTHOR.email}`}>
                    <Mail className="size-4" />
                    {AUTHOR.email}
                  </a>
                </Button>
                <Button
                  variant="ghost"
                  className="w-full justify-start text-muted-foreground"
                  onClick={copyEmail}
                >
                  {copied ? (
                    <Check className="size-4 text-emerald-500" />
                  ) : (
                    <Copy className="size-4" />
                  )}
                  {copied ? "Skopiowano" : "Skopiuj adres"}
                </Button>
              </div>
            </Card>
          </FadeInOnScroll>
        </div>

        <FadeInOnScroll delay={0.15}>
          <div className="mx-auto flex max-w-4xl gap-4 rounded-lg border-l-2 border-amber-500/50 bg-amber-500/[0.04] p-5">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" />
            <p className="text-sm leading-relaxed text-muted-foreground">{t("about.disclaimer")}</p>
          </div>
        </FadeInOnScroll>
      </div>
    </section>
  );
}

function Detail({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wider text-muted-foreground">{term}</dt>
      <dd className="mt-1 text-base font-medium">{value}</dd>
    </div>
  );
}
