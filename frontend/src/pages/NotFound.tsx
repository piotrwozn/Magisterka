import { ArrowLeft } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export function NotFound() {
  const { t } = useTranslation("sections");
  return (
    <div className="container flex min-h-[70vh] flex-col items-center justify-center gap-6 text-center">
      <h1 className="text-9xl font-bold gradient-text">{t("notFound.title")}</h1>
      <p className="text-lg text-muted-foreground">{t("notFound.description")}</p>
      <Button asChild variant="glow" size="lg">
        <Link to="/">
          <ArrowLeft className="size-4" />
          {t("notFound.back")}
        </Link>
      </Button>
    </div>
  );
}
