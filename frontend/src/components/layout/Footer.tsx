import { Mail, Activity } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

export function Footer() {
  const { t } = useTranslation();
  const year = new Date().getFullYear();

  return (
    <footer className="mt-24 border-t border-border bg-card/30">
      <div className="container py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
          <div className="space-y-3">
            <Link to="/" className="flex items-center gap-2.5">
              <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-purple-500">
                <Activity className="size-4 text-primary-foreground" strokeWidth={2.5} />
              </div>
              <span className="text-lg font-bold">SOR-AI</span>
            </Link>
            <p className="max-w-xs text-sm text-muted-foreground">{t("footer.tagline")}</p>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">{t("footer.thesis")}</h3>
            <ul className="space-y-1.5 text-sm text-muted-foreground">
              <li>
                <Link to="/" className="hover:text-foreground">
                  {t("nav.home")}
                </Link>
              </li>
              <li>
                <Link to="/demo" className="hover:text-foreground">
                  {t("nav.demo")}
                </Link>
              </li>
              <li>
                <Link to="/models" className="hover:text-foreground">
                  {t("nav.models")}
                </Link>
              </li>
              <li>
                <Link to="/about" className="hover:text-foreground">
                  {t("nav.about")}
                </Link>
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Kontakt</h3>
            <a
              href="mailto:piotrwozn@gmail.com"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <Mail className="size-4" /> piotrwozn@gmail.com
            </a>
            <p className="text-xs text-muted-foreground/70">inż. Piotr Woźnicki</p>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-3 border-t border-border pt-6 sm:flex-row">
          <p className="text-xs text-muted-foreground">
            © {year} SOR-AI · {t("footer.rights")}
          </p>
          <p className="text-center text-xs italic text-muted-foreground/60">
            "Build things that matter. Make them reliable enough to trust."
          </p>
          <p className="text-xs text-muted-foreground">Praca magisterska · 2027</p>
        </div>
      </div>
    </footer>
  );
}
