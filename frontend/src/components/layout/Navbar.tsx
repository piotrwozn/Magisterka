import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, Activity, Play, Pause } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { NAV_LINKS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { useJourneyStore } from "@/stores/journeyStore";

import { LanguageSwitcher } from "./LanguageSwitcher";
import { ThemeToggle } from "./ThemeToggle";

function scrollToAnchor(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const top = window.scrollY + rect.top - 80;
  window.scrollTo({ top, behavior: "smooth" });
}

export function Navbar() {
  const { t } = useTranslation();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { autoPlay, toggleAutoPlay } = useJourneyStore();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [location]);

  const handleNavClick = useCallback(
    (e: React.MouseEvent, anchor: string) => {
      e.preventDefault();
      if (location.pathname !== "/") {
        navigate(`/#${anchor}`);
        setTimeout(() => scrollToAnchor(anchor), 80);
      } else {
        scrollToAnchor(anchor);
        if (history.replaceState) history.replaceState(null, "", `#${anchor}`);
      }
    },
    [location.pathname, navigate],
  );

  return (
    <header
      className={cn(
        "fixed left-0 right-0 top-0 z-50 w-full transition-all duration-300",
        scrolled ? "border-b border-border/40 bg-background/70 backdrop-blur-xl" : "bg-transparent",
      )}
    >
      <div className="container flex h-16 items-center justify-between gap-4">
        <Link to="/" className="group flex items-center gap-2.5">
          <div className="relative flex size-8 items-center justify-center">
            <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-primary to-purple-500 opacity-90 transition-all group-hover:opacity-100 group-hover:scale-105" />
            <Activity className="relative size-4 text-primary-foreground" strokeWidth={2.5} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-lg font-bold tracking-tight">SOR-AI</span>
            <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
              triage AI
            </span>
          </div>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map((link) => (
            <a
              key={link.anchor}
              href={link.href}
              onClick={(e) => handleNavClick(e, link.anchor)}
              className="relative rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {t(link.labelKey)}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <LanguageSwitcher />
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleAutoPlay}
            title={autoPlay ? "Zatrzymaj auto-scroll" : "Auto-scroll"}
            aria-label={autoPlay ? "Zatrzymaj auto-scroll" : "Auto-scroll"}
            className={cn(autoPlay && "text-primary")}
          >
            {autoPlay ? <Pause className="size-4" /> : <Play className="size-4" />}
          </Button>
          <ThemeToggle />
          <Button
            size="sm"
            variant="glow"
            className="hidden md:inline-flex"
            onClick={(e) => handleNavClick(e, "demo")}
          >
            {t("nav.demo")}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </Button>
        </div>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-b border-border bg-background md:hidden"
          >
            <nav className="container flex flex-col gap-1 py-4">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.anchor}
                  href={link.href}
                  onClick={(e) => handleNavClick(e, link.anchor)}
                  className="rounded-md px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  {t(link.labelKey)}
                </a>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
