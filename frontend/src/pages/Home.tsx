import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import { JourneyAbout } from "@/components/journey/JourneyAbout";
import { JourneyDataPanels } from "@/components/journey/JourneyDataPanels";
import { JourneyDemo } from "@/components/journey/JourneyDemo";
import { JourneyHero } from "@/components/journey/JourneyHero";
import { JourneyModelsGrid } from "@/components/journey/JourneyModelsGrid";
import { JourneyResults } from "@/components/journey/JourneyResults";
import { JourneyTechStack } from "@/components/journey/JourneyTechStack";
import { JourneyTimeline } from "@/components/journey/JourneyTimeline";
import { CinematicJourney, type JourneyChapter } from "@/components/sections/CinematicJourney";
import { MODELS, RESULTS } from "@/lib/constants";
import { formatMetric } from "@/lib/utils";

function buildChapters(lang: "pl" | "en"): JourneyChapter[] {
  const isPl = lang === "pl";

  return [
    {
      id: "intro",
      number: "00",
      name: isPl ? "Wprowadzenie" : "Intro",
      stages: [
        {
          shape: "neuralNet",
          title: "",
          heightFactor: 0.9,
          content: <JourneyHero />,
        },
      ],
    },
    {
      id: "problem",
      number: "01",
      name: isPl ? "Problem" : "The problem",
      stages: [
        {
          eyebrow: isPl ? "Codzienność SOR" : "Reality of ED",
          title: isPl
            ? "8,3 mln wizyt rocznie. 90 sekund na decyzję."
            : "8.3M visits a year. 90 seconds per decision.",
          body: isPl
            ? "Pielęgniarka triażowa klasyfikuje pacjenta w warunkach silnego stresu, z niepełnymi danymi, średnio w 90 sekund. Każda pomyłka ma cenę."
            : "Triage nurses classify patients under severe stress with incomplete data, in roughly 90 seconds. Every mistake has a price.",
          side: "left",
          shape: "vortex",
          stat: { value: "90s", label: isPl ? "na pacjenta" : "per patient" },
        },
        {
          eyebrow: "Emergency Severity Index",
          title: isPl ? "Pięć poziomów ESI. Jedna decyzja." : "Five ESI levels. One verdict.",
          body: isPl
            ? "ESI 1–5, w TOPSOR wyświetlane jako czerwony, pomarańczowy, żółty, zielony i niebieski (maks. 0 / 10 / 60 / 120 / 240 min). ESI 1 to tylko 0,9% wizyt — 47× rzadziej niż ESI 3. Zaniżenie pilności kosztuje 3× więcej niż jej zawyżenie."
            : "ESI 1–5, shown in TOPSOR as red, orange, yellow, green and blue (max 0 / 10 / 60 / 120 / 240 min). ESI 1 is only 0.9% of visits — 47× rarer than ESI 3. Undertriage carries 3× the clinical cost of overtriage.",
          side: "right",
          shape: "starburst",
          stat: { value: "0.9%", label: "ESI 1" },
        },
      ],
    },
    {
      id: "data",
      number: "02",
      name: isPl ? "Dane" : "Data",
      stages: [
        {
          eyebrow: "Yale EMMLC",
          title: isPl
            ? "558 027 wizyt SOR. Tylko dane z triażu."
            : "558,027 ED visits. Triage data only.",
          body: isPl
            ? "Anonimizowany zbiór z Yale-New Haven Health (3 oddziały). Model widzi to, co pielęgniarka w chwili triażu i co jest dostępne w każdym SOR: parametry życiowe, wiek, płeć, sposób przybycia, 199 kategorii skarg głównych — bez cech typowo amerykańskich (np. ubezpieczenia)."
            : "Anonymized dataset from Yale-New Haven Health (3 sites). The model sees what the nurse knows at triage and what any ED has: vital signs, age, sex, arrival mode, 199 chief-complaint categories — no US-specific features (e.g. insurance).",
          side: "left",
          shape: "lattice2D",
          stat: { value: "558k", label: isPl ? "wizyt" : "visits" },
        },
        {
          shape: "cube",
          title: "",
          heightFactor: 1.4,
          content: <JourneyDataPanels />,
        },
      ],
    },
    {
      id: "pipeline",
      number: "03",
      name: isPl ? "Architektura pipeline'u" : "Pipeline architecture",
      stages: [
        {
          eyebrow: isPl ? "Layer 0 · Parser" : "Layer 0 · Parser",
          title: isPl ? "Llama 3.2 strukturyzuje tekst" : "Llama 3.2 structures raw text",
          body: isPl
            ? "Surowy opis pielęgniarki → ustrukturyzowany JSON. Structured Output gwarantuje poprawny format. Lokalnie, poniżej sekundy."
            : "Nurse's raw note → structured JSON. Structured Output guarantees valid format. On-premise, under a second.",
          side: "right",
          shape: "helix",
          stat: { value: "<1s", label: isPl ? "lokalnie" : "on-premise" },
        },
        {
          eyebrow: isPl ? "Layer 1A · ML" : "Layer 1A · ML",
          title: isPl ? "Ensemble sześciu modeli" : "Ensemble of six models",
          body: isPl
            ? "XGBoost, LightGBM, CatBoost, Random Forest, EBM, LogReg. Ważona średnia prawdopodobieństw ESI 1–5 z wagami dobranymi na walidacji. SHAP top-5 cech dla każdej predykcji."
            : "XGBoost, LightGBM, CatBoost, Random Forest, EBM, LogReg. Weighted average of ESI 1–5 probabilities, weights chosen on validation. SHAP top-5 features per prediction.",
          side: "left",
          shape: "constellation",
          stat: { value: String(MODELS.length), label: isPl ? "modeli" : "models" },
        },
        {
          eyebrow: isPl ? "Layer 1B · NLP" : "Layer 1B · NLP",
          title: isPl ? "MedGemma rozumie notatkę" : "MedGemma reads the note",
          body: isPl
            ? "27 miliardów parametrów trenowanych na PubMed i notatkach klinicznych. Wychwytuje sygnały, których dane tabelaryczne nie zawierają."
            : "27B parameters trained on PubMed and clinical notes. Catches signals tabular data misses.",
          side: "right",
          shape: "sphere",
          stat: { value: "27B", label: isPl ? "parametrów" : "parameters" },
        },
        {
          eyebrow: isPl ? "Layer 2 · Synteza" : "Layer 2 · Synthesis",
          title: isPl ? "Qwen3 rozstrzyga konflikt" : "Qwen3 resolves conflict",
          body: isPl
            ? "Łączy wynik tabelaryczny z oceną kliniczną. Zasada nadrzędna: eskaluj, nigdy nie obniżaj. Różnica ≥ 2 poziomów ESI → alert lekarski."
            : "Combines tabular and clinical scores. Master rule: escalate, never downgrade. Gap ≥ 2 ESI levels → doctor alert.",
          side: "left",
          shape: "pyramid",
          stat: { value: "Δ≥2", label: "alert" },
        },
      ],
    },
    {
      id: "models",
      number: "04",
      name: isPl ? "Modele" : "Models",
      stages: [
        {
          shape: "grid",
          title: "",
          heightFactor: 1.4,
          content: <JourneyModelsGrid />,
        },
      ],
    },
    {
      id: "tuning",
      number: "05",
      name: isPl ? "Strojenie" : "Tuning",
      stages: [
        {
          eyebrow: "Optuna",
          title: isPl
            ? "MultivariateTPE · k-fold CV · wznawianie"
            : "MultivariateTPE · k-fold CV · resumable",
          body: isPl
            ? "Jedno polecenie trenuje wszystkie modele: GPU tam, gdzie pomaga (XGBoost, CatBoost), CPU z ostrzeżeniem, gdy GPU zawiedzie. Budżety prób i czasu wg profilu, SQLite storage — wznowienie po awarii lub restarcie."
            : "One command trains every model: GPU where it helps (XGBoost, CatBoost), CPU with a warning when the GPU fails. Trial and time budgets per profile, SQLite storage — resume after a crash or reboot.",
          side: "right",
          shape: "torus",
          stat: { value: "GPU", label: isPl ? "+ CPU fallback" : "+ CPU fallback" },
        },
      ],
    },
    {
      id: "results",
      number: "06",
      name: isPl ? "Wyniki" : "Outcomes",
      stages: [
        {
          eyebrow: isPl ? "Holdout test set" : "Holdout test set",
          title: RESULTS
            ? `QWK ${formatMetric(RESULTS.qwk, 4)}. Critical miss ${formatMetric(RESULTS.criticalMiss, 2, "%")}.`
            : isPl ? "Wyniki ESI po treningu końcowym." : "ESI results after the final training.",
          body: isPl
            ? "Quadratic Weighted Kappa — metryka dla klasyfikacji porządkowej. Bezpieczeństwo: undertriage (ESI 1–2 jako ESI 3–5) i critical miss (ESI 1 jako ESI 4–5). Wagi ensembla dobierane na walidacji, raport na teście."
            : "Quadratic Weighted Kappa — the metric for ordinal classification. Safety: undertriage (ESI 1–2 as ESI 3–5) and critical miss (ESI 1 as ESI 4–5). Ensemble weights chosen on validation, reported on test.",
          side: "left",
          shape: "waveform",
          stat: { value: formatMetric(RESULTS?.qwk, 4), label: "QWK" },
        },
        {
          shape: "columns",
          title: "",
          heightFactor: 1.2,
          content: <JourneyResults />,
        },
      ],
    },
    {
      id: "demo",
      number: "07",
      name: isPl ? "Demo" : "Demo",
      stages: [
        {
          shape: "ring",
          title: "",
          heightFactor: 2.5,
          content: <JourneyDemo />,
        },
      ],
    },
    {
      id: "stack",
      number: "08",
      name: isPl ? "Stos technologiczny" : "Technology stack",
      stages: [
        {
          shape: "flower",
          title: "",
          heightFactor: 1.2,
          content: <JourneyTechStack />,
        },
      ],
    },
    {
      id: "timeline",
      number: "09",
      name: isPl ? "Historia" : "Timeline",
      stages: [
        {
          shape: "spiral",
          title: "",
          heightFactor: 1.4,
          content: <JourneyTimeline />,
        },
      ],
    },
    {
      id: "about",
      number: "10",
      name: isPl ? "Autor" : "Author",
      stages: [
        {
          shape: "heart",
          title: "",
          heightFactor: 1.3,
          content: <JourneyAbout />,
        },
      ],
    },
  ];
}

export function Home() {
  const { i18n } = useTranslation();
  const lang = ((i18n.resolvedLanguage ?? "pl").slice(0, 2) === "en" ? "en" : "pl") as "pl" | "en";
  const chapters = useMemo(() => buildChapters(lang), [lang]);

  return <CinematicJourney chapters={chapters} />;
}
