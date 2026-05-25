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
          eyebrow: isPl ? "Manchester Triage System" : "Manchester Triage System",
          title: isPl ? "Pięć klas pilności. Jedna decyzja." : "Five urgency classes. One verdict.",
          body: isPl
            ? "Czerwony, Pomarańczowy, Żółty, Zielony, Niebieski. Rozkład klas 36× niesymetryczny (Red 68,6% vs Blue 1,9%). Zaniżenie pilności kosztuje 3× więcej niż jej zawyżenie."
            : "Red, Orange, Yellow, Green, Blue. 36× class imbalance (Red 68.6% vs Blue 1.9%). Undertriage carries 3× the clinical cost of overtriage.",
          side: "right",
          shape: "starburst",
          stat: { value: "36×", label: "imbalance" },
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
            ? "558 029 wizyt SOR. 1090 cech klinicznych."
            : "558,029 ED visits. 1090 clinical features.",
          body: isPl
            ? "Anonimizowany dataset z Yale-New Haven Hospital. Średni wiek 50 lat, 55% kobiet. Vitalne parametry, 200 typów chief complaint, choroby współistniejące, leki, notatki triażowe."
            : "Anonymized dataset from Yale-New Haven Hospital. Mean age 50, 55% female. Vitals, 200 chief-complaint categories, comorbidities, medications, triage notes.",
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
          title: isPl ? "Ensemble siedmiu modeli" : "Ensemble of seven models",
          body: isPl
            ? "CatBoost, XGBoost, LightGBM, RF, ExtraTrees, HistGBT, EBM. Stacking z meta-learnerem LogReg. SHAP top-5 cech dla każdej predykcji."
            : "CatBoost, XGBoost, LightGBM, RF, ExtraTrees, HistGBT, EBM. LogReg stacking. SHAP top-5 features per prediction.",
          side: "left",
          shape: "constellation",
          stat: { value: "7", label: isPl ? "modeli" : "models" },
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
            ? "Łączy wynik tabelaryczny z oceną kliniczną. Zasada nadrzędna: eskaluj, nigdy nie obniżaj. Różnica ≥ 2 stopnie → alert lekarski."
            : "Combines tabular and clinical scores. Master rule: escalate, never downgrade. Gap ≥ 2 grades → doctor alert.",
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
            ? "MultivariateTPE · 5-fold CV · 150 trials"
            : "MultivariateTPE · 5-fold CV · 150 trials",
          body: isPl
            ? "Stały seed dla fair comparison. Wagi klas jako hiperparametry. n_startup_trials=50 eksploracji. SQLite storage do wznowienia."
            : "Fixed seed for fair comparison. Class weights as hyperparameters. 50 startup trials. SQLite for resume.",
          side: "right",
          shape: "torus",
          stat: { value: "150", label: isPl ? "trials/model" : "trials/model" },
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
          title: isPl ? "QWK 0,8718. Critical miss 0,06%." : "QWK 0.8718. Critical miss 0.06%.",
          body: isPl
            ? "Quadratic Weighted Kappa — najlepsza metryka dla klasyfikacji ordinalnej. Undertriage 2,7% przy targecie < 2%. Stabilne przez foldy CV."
            : "Quadratic Weighted Kappa — the right metric for ordinal classification. Undertriage 2.7% against < 2% target. Stable across CV folds.",
          side: "left",
          shape: "waveform",
          stat: { value: "0.8718", label: "QWK" },
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
