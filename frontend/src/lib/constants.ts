/**
 * Global constants for SOR-AI frontend.
 * Synced with backend types (backend/app/models/schemas.py).
 *
 * Label: Emergency Severity Index (ESI 1–5). The API returns class indexes
 * 0–4 = ESI 1–5; TOPSOR colours (red … blue) and max. waits are display only.
 */

export const ESI_LEVELS = [
  { id: 0, level: 1, code: "red",    colorPl: "Czerwony",     colorEn: "Red",    maxWait: "0 min",   maxWaitMin: 0,   color: "#DC2626" },
  { id: 1, level: 2, code: "orange", colorPl: "Pomarańczowy", colorEn: "Orange", maxWait: "10 min",  maxWaitMin: 10,  color: "#EA580C" },
  { id: 2, level: 3, code: "yellow", colorPl: "Żółty",        colorEn: "Yellow", maxWait: "60 min",  maxWaitMin: 60,  color: "#EAB308" },
  { id: 3, level: 4, code: "green",  colorPl: "Zielony",      colorEn: "Green",  maxWait: "120 min", maxWaitMin: 120, color: "#16A34A" },
  { id: 4, level: 5, code: "blue",   colorPl: "Niebieski",    colorEn: "Blue",   maxWait: "240 min", maxWaitMin: 240, color: "#2563EB" },
] as const;

/** Class index returned by the API (0–4 = ESI 1–5). */
export type EsiLevelId = (typeof ESI_LEVELS)[number]["id"];

/** "ESI 2 · Pomarańczowy" */
export function esiLabel(id: number, lang: "pl" | "en" = "pl"): string {
  const lvl = ESI_LEVELS[id];
  if (!lvl) return "—";
  return `ESI ${lvl.level} · ${lang === "pl" ? lvl.colorPl : lvl.colorEn}`;
}

export type ModelStatus = "trained" | "training" | "planned";

export interface ModelClassRecall {
  label: string;
  value: number; // 0–100
  color: string;
}

/**
 * Model card. Metrics are TEST-split values in % (undertriage, criticalMiss, accuracy)
 * or 0–1 (qwk, f1Macro, aucMacro, cohenKappa); undefined = not trained yet.
 * Fill them from training/runs/<run>/metrics.json after the final training.
 */
export interface Model {
  id: string;
  name: string;
  type: string;
  qwk?: number;
  accuracy?: number;
  f1Macro?: number;
  aucMacro?: number;
  cohenKappa?: number;
  undertriage?: number;
  criticalMiss?: number;
  trainTimeMin?: number;
  device: string;
  color: string;
  status: ModelStatus;
  description: string;
  classRecalls?: ModelClassRecall[];
  params?: Record<string, string | number>;
  notes?: string;
}

/** The 6 models served by the backend ensemble (backend/app/models/ensemble.py). */
export const MODELS: Model[] = [
  {
    id: "xgboost",
    name: "XGBoost",
    type: "Gradient Boosting",
    device: "GPU (CUDA) / CPU",
    color: "#00BFFF",
    status: "planned",
    description: "Histogram GBDT, early stopping na fragmencie zbioru treningowego. Tuning Optuna (TPE, wznawialny). GPU, gdy działa — inaczej CPU z ostrzeżeniem.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
  {
    id: "lightgbm",
    name: "LightGBM",
    type: "Gradient Boosting",
    device: "CPU",
    color: "#7CFC00",
    status: "planned",
    description: "Leaf-wise GBDT, natywna obsługa braków danych. Tuning Optuna z early stoppingiem.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
  {
    id: "catboost",
    name: "CatBoost",
    type: "Gradient Boosting",
    device: "GPU (CUDA) / CPU",
    color: "#FFCC00",
    status: "planned",
    description: "Oblivious (symmetric) trees, natywne CUDA. Tuning Optuna z early stoppingiem.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
  {
    id: "random_forest",
    name: "Random Forest",
    type: "Bagging",
    device: "CPU",
    color: "#FF69B4",
    status: "planned",
    description: "scikit-learn RandomForest (natywne NaN). Inny inductive bias niż boosting — różnorodność ensembla.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
  {
    id: "ebm",
    name: "EBM",
    type: "Glassbox",
    device: "CPU",
    color: "#9370DB",
    status: "planned",
    description: "Explainable Boosting Machine (InterpretML) — w pełni interpretowalny model addytywny.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
  {
    id: "logreg",
    name: "LogReg",
    type: "Linear baseline",
    device: "CPU",
    color: "#94A3B8",
    status: "planned",
    description: "Regresja logistyczna (imputacja medianą + standaryzacja) — liniowy punkt odniesienia.",
    notes: "Wyniki ESI pojawią się po treningu końcowym (runs/leaderboard.csv).",
  },
];

/** Model with the highest test QWK; null while no model has results. */
export function bestModel(models: Model[] = MODELS): Model | null {
  const scored = models.filter((m) => m.qwk !== undefined);
  if (scored.length === 0) return null;
  return scored.reduce((a, b) => ((a.qwk ?? 0) >= (b.qwk ?? 0) ? a : b));
}

/**
 * Final ensemble results on the TEST split (ESI 1–5). null = not trained yet —
 * the UI shows "—". Fill from training/runs (weights chosen on val, reported on test).
 * Rates in %, recallPerLevel in % for ESI 1..5, confusionMatrix rows = true ESI 1..5.
 */
export interface EnsembleResults {
  qwk: number;
  undertriage: number;
  criticalMiss: number;
  overtriage: number;
  recallPerLevel?: [number, number, number, number, number];
  confusionMatrix?: number[][];
  featureImportance?: { feature: string; value: number }[];
  source: string;
}

export const RESULTS: EnsembleResults | null = null;

export const VITALS_RANGES = {
  age: { min: 0, max: 120, default: 45, unit: "lat" },
  temp: { min: 32, max: 42, default: 36.6, step: 0.1, unit: "°C" },
  hr: { min: 30, max: 220, default: 75, unit: "bpm" },
  sbp: { min: 50, max: 250, default: 120, unit: "mmHg" },
  dbp: { min: 30, max: 150, default: 80, unit: "mmHg" },
  rr: { min: 8, max: 60, default: 16, unit: "/min" },
  o2: { min: 60, max: 100, default: 98, unit: "%" },
} as const;

export const NAV_LINKS = [
  { href: "/#problem", labelKey: "nav.home", anchor: "problem" },
  { href: "/#demo", labelKey: "nav.demo", anchor: "demo" },
  { href: "/#models", labelKey: "nav.models", anchor: "models" },
  { href: "/#about", labelKey: "nav.about", anchor: "about" },
] as const;

export const SECTIONS = [
  { id: "hero", labelKey: "nav.sections.hero" },
  { id: "problem", labelKey: "nav.sections.problem" },
  { id: "data", labelKey: "nav.sections.data" },
  { id: "architecture", labelKey: "nav.sections.architecture" },
  { id: "models", labelKey: "nav.sections.models" },
  { id: "results", labelKey: "nav.sections.results" },
  { id: "how-it-works", labelKey: "nav.sections.howItWorks" },
  { id: "demo", labelKey: "nav.sections.demo" },
  { id: "tech-stack", labelKey: "nav.sections.techStack" },
  { id: "timeline", labelKey: "nav.sections.timeline" },
] as const;

export interface TechItem {
  name: string;
  color: string;
  version?: string;
  role: string;
  why: string;
  how: string;
  where: string;
}

export const TECH_STACK: Record<string, TechItem[]> = {
  frontend: [
    {
      name: "React 18", color: "#61DAFB", version: "18.3",
      role: "Główny framework UI",
      why: "Concurrent rendering, Suspense dla lazy-load three.js i Recharts. Największy ekosystem komponentów.",
      how: "SPA z React Router, lazy imports dla ciężkich chunków (three.js 820KB gzip). useMemo/useCallback stabilizują referencje dla Framer Motion.",
      where: "Cały frontend — journey, demo, modele, wyniki.",
    },
    {
      name: "TypeScript", color: "#3178C6", version: "5.5",
      role: "Typowanie statyczne",
      why: "Strict mode eliminuje całą klasę błędów runtime. Autocomplete dla interfejsów API i stanu.",
      how: "strict: true, path aliases @/*. Interfejsy dla PredictResponse, Model, JourneyStage synchronizowane z backendem.",
      where: "Cały projekt — 0 błędów tsc przed każdym deployem.",
    },
    {
      name: "Vite", color: "#646CFF", version: "5.4",
      role: "Build tool i dev server",
      why: "ESM-native HMR — zmiana pliku widoczna w <100ms. Build produkcyjny ~9s vs ~60s webpack.",
      how: "Rollup pod spodem, manualne chunki dla three.js i Recharts. Aliasy ścieżek, env variables.",
      where: "Dev server i build pipeline. Wrangler Pages deploy bezpośrednio z /dist.",
    },
    {
      name: "Tailwind CSS", color: "#06B6D4", version: "3.4",
      role: "Utility-first CSS",
      why: "Zero runtime CSS-in-JS. JIT kompilacja — tylko użyte klasy w bundlu. Design tokens w jednym miejscu.",
      how: "Custom konfiguracja: gradient-text, grid-bg, glow animacje. Dark mode przez class strategy.",
      where: "Cały styling — komponenty UI, animacje, responsywność.",
    },
    {
      name: "Framer Motion", color: "#FF4154", version: "11.3",
      role: "Biblioteka animacji",
      why: "Spring physics zamiast CSS transitions — naturalny ruch. AnimatePresence dla mount/unmount. useScroll dla scroll-driven journey.",
      how: "useMotionValueEvent śledzi scrollYProgress. motion.div z opacity/x transforms per stage. AnimatePresence w modal i chapter indicator.",
      where: "CinematicJourney (scroll transforms), modals (spring scale), SHAP bars (width animation).",
    },
    {
      name: "three.js", color: "#FFFFFF", version: "0.168",
      role: "3D particle system",
      why: "WebGL dla 2400 cząsteczek w 60fps nie da się zrobić w CSS/Canvas wydajnie.",
      how: "React Three Fiber (@react-three/fiber). MorphingParticles.tsx: 17 shape generators, LERP morphing, per-particle brightness jitter. Lazy import — ładuje się tylko gdy sekcja wejdzie w viewport.",
      where: "CinematicJourney — sticky particle canvas przez całą podróż scrolltelling.",
    },
  ],
  backend: [
    {
      name: "Java · Spring Boot", color: "#6DB33F", version: "3.3",
      role: "Główny API server",
      why: "Enterprise patterns, connection pooling, Spring Security dla endpoints. Sprawdzony w środowiskach szpitalnych.",
      how: "REST controllers dla /api/predict, /api/health. Async processing z CompletableFuture. Kafka producer dla event logging.",
      where: "Bramka wejściowa — przyjmuje żądania frontend, orkiestruje ML i LLM.",
    },
    {
      name: "FastAPI (Python)", color: "#009688", version: "0.115",
      role: "ML inference microservice",
      why: "Python-native dla joblib modeli. Async endpoints, automatyczna dokumentacja OpenAPI. Pydantic walidacja danych wejściowych.",
      how: "Ładuje przy starcie 6 modeli (*.joblib) i opcjonalne wagi ensemble_weights.json. /predict: feature engineering → ensemble ESI 1–5 → reguły bezpieczeństwa ESI → SHAP → response.",
      where: "Serwis ML — odizolowany od głównego backendu, komunikacja przez HTTP.",
    },
    {
      name: "Apache Kafka", color: "#231F20", version: "3.7",
      role: "Event streaming",
      why: "Asynchroniczny audit log każdej predykcji. Odporność na awarie — zdarzenia nie giną. Podstawa pod przyszłą analizę dryfu modelu.",
      how: "Topic: triage-predictions. Producent: Spring Boot. Konsument: serwis analityczny (planowany). Retention 30 dni.",
      where: "Między Spring Boot a serwisem logowania — każda predykcja zapisana jako event.",
    },
    {
      name: "Nginx · load balancer", color: "#009639", version: "1.27",
      role: "Reverse proxy i SSL",
      why: "Terminacja SSL, routing /api → Spring Boot, /ml → FastAPI. Statyczne pliki bezpośrednio z dysku.",
      how: "upstream blocks dla obu serwisów. Rate limiting: 10 req/s per IP. Gzip kompresja odpowiedzi.",
      where: "Punkt wejścia do całego systemu. Port 443, redirect z 80.",
    },
  ],
  ml: [
    {
      name: "XGBoost", color: "#1F77B4", version: "2.1",
      role: "Model gradientowy #1",
      why: "tree_method=hist + device=cuda — szybki trening na GPU, automatyczny powrót na CPU.",
      how: "Early stopping na fragmencie zbioru treningowego, tuning Optuna (k-fold CV).",
      where: "Warstwa 1A ensembla (wagi dobierane na zbiorze walidacyjnym).",
    },
    {
      name: "CatBoost", color: "#FFCC00", version: "1.2",
      role: "Model gradientowy #2",
      why: "Natywna obsługa CUDA. Oblivious symmetric trees — szybki inference.",
      how: "Trening na GPU (RTX 4080 Super) z early stoppingiem; tuning Optuna.",
      where: "Warstwa 1A ensembla.",
    },
    {
      name: "LightGBM", color: "#7CB342", version: "4.5",
      role: "Model gradientowy #3",
      why: "Leaf-wise splitting zamiast level-wise — szybki i dokładny na danych tabelarycznych.",
      how: "CPU (wszystkie rdzenie fizyczne), early stopping, tuning Optuna.",
      where: "Warstwa 1A ensembla.",
    },
    {
      name: "scikit-learn", color: "#F7931E", version: "1.5",
      role: "Random Forest, LogReg, walidacja",
      why: "RandomForest z natywną obsługą NaN, pipeline LogReg (imputacja + skalowanie), StratifiedKFold. Spójne API.",
      how: "StratifiedKFold dla CV w tuningu. cohen_kappa_score (QWK) jako główna metryka ordinalna.",
      where: "Modele random_forest i logreg oraz metryki ewaluacji.",
    },
    {
      name: "InterpretML · EBM", color: "#76B900", version: "0.7",
      role: "Model glassbox",
      why: "Explainable Boosting Machine — dokładność bliska boostingowi przy pełnej interpretowalności (wymóg MDR).",
      how: "ExplainableBoostingClassifier (interpret-core), wiele rdzeni CPU, tuning Optuna na próbce.",
      where: "Model ebm w ensemblu.",
    },
    {
      name: "Optuna", color: "#1F4E79", version: "4.1",
      role: "Hyperparameter tuning",
      why: "MultivariateTPE modeluje korelacje między hiperparametrami. SQLite storage — wznowienie po awarii lub restarcie.",
      how: "Budżety prób i czasu wg profilu (smoke / quick / standard / thorough), QWK z k-fold CV jako objective.",
      where: "Tuning każdego z 6 modeli.",
    },
  ],
  llm: [
    {
      name: "Ollama", color: "#FFFFFF", version: "0.5",
      role: "Lokalny serwer LLM",
      why: "Dane pacjentów NIE opuszczają sieci szpitalnej. Brak API calls do chmury. RODO i polskie prawo medyczne.",
      how: "REST API: POST /api/generate, /api/chat. Ładuje modele do VRAM RTX 5090. Obsługa wielu modeli równolegle.",
      where: "Warstwa 1B systemu — serwer modeli językowych działający lokalnie.",
    },
    {
      name: "Llama 3.2 3B · parser", color: "#1877F2", version: "3.2",
      role: "Parser notatki klinicznej",
      why: "3B parametrów — inference <1s. temperature=0 + structured output = deterministyczny JSON. Nie wymaga GPU.",
      how: "Wejście: surowa notatka pielęgniarska. Wyjście: JSON {symptoms, vitals_mentions, urgency_keywords}. Używa Ollama structured output mode.",
      where: "Warstwa 0 — pierwsza transformacja przed modelem ML.",
    },
    {
      name: "MedGemma 27B · NLP", color: "#4285F4", version: "27B",
      role: "Kliniczne uzasadnienie",
      why: "Google MedGemma trenowany na danych medycznych. Rozumie terminologię kliniczną, ICD-10 i skalę ESI.",
      how: "Wejście: predykcja ML + parametry życiowe + notatka. Wyjście: poziom ESI 1–5, reasoning, risk_flags, key_findings.",
      where: "Warstwa 1B — generuje słowne uzasadnienie pokazywane w demo.",
    },
    {
      name: "Qwen3 32B · synteza", color: "#A855F7", version: "32B",
      role: "Synteza decyzji końcowej",
      why: "32B parametrów dla złożonego rozumowania wielokrokowego. Qwen3 ma silne reasoning capabilities (Chain-of-Thought).",
      how: "Wejście: predykcje ML + ocena MedGemma + flagi konfliktu. Wyjście: final_decision report z uzasadnieniem. Uruchamiany tylko gdy conflict.detected=true.",
      where: "Warstwa 2 — finalny raport przy wykrytej rozbieżności między modelami.",
    },
  ],
};

export const TIMELINE_EVENTS = [
  { date: "2025-05", titleKey: "timeline.events.idea" },
  { date: "2025-07", titleKey: "timeline.events.dataAnalysis" },
  { date: "2025-09", titleKey: "timeline.events.preprocessing" },
  { date: "2025-10", titleKey: "timeline.events.firstXGBoost" },
  { date: "2025-11", titleKey: "timeline.events.optunaTuning" },
  { date: "2026-01", titleKey: "timeline.events.multiGpu" },
  { date: "2026-03", titleKey: "timeline.events.quantileDMatrix" },
  { date: "2026-05", titleKey: "timeline.events.catboostFinal" },
  { date: "2026-05", titleKey: "timeline.events.xgboostA5000" },
  { date: "2026-06", titleKey: "timeline.events.frontend" },
  { date: "2026-10", titleKey: "timeline.events.esiNative" },
  { date: "2027-07", titleKey: "timeline.events.defense" },
] as const;

/** ESI distribution in Yale EMMLC (558 027 visits; training/DATA_AUDIT.md). */
export const CLASS_DISTRIBUTION = [
  { name: "ESI 1", value: 0.9, color: "#DC2626" },
  { name: "ESI 2", value: 29.3, color: "#EA580C" },
  { name: "ESI 3", value: 42.3, color: "#EAB308" },
  { name: "ESI 4", value: 22.4, color: "#16A34A" },
  { name: "ESI 5", value: 5.0, color: "#2563EB" },
] as const;
