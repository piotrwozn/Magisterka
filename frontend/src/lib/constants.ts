/**
 * Global constants for SOR-AI frontend.
 * Synced with backend types.
 */

export const MTS_CATEGORIES = [
  { id: 0, code: "red", labelPl: "Czerwony", labelEn: "Red", maxWait: "0 min", color: "#DC2626" },
  { id: 1, code: "orange", labelPl: "Pomarańczowy", labelEn: "Orange", maxWait: "10 min", color: "#EA580C" },
  { id: 2, code: "yellow", labelPl: "Żółty", labelEn: "Yellow", maxWait: "60 min", color: "#EAB308" },
  { id: 3, code: "green", labelPl: "Zielony", labelEn: "Green", maxWait: "120 min", color: "#16A34A" },
  { id: 4, code: "blue", labelPl: "Niebieski", labelEn: "Blue", maxWait: "240 min", color: "#2563EB" },
] as const;

export type MTSCategoryId = (typeof MTS_CATEGORIES)[number]["id"];

export type ModelStatus = "trained" | "training" | "planned";

export interface ModelClassRecall {
  label: string;
  value: number; // 0–100
  color: string;
}

export interface Model {
  id: string;
  name: string;
  type: string;
  qwk: number;
  accuracy?: number;
  f1Macro?: number;
  aucMacro?: number;
  cohenKappa?: number;
  undertriage: number;
  criticalMiss: number;
  trainTimeMin: number;
  device: string;
  color: string;
  status: ModelStatus;
  description: string;
  classRecalls?: ModelClassRecall[];
  params?: Record<string, string | number>;
  notes?: string;
}

export const MODELS: Model[] = [
  {
    id: "catboost",
    name: "CatBoost",
    type: "Gradient Boosting",
    qwk: 0.8729,
    accuracy: 87.20,
    f1Macro: 0.6869,
    aucMacro: 0.9711,
    cohenKappa: 0.7321,
    undertriage: 3.22,
    criticalMiss: 0.055,
    trainTimeMin: 240,
    device: "GPU (CUDA)",
    color: "#FFCC00",
    status: "trained",
    description: "Oblivious symmetric trees z natywnym CUDA. Wytrenowany na 8× RTX 5090, 3800 iteracji, Optuna 154 triiali.",
    classRecalls: [
      { label: "Red",    value: 99.1, color: "#DC2626" },
      { label: "Orange", value: 68.1, color: "#EA580C" },
      { label: "Yellow", value: 53.4, color: "#EAB308" },
      { label: "Green",  value: 52.1, color: "#16A34A" },
      { label: "Blue",   value: 76.0, color: "#2563EB" },
    ],
    params: {
      "iterations": 3800,
      "depth": 9,
      "learning_rate": 0.130,
      "l2_leaf_reg": 0.127,
      "subsample": 0.639,
      "grow_policy": "SymmetricTree",
      "bootstrap_type": "Bernoulli",
    },
    notes: "Najniższy critical miss (0.055%). Red recall 99.1% — minimalne przeoczenie stanów krytycznych. Optuna 154 triiale, MultivariateTPE, 5-fold CV.",
  },
  {
    id: "xgboost",
    name: "XGBoost",
    type: "Gradient Boosting",
    qwk: 0.8691,
    accuracy: 86.74,
    f1Macro: 0.6799,
    aucMacro: 0.9706,
    cohenKappa: 0.7236,
    undertriage: 3.83,
    criticalMiss: 0.07,
    trainTimeMin: 180,
    device: "GPU (CUDA)",
    color: "#00BFFF",
    status: "trained",
    description: "QuantileDMatrix + tree_method=hist, 5 foldów równolegle na osobnych RTX 5090. Optuna 150 triali, lossguide grow policy.",
    classRecalls: [
      { label: "Red",    value: 99.1, color: "#DC2626" },
      { label: "Orange", value: 64.7, color: "#EA580C" },
      { label: "Yellow", value: 50.8, color: "#EAB308" },
      { label: "Green",  value: 60.8, color: "#16A34A" },
      { label: "Blue",   value: 76.0, color: "#2563EB" },
    ],
    params: {
      "n_estimators": 3500,
      "max_depth": 14,
      "learning_rate": 0.0113,
      "subsample": 0.416,
      "colsample_bytree": 0.830,
      "colsample_bylevel": 0.569,
      "colsample_bynode": 0.899,
      "min_child_weight": 4.86,
      "reg_alpha": 0.702,
      "reg_lambda": 0.620,
      "max_bin": 1280,
      "grow_policy": "lossguide",
      "tree_method": "hist",
    },
    notes: "Best trial: #83 z 150 (QWK CV=0.8786). Lossguide grow policy + 8× RTX 5090 multi-GPU. Critical miss 0.07% — drugi najlepszy po CatBoost.",
  },
{
    id: "lightgbm",
    name: "LightGBM",
    type: "Gradient Boosting",
    qwk: 0.8719,
    accuracy: 87.25,
    f1Macro: 0.6899,
    aucMacro: 0.9724,
    cohenKappa: 0.7328,
    undertriage: 2.38,
    criticalMiss: 0.04,
    trainTimeMin: 25,
    device: "CPU (Lokalnie)",
    color: "#7CFC00",
    status: "trained",
    description: "Leaf-wise splitting GBDT z custom class weights (Optuna, retrained 2026-05-24). 2100 drzew, najniższy critical miss i undertriage ze wszystkich modeli.",
    classRecalls: [
      { label: "Red",    value: 99.2, color: "#DC2626" },
      { label: "Orange", value: 72.9, color: "#EA580C" },
      { label: "Yellow", value: 48.5, color: "#EAB308" },
      { label: "Green",  value: 56.5, color: "#16A34A" },
      { label: "Blue",   value: 65.8, color: "#2563EB" },
    ],
    params: {
      "n_estimators": 2100,
      "max_depth": 12,
      "num_leaves": 33,
      "learning_rate": 0.0559,
      "subsample": 0.523,
      "colsample_bytree": 0.551,
      "max_bin": 960,
    },
    notes: "Retrenowany z custom weights — critical miss spadł z 0.41% do 0.04% (10×). Najniższy undertriage 2.38%. Red recall 99.2%.",
  },
    notes: "Najlepszy recall Orange (73.7%) i Green (60.6%) spośród wytrenowanych modeli. Optuna 152 triiale, MultivariateTPE, 5-fold CV.",
  },
  {
    id: "random_forest",
    name: "Random Forest",
    type: "Bagging",
    qwk: 0.8400,
    accuracy: 84.73,
    f1Macro: 0.6084,
    aucMacro: 0.97,
    cohenKappa: 0.6783,
    undertriage: 4.98,
    criticalMiss: 0.04,
    trainTimeMin: 75,
    device: "GPU (cuML)",
    color: "#FF69B4",
    status: "trained",
    description: "cuML RandomForest, 900 drzew, 150 triali Optuny z custom class weights. 5 foldów równolegle na RTX 5090.",
    classRecalls: [
      { label: "Red",    value: 99.3, color: "#DC2626" },
      { label: "Orange", value: 54.7, color: "#EA580C" },
      { label: "Yellow", value: 54.3, color: "#EAB308" },
      { label: "Green",  value: 32.1, color: "#16A34A" },
      { label: "Blue",   value: 68.2, color: "#2563EB" },
    ],
    params: {
      "n_estimators": 900,
      "max_depth": 27,
      "min_samples_split": 39,
      "max_features": "0.4",
      "bootstrap": "True",
      "max_leaf_nodes": 5000,
    },
    notes: "Custom class weights (Optuna). QWK 0.84 — drugi najgorszy ale kluczowy diversity booster do stackingu. Red recall 99.3% — drugi najlepszy po CatBoost.",
  },
  {
    id: "extra_trees",
    name: "ExtraTrees",
    type: "Bagging",
    qwk: 0.83,
    undertriage: 5.1,
    criticalMiss: 0.18,
    trainTimeMin: 90,
    device: "CPU",
    color: "#DA70D6",
    status: "training",
    description: "Maksymalna randomizacja splits, komplementarny do RF w ensemblu. Tuning w toku.",
  },
  {
    id: "hist_gbt",
    name: "HistGBT",
    type: "Gradient Boosting",
    qwk: 0.84,
    undertriage: 4.1,
    criticalMiss: 0.14,
    trainTimeMin: 150,
    device: "CPU",
    color: "#FF7F50",
    status: "planned",
    description: "sklearn HistGradientBoosting, inny inductive bias. Wartości szacowane.",
  },
  {
    id: "ebm",
    name: "EBM",
    type: "Glassbox",
    qwk: 0.81,
    undertriage: 5.8,
    criticalMiss: 0.21,
    trainTimeMin: 480,
    device: "CPU",
    color: "#9370DB",
    status: "planned",
    description: "Explainable Boosting Machine (InterpretML). Pełna interpretowalność wymagana przez MDR. Wartości szacowane.",
  },
];

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
      how: "Ładuje modele przy starcie: CatBoost, XGBoost, LightGBM z /models/*.joblib. /predict endpoint: feature engineering → ensemble → SHAP → response.",
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
      why: "QuantileDMatrix 2–3× szybszy niż DMatrix na GPU. tree_method=hist + device=cuda. Najlepszy QWK w dotychczasowych testach.",
      how: "5-fold CV równolegle na 5 GPU. 3500 estimatorów, max_depth=14. Optuna 150 triali (w toku). Wagi klas cost-sensitive.",
      where: "Warstwa 1A ensembla. Wejście do meta-learnera stacking LogReg.",
    },
    {
      name: "CatBoost", color: "#FFCC00", version: "1.2",
      role: "Model gradientowy #2",
      why: "Natywna obsługa CUDA. Oblivious symmetric trees — szybszy inference. Najniższy critical miss (0.055%).",
      how: "GPU: 8× RTX 5090. 3800 iteracji, depth=9. Optuna 154 triiale, MultivariateTPE. Bernoulli bootstrap.",
      where: "Warstwa 1A ensembla. QWK 0.8729 na test set.",
    },
    {
      name: "LightGBM", color: "#7CB342", version: "4.5",
      role: "Model gradientowy #3",
      why: "Leaf-wise splitting zamiast level-wise — głębsze drzewa dla 336 features. Najlepszy recall Orange/Green.",
      how: "CPU, 2100 drzew, num_leaves=33, max_depth=12. Optuna 152 triiale. 5-fold CV z OMP parallelism per fold.",
      where: "Warstwa 1A ensembla. QWK 0.8705 na test set.",
    },
    {
      name: "scikit-learn", color: "#F7931E", version: "1.5",
      role: "Preprocessing i stacking",
      why: "StandardScaler, StratifiedKFold, LogisticRegression meta-learner. Spójne API z resztą ML pipeline.",
      how: "StratifiedKFold(5) dla CV. LogisticRegression jako meta-learner stacking (wejście: predykcje 7 modeli). cohen_kappa_score jako główna metryka.",
      where: "Preprocessing pipeline i warstwa meta-learnera stacking.",
    },
    {
      name: "cuML · RAPIDS", color: "#76B900", version: "24.10",
      role: "GPU Random Forest",
      why: "cuML RF 10–50× szybszy niż sklearn na GPU dla dużych danych. Ta sama API co sklearn.",
      how: "from cuml.ensemble import RandomForestClassifier. Auto-detect GPU przy starcie tuningu. Fallback do sklearn CPU jeśli brak GPU.",
      where: "Model random_forest w ensemblu. 5-fold równolegle na 5 GPU.",
    },
    {
      name: "Optuna", color: "#1F4E79", version: "4.1",
      role: "Hyperparameter tuning",
      why: "MultivariateTPE modeluje korelacje między hiperparametrami. SQLite storage — wznowienie po crashu/wyłączeniu serwera.",
      how: "NopPruner (żaden trial nie ubijany). n_startup_trials=50 przed TPE. Storage: /tmp/opencode/optuna_studies_{model}.db. 5-fold QWK jako objective.",
      where: "Tuning każdego z 7 modeli. ~150 triali per model.",
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
      why: "Google MedGemma trenowany na danych medycznych. Rozumie terminologię kliniczną, ICD-10, protokoły MTS.",
      how: "Wejście: predykcja ML + parametry życiowe + notatka. Wyjście: reasoning, risk_flags, key_findings. Inference ~3–5s na RTX 5090.",
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
  { date: "2027-07", titleKey: "timeline.events.defense" },
] as const;

export const CLASS_DISTRIBUTION = [
  { name: "Red", value: 68.6, color: "#DC2626" },
  { name: "Orange", value: 13.2, color: "#EA580C" },
  { name: "Yellow", value: 12.2, color: "#EAB308" },
  { name: "Green", value: 4.1, color: "#16A34A" },
  { name: "Blue", value: 1.9, color: "#2563EB" },
] as const;
