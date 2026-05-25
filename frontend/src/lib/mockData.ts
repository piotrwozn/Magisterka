import type { PredictRequest, PredictResponse } from "./types";

/**
 * Realistic mock prediction for demo when backend unavailable.
 * Heuristic: derives category from vitals (SBP, HR, temp, age, O2, RR).
 */
export function mockPredict(req: PredictRequest): PredictResponse {
  const v = req.vitals;
  const note = req.clinicalNote.toLowerCase();

  let score = 0;

  if (v.sbp < 90) score += 4;
  else if (v.sbp < 100) score += 2;

  if (v.hr > 130) score += 4;
  else if (v.hr > 110) score += 2;
  else if (v.hr < 50) score += 3;

  if (v.temp >= 39) score += 3;
  else if (v.temp >= 38) score += 1;
  else if (v.temp < 35) score += 4;

  if (v.o2 < 88) score += 5;
  else if (v.o2 < 92) score += 3;
  else if (v.o2 < 95) score += 1;

  if (v.rr > 28) score += 3;
  else if (v.rr > 22) score += 1;

  if (v.age >= 75) score += 1;

  const noteSignals: Record<string, number> = {
    "ból w klatce": 4,
    "chest pain": 4,
    duszność: 3,
    dyspnea: 3,
    "utrata przytomności": 5,
    unconscious: 5,
    krwawienie: 3,
    bleeding: 3,
    blady: 2,
    pale: 2,
    spocony: 2,
    sweating: 2,
    drgawki: 4,
    seizure: 4,
    udar: 5,
    stroke: 5,
    zawał: 5,
    "heart attack": 5,
    drobiazg: -2,
    "minor injury": -2,
    "wizyta kontrolna": -3,
    "follow-up": -3,
  };

  for (const [k, v] of Object.entries(noteSignals)) {
    if (note.includes(k)) score += v;
  }

  let category: number;
  if (score >= 8) category = 0;
  else if (score >= 5) category = 1;
  else if (score >= 2) category = 2;
  else if (score >= 0) category = 3;
  else category = 4;

  const probs = [0, 0, 0, 0, 0];
  const noise = () => 0.02 + Math.random() * 0.05;
  for (let i = 0; i < 5; i++) {
    const dist = Math.abs(i - category);
    probs[i] = Math.max(0.01, 0.85 / (1 + dist * 2) + noise() - 0.03);
  }
  const sum = probs.reduce((a, b) => a + b, 0);
  const normalized = probs.map((p) => p / sum);

  return {
    finalCategory: category,
    confidence: normalized[category]!,
    modelPredictions: [
      { modelName: "catboost", category, probabilities: normalized, confidence: normalized[category]! },
      {
        modelName: "xgboost",
        category: Math.max(0, category + (Math.random() > 0.85 ? 1 : 0)),
        probabilities: normalized,
        confidence: normalized[category]! * 0.97,
      },
      {
        modelName: "lightgbm",
        category,
        probabilities: normalized,
        confidence: normalized[category]! * 0.93,
      },
      {
        modelName: "random_forest",
        category,
        probabilities: normalized,
        confidence: normalized[category]! * 0.88,
      },
      {
        modelName: "ebm",
        category,
        probabilities: normalized,
        confidence: normalized[category]! * 0.84,
      },
    ],
    medgemma: {
      category,
      confidence: 0.82 + Math.random() * 0.13,
      reasoning:
        category <= 1
          ? "Parametry życiowe wskazują na stan zagrożenia. Wymagana natychmiastowa interwencja."
          : category === 2
            ? "Stabilne parametry z odchyleniami. Wymagana ocena lekarska w ciągu 60 minut."
            : "Stan stabilny, niski priorytet medyczny.",
      riskFlags:
        category === 0
          ? ["hipotensja", "tachykardia", "hipoksja"]
          : category === 1
            ? ["odchylenia parametrów", "wymaga obserwacji"]
            : [],
      keyFindings:
        category <= 1
          ? [`SBP=${v.sbp}`, `HR=${v.hr}`, `SpO2=${v.o2}%`]
          : [`temp=${v.temp}°C`, `wiek=${v.age}`],
    },
    shapTop5: [
      { feature: "triage_vital_sbp", value: v.sbp < 100 ? 0.342 : -0.082, direction: v.sbp < 100 ? "positive" : "negative" },
      { feature: "triage_vital_hr", value: v.hr > 110 ? 0.287 : -0.041, direction: v.hr > 110 ? "positive" : "negative" },
      { feature: "triage_vital_o2", value: v.o2 < 95 ? 0.231 : -0.038, direction: v.o2 < 95 ? "positive" : "negative" },
      { feature: "triage_vital_temp", value: v.temp >= 38 ? 0.184 : -0.022, direction: v.temp >= 38 ? "positive" : "negative" },
      { feature: "age_group", value: v.age >= 70 ? 0.118 : -0.015, direction: v.age >= 70 ? "positive" : "negative" },
    ],
    conflict: (() => {
      const detected = Math.random() > 0.65;
      const severity = detected && Math.random() > 0.6 ? "high" : "low";
      const alertDoctor = severity === "high";
      return {
        detected,
        severity,
        alertDoctor,
        message: alertDoctor
          ? "Znaczna rozbieżność między modelami — wymagana konsultacja lekarska"
          : detected
            ? "Niewielka rozbieżność między modelami — wymaga obserwacji"
            : "Wszystkie modele zgodne",
      };
    })(),
    processingTimeMs: 350 + Math.floor(Math.random() * 600),
  };
}

export const EXAMPLE_PATIENTS = [
  {
    nameKey: "demo.examples.chestPain",
    vitals: { age: 67, temp: 38.2, hr: 118, sbp: 95, dbp: 62, rr: 22, o2: 94 },
    clinicalNote:
      "Pacjent blady, spocony, ból w klatce promieniujący do żuchwy, duszność, niespokojny",
  },
  {
    nameKey: "demo.examples.fracture",
    vitals: { age: 32, temp: 36.8, hr: 88, sbp: 132, dbp: 84, rr: 16, o2: 99 },
    clinicalNote: "Złamanie ręki po upadku z roweru, obrzęk, ból przy ruchu, świadomy, stabilny",
  },
  {
    nameKey: "demo.examples.cold",
    vitals: { age: 28, temp: 37.4, hr: 82, sbp: 124, dbp: 78, rr: 14, o2: 98 },
    clinicalNote: "Przeziębienie, lekki katar, kaszel, ogólne złe samopoczucie",
  },
  {
    nameKey: "demo.examples.followUp",
    vitals: { age: 45, temp: 36.6, hr: 72, sbp: 118, dbp: 76, rr: 14, o2: 99 },
    clinicalNote: "Wizyta kontrolna po wcześniejszym leczeniu, brak nowych dolegliwości",
  },
  {
    nameKey: "demo.examples.intoxication",
    vitals: { age: 41, temp: 36.2, hr: 105, sbp: 110, dbp: 70, rr: 18, o2: 96 },
    clinicalNote: "Pacjent pod wpływem alkoholu, agresywny, splątany, niestabilny w pozycji stojącej",
  },
] as const;
