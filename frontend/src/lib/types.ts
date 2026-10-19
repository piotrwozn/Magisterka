/**
 * Type definitions shared across frontend.
 * Matches backend API schema (backend/app/models/schemas.py, camelCase).
 *
 * Every `category` is an ESI class index: 0–4 = ESI 1–5
 * (TOPSOR colours red, orange, yellow, green, blue — display only).
 */

export interface Vitals {
  age: number;
  temp: number;
  hr: number;
  sbp: number;
  dbp: number;
  rr: number;
  o2: number;
}

export type Sex = "M" | "F";
export type ArrivalMode = "ambulance" | "self" | "other";

/** Optional patient context (features of the models; omitted = unknown). */
export interface PatientContext {
  sex?: Sex;
  arrivalMode?: ArrivalMode;
  o2Device?: boolean;
}

export interface PredictRequest extends PatientContext {
  vitals: Vitals;
  clinicalNote: string;
}

export interface ModelPrediction {
  modelName: string;
  /** ESI class index 0–4 (= ESI 1–5) */
  category: number;
  probabilities: number[];
  confidence: number;
}

export interface ShapValue {
  feature: string;
  value: number;
  direction: "positive" | "negative";
}

export interface MedGemmaAssessment {
  category: number;
  confidence: number;
  reasoning: string;
  riskFlags: string[];
  keyFindings: string[];
}

export interface ConflictInfo {
  detected: boolean;
  severity: "low" | "high";
  alertDoctor: boolean;
  message: string;
}

export interface PredictResponse {
  /** ESI class index 0–4 (= ESI 1–5) after the safety rules */
  finalCategory: number;
  confidence: number;
  modelPredictions: ModelPrediction[];
  medgemma: MedGemmaAssessment;
  shapTop5: ShapValue[];
  conflict: ConflictInfo;
  processingTimeMs: number;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "down";
  modelsLoaded: string[];
  ollamaReady: boolean;
  uptimeSeconds: number;
}

export type Language = "pl" | "en";
export type Theme = "light" | "dark" | "system";
