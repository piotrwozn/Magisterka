/**
 * Type definitions shared across frontend.
 * Matches backend API schema.
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

export interface PredictRequest {
  vitals: Vitals;
  clinicalNote: string;
}

export interface ModelPrediction {
  modelName: string;
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
