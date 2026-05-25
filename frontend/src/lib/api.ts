import axios, { type AxiosInstance } from "axios";

import type { HealthStatus, PredictRequest, PredictResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (import.meta.env.DEV) {
      console.error("[API]", error.config?.url, error.message);
    }
    return Promise.reject(error);
  },
);

export const api = {
  predict(data: PredictRequest): Promise<PredictResponse> {
    return apiClient.post<PredictResponse>("/api/v1/predict", data).then((r) => r.data);
  },
  health(): Promise<HealthStatus> {
    return apiClient.get<HealthStatus>("/api/v1/health").then((r) => r.data);
  },
};
