import { useMutation } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { mockPredict } from "@/lib/mockData";
import type { PredictRequest, PredictResponse } from "@/lib/types";

/**
 * VITE_USE_MOCK controls behaviour:
 *   - "true"  → always use mock data (offline demo).
 *   - "false" → call the real backend; fall back to mock only on network/5xx errors.
 *   - unset   → defaults to "false" (production).
 *
 * VITE_API_URL points the API client at the deployed backend. If unset the
 * api module defaults to http://localhost:8000.
 */
const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

async function predict(req: PredictRequest): Promise<PredictResponse> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 600 + Math.random() * 700));
    return mockPredict(req);
  }
  try {
    return await api.predict(req);
  } catch (err) {
    if (import.meta.env.DEV) {
      console.warn("[predict] backend unreachable, falling back to mock", err);
    }
    await new Promise((r) => setTimeout(r, 400));
    return mockPredict(req);
  }
}

export function usePredict() {
  return useMutation({ mutationFn: predict });
}
