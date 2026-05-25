import { create } from "zustand";

import { VITALS_RANGES } from "@/lib/constants";
import type { PredictResponse, Vitals } from "@/lib/types";

interface DemoState {
  vitals: Vitals;
  clinicalNote: string;
  result: PredictResponse | null;
  setVitals: (vitals: Partial<Vitals>) => void;
  setClinicalNote: (note: string) => void;
  setResult: (result: PredictResponse | null) => void;
  reset: () => void;
  loadExample: (vitals: Vitals, note: string) => void;
}

const defaultVitals: Vitals = {
  age: VITALS_RANGES.age.default,
  temp: VITALS_RANGES.temp.default,
  hr: VITALS_RANGES.hr.default,
  sbp: VITALS_RANGES.sbp.default,
  dbp: VITALS_RANGES.dbp.default,
  rr: VITALS_RANGES.rr.default,
  o2: VITALS_RANGES.o2.default,
};

export const useDemoStore = create<DemoState>((set) => ({
  vitals: { ...defaultVitals },
  clinicalNote: "",
  result: null,
  setVitals: (vitals) => set((s) => ({ vitals: { ...s.vitals, ...vitals } })),
  setClinicalNote: (clinicalNote) => set({ clinicalNote }),
  setResult: (result) => set({ result }),
  reset: () => set({ vitals: { ...defaultVitals }, clinicalNote: "", result: null }),
  loadExample: (vitals, clinicalNote) => set({ vitals, clinicalNote, result: null }),
}));
