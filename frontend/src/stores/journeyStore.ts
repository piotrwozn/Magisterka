import { create } from "zustand";

interface JourneyState {
  autoPlay: boolean;
  toggleAutoPlay: () => void;
  setAutoPlay: (v: boolean) => void;
}

export const useJourneyStore = create<JourneyState>((set) => ({
  autoPlay: false,
  toggleAutoPlay: () => set((s) => ({ autoPlay: !s.autoPlay })),
  setAutoPlay: (autoPlay) => set({ autoPlay }),
}));
