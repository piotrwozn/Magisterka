# SOR-AI Frontend

System wspomagania decyzji triażowych SOR — interfejs prezentacyjny.

Etykieta: **Emergency Severity Index (ESI 1–5)**. API zwraca indeks klasy 0–4 = ESI 1–5;
kolory TOPSOR (czerwony, pomarańczowy, żółty, zielony, niebieski) i maksymalny czas
oczekiwania (0 / 10 / 60 / 120 / 240 min) służą wyłącznie do wyświetlania
(`ESI_LEVELS` w `src/lib/constants.ts`, komponent `ESILevelBadge`).

## Stack

React 18 · TypeScript · Vite · TailwindCSS · Framer Motion · three.js · react-i18next

## Uruchomienie

```bash
npm install
npm run dev      # http://localhost:5173
```

## Build produkcyjny

```bash
npm run build
npm run preview
```

## Struktura

Patrz [PLAN.md](./PLAN.md) — szczegółowy plan z statusem implementacji.

## Multilingual

PL (default) + EN. Język wykrywany z przeglądarki, zapisywany w localStorage. Switcher w Navbar.

## Konfiguracja API

`.env.local`:

```
VITE_API_URL=http://localhost:8000
```

Gdy backend niedostępny — demo działa na mockowanych danych (`src/lib/mockData.ts`).

## Wyniki modeli

Metryki modeli (`MODELS`) i ensembla (`RESULTS`) w `src/lib/constants.ts` są puste
(„—”) do czasu treningu końcowego. Po treningu uzupełnij je wartościami z
`training/runs/<run>/metrics.json` (zbiór testowy; wagi ensembla wybrane na walidacji).

## Kontrakt API

`src/lib/types.ts` = `backend/app/models/schemas.py` (camelCase). Opcjonalne pola
żądania: `sex` (`"M"` | `"F"`), `arrivalMode` (`"ambulance"` | `"self"` | `"other"`),
`o2Device` (boolean) — formularz demo ma je w sekcji „Kontekst (opcjonalnie)”.
