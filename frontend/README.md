# SOR-AI Frontend

System wspomagania decyzji triażowych SOR — interfejs prezentacyjny.

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
