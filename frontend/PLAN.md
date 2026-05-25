# SOR-AI Frontend — Plan Implementacji

> System wspomagania decyzji triażowych SOR oparty na ML + LLM.
> Strona prezentacyjna pracy magisterskiej z sekcją demo na żywo.
> **Multilingual: PL (default) + EN. Production-grade React 18 + TypeScript.**

---

## Status

**Legenda:** ✅ done · 🚧 in progress · ⏳ pending

**🎉 ALL CORE FEATURES IMPLEMENTED — production-ready build verified.**

**Verified:**
- ✅ `npm install` — 313 packages, no errors
- ✅ `tsc --noEmit` — 0 type errors (strict mode + noUncheckedIndexedAccess)
- ✅ `vite build` — 2939 modules transformed in 9.33s
- ✅ `vite dev` — server ready in 353ms
- ✅ Bundle: 28KB initial gzip, three.js + charts lazy-loaded
- ✅ Code splitting: react/three/motion/charts/i18n in separate chunks

---

## 1. Stack technologiczny

| Warstwa | Technologia | Status |
|---|---|---|
| **Framework** | React 18 + TypeScript (strict) | ✅ |
| **Build** | Vite 5 | ✅ |
| **Styling** | TailwindCSS 3 + CSS vars | ✅ |
| **Animacje** | Framer Motion 11 | ✅ |
| **3D** | three.js + @react-three/fiber + drei | ✅ |
| **i18n** | react-i18next (PL/EN) | ✅ |
| **Wykresy** | Recharts | ✅ |
| **Routing** | React Router v6 | ✅ |
| **State** | Zustand | ✅ |
| **API** | Axios + TanStack Query | ✅ |
| **Forms** | react-hook-form + zod | ✅ |
| **Ikony** | lucide-react | ✅ |

---

## 2. Struktura — co zbudowane

### Config files
- ✅ `package.json` — dependencies + scripts
- ✅ `tsconfig.json` — strict mode, path aliases
- ✅ `tsconfig.node.json`
- ✅ `vite.config.ts` — aliasy `@/*`
- ✅ `tailwind.config.ts` — design tokens, dark mode
- ✅ `postcss.config.js`
- ✅ `index.html` — meta tags, fonts, dark-mode init
- ✅ `.gitignore`
- ✅ `.env.example`

### Core
- ✅ `src/main.tsx` — entry point + providers
- ✅ `src/App.tsx` — routing + layout
- ✅ `src/vite-env.d.ts`
- ✅ `src/styles/globals.css` — Tailwind + CSS vars

### Lib
- ✅ `src/lib/utils.ts` — cn() helper, format helpers
- ✅ `src/lib/constants.ts` — MTS categories, colors, models
- ✅ `src/lib/types.ts` — TS types z backend schema
- ✅ `src/lib/api.ts` — axios + endpoints
- ✅ `src/lib/i18n.ts` — react-i18next config
- ✅ `src/lib/mockData.ts` — fallback data dla demo

### Locales
- ✅ `src/locales/pl/common.json`
- ✅ `src/locales/en/common.json`
- ✅ `src/locales/pl/sections.json` (all sections in one file)
- ✅ `src/locales/en/sections.json`

### Hooks
- ✅ `src/hooks/usePredict.ts`
- ✅ `src/hooks/useScrollPosition.ts`
- ✅ `src/hooks/useReducedMotion.ts`
- ✅ `src/hooks/useTheme.ts`

### Stores
- ✅ `src/stores/demoStore.ts`
- ✅ `src/stores/themeStore.ts`

### UI primitives (shadcn-style)
- ✅ `Button`, `Card`, `Input`, `Label`, `Badge`, `Textarea`, `Tooltip`, `Progress`, `Separator`, `Skeleton`, `Dialog`

### Layout
- ✅ `Navbar` — sticky, glass effect, scroll-aware
- ✅ `Footer`
- ✅ `ScrollProgress` — top progress bar
- ✅ `LanguageSwitcher` (PL/EN)
- ✅ `ThemeToggle` (dark/light/system)

### Animations
- ✅ `NeuralNetworkScene` — three.js animated network (hero)
- ✅ `ParticleField` — canvas particles background
- ✅ `FadeInOnScroll`
- ✅ `TypewriterText`
- ✅ `CountUp`
- ✅ `PulsingDot`
- ✅ `GradientOrb`
- ✅ `MagneticButton`

### Shared
- ✅ `MTSCategoryBadge` — kolorowane chipy 5 kategorii
- ✅ `MetricCard`
- ✅ `SectionHeader`
- ✅ `CodeBlock`

### Sections (one-page scroll)
- ✅ `Hero` — three.js neural net + typewriter + CTA
- ✅ `ProblemStatement` — MTS 5 kategorii, statystyki SOR
- ✅ `DataSection` — Yale EMMLC, 336 cech, pie chart klas
- ✅ `ArchitectureDiagram` — animowany flow 3 warstw (SVG paths)
- ✅ `ModelsShowcase` — 7 modeli z metrykami
- ✅ `ResultsSection` — confusion matrix, ROC, big numbers
- ✅ `HowItWorks` — scrollytelling pipeline
- ✅ `DemoSection` — full interaktywne demo
- ✅ `TechStack` — logos grid
- ✅ `Timeline` — historia projektu
- ✅ `About`

### Demo components
- ✅ `PatientForm` — react-hook-form + zod
- ✅ `VitalsInput` — sliders + number inputs
- ✅ `ClinicalNoteInput` — textarea + przykłady
- ✅ `PredictionResult` — animated final category
- ✅ `ModelComparison` — co każdy model powiedział
- ✅ `ShapWaterfall` — top 5 cech, animated bars
- ✅ `ConfidenceGauge` — radial gauge animated
- ✅ `ConflictDetection` — alert UI

### Pages
- ✅ `Home` — one-page (wszystkie sekcje)
- ✅ `Demo` — standalone demo
- ✅ `Models` — deep dive 7 modeli
- ✅ `About` — kontakt + GitHub
- ✅ `NotFound`

---

## 3. Animacje — reguły

- ✅ 60fps minimum, GPU-accelerated transforms
- ✅ `prefers-reduced-motion: reduce` respektowany globalnie
- ✅ three.js scene lazy-loaded (Intersection Observer)
- ✅ Throttled scroll handlers (rAF)
- ✅ Code splitting per route

---

## 4. i18n — multilingual

**Języki:** PL (default) + EN.

**Library:** `react-i18next` + `i18next-browser-languagedetector`.

**Persistance:** localStorage `i18nextLng`.

**Namespaces:** `common`, `sections` (po jednym pliku per język dla prostoty).

**Switcher:** w Navbar (PL/EN buttony).

---

## 5. Komendy

```bash
cd frontend
npm install                  # ~1-2 min
npm run dev                  # http://localhost:5173
npm run build                # production build → dist/
npm run preview              # serve dist/
npm run lint                 # eslint
npm run type-check           # tsc --noEmit
```

---

## 6. API integration

**Endpoints (mocked w demo gdy backend down):**

```ts
POST /api/v1/predict
GET  /api/v1/models/list
GET  /api/v1/health
```

**Fallback:** `src/lib/mockData.ts` — realistyczne dane oparte na rzeczywistych metrykach.

---

## 7. Deploy (przyszłość)

- Static build → Nginx/Caddy
- Backend osobno: Spring Boot (Java) + Python FastAPI mikroserwis
- Load balancer: Nginx
- Queue: Kafka między Java a Python ML
- Hostowane lokalnie na serwerze uczelni / Vercel preview

---

*Plan stworzony: 2026-05-23 · Implementacja zakończona: 2026-05-23*
