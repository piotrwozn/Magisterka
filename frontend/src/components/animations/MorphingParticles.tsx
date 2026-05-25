import { useFrame } from "@react-three/fiber";
import { Canvas } from "@react-three/fiber";
import type { MotionValue } from "framer-motion";
import { Suspense, useMemo, useRef } from "react";
import * as THREE from "three";

import { useReducedMotion } from "@/hooks/useReducedMotion";

const PARTICLE_COUNT = 2400;

/* ─────────── Shape generators ─────────── */

function neuralNet(n: number): Float32Array {
  const layers = [12, 22, 22, 12];
  const xs = [-3.8, -1.3, 1.3, 3.8];
  const out = new Float32Array(n * 3);
  const total = layers.reduce((a, b) => a + b);
  let start = 0;
  for (let l = 0; l < layers.length; l++) {
    const count = layers[l]!;
    const chunk = Math.round((count / total) * n);
    for (let i = 0; i < chunk; i++) {
      const idx = start + i;
      if (idx >= n) break;
      const y = ((i + 0.5) / chunk - 0.5) * 4.8;
      out[idx * 3] = xs[l]! + (Math.random() - 0.5) * 0.08;
      out[idx * 3 + 1] = y + (Math.random() - 0.5) * 0.05;
      out[idx * 3 + 2] = (Math.random() - 0.5) * 0.3;
    }
    start += chunk;
  }
  for (let i = start; i < n; i++) {
    out[i * 3] = (Math.random() - 0.5) * 7.5;
    out[i * 3 + 1] = (Math.random() - 0.5) * 4.8;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.6;
  }
  return out;
}

function sphere(n: number, radius = 2.8): Float32Array {
  const out = new Float32Array(n * 3);
  const phi = Math.PI * (Math.sqrt(5) - 1);
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = phi * i;
    out[i * 3] = Math.cos(theta) * r * radius;
    out[i * 3 + 1] = y * radius;
    out[i * 3 + 2] = Math.sin(theta) * r * radius;
  }
  return out;
}

function dnaHelix(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const turns = 5;
  const radius = 1.9;
  const height = 5.4;
  const half = n / 2;
  for (let i = 0; i < n; i++) {
    const strand = i < half ? 0 : 1;
    const t = (i % half) / half;
    const angle = t * turns * Math.PI * 2 + strand * Math.PI;
    out[i * 3] = Math.cos(angle) * radius;
    out[i * 3 + 1] = t * height - height / 2;
    out[i * 3 + 2] = Math.sin(angle) * radius;
  }
  return out;
}

function gridLattice(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const side = Math.ceil(Math.cbrt(n));
  const spacing = 5.2 / side;
  for (let i = 0; i < n; i++) {
    const x = i % side;
    const y = Math.floor(i / side) % side;
    const z = Math.floor(i / (side * side));
    out[i * 3] = (x - side / 2 + 0.5) * spacing;
    out[i * 3 + 1] = (y - side / 2 + 0.5) * spacing;
    out[i * 3 + 2] = (z - side / 2 + 0.5) * spacing;
  }
  return out;
}

function torus(n: number, R = 2.5, r = 0.9): Float32Array {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const u = (i / n) * Math.PI * 2;
    const v = (((i * 7) % n) / n) * Math.PI * 2;
    out[i * 3] = (R + r * Math.cos(v)) * Math.cos(u);
    out[i * 3 + 1] = r * Math.sin(v);
    out[i * 3 + 2] = (R + r * Math.cos(v)) * Math.sin(u);
  }
  return out;
}

function spiralGalaxy(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const arms = 4;
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const arm = i % arms;
    const radius = 0.35 + t * 3.2;
    const angle = t * 9 + (arm / arms) * Math.PI * 2;
    const jitter = (Math.random() - 0.5) * 0.25;
    out[i * 3] = Math.cos(angle + jitter) * radius;
    out[i * 3 + 1] = (Math.random() - 0.5) * 0.4 * (1 - t);
    out[i * 3 + 2] = Math.sin(angle + jitter) * radius;
  }
  return out;
}

function pyramid3layers(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const layers = 4;
  const perLayer = Math.floor(n / layers);
  for (let i = 0; i < n; i++) {
    const layer = Math.min(layers - 1, Math.floor(i / perLayer));
    const idxInLayer = i - layer * perLayer;
    const sideCount = perLayer;
    const radius = 2.8 - layer * 0.7;
    const angle = (idxInLayer / sideCount) * Math.PI * 2;
    out[i * 3] = Math.cos(angle) * radius;
    out[i * 3 + 1] = (layer - 1.5) * 1.2;
    out[i * 3 + 2] = Math.sin(angle) * radius;
  }
  return out;
}

function constellation7(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const clusters = 9;
  const centers: [number, number, number][] = [];
  for (let c = 0; c < clusters; c++) {
    const angle = (c / clusters) * Math.PI * 2;
    centers.push([
      Math.cos(angle) * 2.6,
      Math.sin(angle) * 2.6,
      (Math.random() - 0.5) * 1.6,
    ]);
  }
  for (let i = 0; i < n; i++) {
    const c = centers[i % clusters]!;
    out[i * 3] = c[0] + (Math.random() - 0.5) * 0.75;
    out[i * 3 + 1] = c[1] + (Math.random() - 0.5) * 0.75;
    out[i * 3 + 2] = c[2] + (Math.random() - 0.5) * 0.55;
  }
  return out;
}

function waveform(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = (i / n - 0.5) * 7.6;
    const y = Math.sin(x * 1.4) * Math.cos(x * 0.65) * 2;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.5;
  }
  return out;
}

function columns(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const cols = 8;
  const perCol = Math.floor(n / cols);
  const heights = [0.4, 0.7, 0.55, 0.85, 0.6, 0.75, 0.5, 0.65];
  for (let i = 0; i < n; i++) {
    const col = Math.min(cols - 1, Math.floor(i / perCol));
    const idx = i - col * perCol;
    const x = (col - cols / 2 + 0.5) * 0.85;
    const yMax = heights[col]! * 4.2;
    const y = (idx / perCol) * yMax - 1.9;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.25;
  }
  return out;
}

function heart(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const scale = 0.12;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const x = 16 * Math.pow(Math.sin(t), 3);
    const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    out[i * 3] = x * scale + (Math.random() - 0.5) * 0.3;
    out[i * 3 + 1] = y * scale + (Math.random() - 0.5) * 0.3;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.6;
  }
  return out;
}

function vortex(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const angle = t * 14;
    const radius = 0.2 + t * 3;
    const height = (t - 0.5) * 5;
    out[i * 3] = Math.cos(angle) * radius;
    out[i * 3 + 1] = height;
    out[i * 3 + 2] = Math.sin(angle) * radius;
  }
  return out;
}

function cube(n: number, size = 2.6): Float32Array {
  const out = new Float32Array(n * 3);
  const perEdge = Math.floor(n / 12);
  const edges: [number, number, number, number, number, number][] = [
    [-1, -1, -1, 1, -1, -1], [1, -1, -1, 1, 1, -1], [1, 1, -1, -1, 1, -1], [-1, 1, -1, -1, -1, -1],
    [-1, -1, 1, 1, -1, 1], [1, -1, 1, 1, 1, 1], [1, 1, 1, -1, 1, 1], [-1, 1, 1, -1, -1, 1],
    [-1, -1, -1, -1, -1, 1], [1, -1, -1, 1, -1, 1], [1, 1, -1, 1, 1, 1], [-1, 1, -1, -1, 1, 1],
  ];
  for (let i = 0; i < n; i++) {
    const edgeIdx = Math.min(11, Math.floor(i / perEdge));
    const t = (i % perEdge) / perEdge;
    const e = edges[edgeIdx]!;
    out[i * 3] = (e[0] + (e[3] - e[0]) * t) * size + (Math.random() - 0.5) * 0.05;
    out[i * 3 + 1] = (e[1] + (e[4] - e[1]) * t) * size + (Math.random() - 0.5) * 0.05;
    out[i * 3 + 2] = (e[2] + (e[5] - e[2]) * t) * size + (Math.random() - 0.5) * 0.05;
  }
  return out;
}

function starburst(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const rays = 8;
  const perRay = Math.floor(n / rays);
  for (let i = 0; i < n; i++) {
    const ray = Math.min(rays - 1, Math.floor(i / perRay));
    const t = (i % perRay) / perRay;
    const angle = (ray / rays) * Math.PI * 2;
    const radius = 0.2 + t * 3.2;
    out[i * 3] = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.15;
    out[i * 3 + 1] = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.15;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
  }
  return out;
}

function ring(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const rings = 3;
  for (let i = 0; i < n; i++) {
    const r = i % rings;
    const radius = 1.3 + r * 0.9;
    const angle = (i / n) * Math.PI * 2 * 4;
    out[i * 3] = Math.cos(angle) * radius;
    out[i * 3 + 1] = (r - 1) * 0.6 + Math.sin(angle * 3) * 0.1;
    out[i * 3 + 2] = Math.sin(angle) * radius;
  }
  return out;
}

function lattice2D(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const side = Math.ceil(Math.sqrt(n));
  const spacing = 5.4 / side;
  for (let i = 0; i < n; i++) {
    const x = i % side;
    const y = Math.floor(i / side);
    out[i * 3] = (x - side / 2 + 0.5) * spacing;
    out[i * 3 + 1] = (y - side / 2 + 0.5) * spacing;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.2;
  }
  return out;
}

function flower(n: number): Float32Array {
  const out = new Float32Array(n * 3);
  const petals = 6;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const r = 2.4 * Math.cos(petals * t) + 0.6;
    out[i * 3] = Math.cos(t) * r;
    out[i * 3 + 1] = Math.sin(t) * r;
    out[i * 3 + 2] = (Math.random() - 0.5) * 0.4;
  }
  return out;
}

const GENERATORS = {
  neuralNet,
  sphere,
  helix: dnaHelix,
  grid: gridLattice,
  torus,
  spiral: spiralGalaxy,
  pyramid: pyramid3layers,
  constellation: constellation7,
  waveform,
  columns,
  heart,
  vortex,
  cube,
  starburst,
  ring,
  lattice2D,
  flower,
} as const;

export type ShapeName = keyof typeof GENERATORS;

/* ─────────── Color palettes per scene (8 colors each) ─────────── */

const PALETTES: string[][] = [
  ["#38bdf8", "#06b6d4", "#0ea5e9", "#22d3ee", "#67e8f9", "#0891b2", "#0284c7", "#7dd3fc"], // intro cyan
  ["#f97316", "#fb923c", "#fbbf24", "#fde047", "#ef4444", "#f87171", "#fca5a5", "#fecaca"], // problem fire
  ["#10b981", "#22c55e", "#34d399", "#84cc16", "#65a30d", "#16a34a", "#15803d", "#4ade80"], // data green
  ["#a855f7", "#c084fc", "#d946ef", "#e879f9", "#9333ea", "#7e22ce", "#f0abfc", "#f5d0fe"], // features purple
  ["#ec4899", "#f472b6", "#f43f5e", "#fb7185", "#e11d48", "#be185d", "#fda4af", "#fbcfe8"], // model pink
  ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#7c3aed", "#5b21b6", "#818cf8", "#c4b5fd"], // results blue→purple
  ["#06b6d4", "#0ea5e9", "#38bdf8", "#7dd3fc", "#22d3ee", "#67e8f9", "#a5f3fc", "#cffafe"], // demo cyan
  ["#f59e0b", "#fbbf24", "#fde047", "#fef08a", "#eab308", "#ca8a04", "#facc15", "#fef9c3"], // tech yellow
  ["#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6", "#2dd4bf", "#5eead4", "#99f6e4", "#0891b2"], // timeline teal→blue
  ["#e11d48", "#f43f5e", "#ec4899", "#a855f7", "#be185d", "#9d174d", "#fb7185", "#f9a8d4"], // rose→pink
  ["#8b5cf6", "#a855f7", "#c084fc", "#e879f9", "#6d28d9", "#7c3aed", "#d8b4fe", "#f0abfc"], // violet
  ["#22d3ee", "#06b6d4", "#0ea5e9", "#38bdf8", "#67e8f9", "#a5f3fc", "#7dd3fc", "#0284c7"], // ending cyan
];

const COLOR_GROUPS = 8;

function lerpColor(a: THREE.Color, b: THREE.Color, t: number, out: THREE.Color) {
  out.r = a.r + (b.r - a.r) * t;
  out.g = a.g + (b.g - a.g) * t;
  out.b = a.b + (b.b - a.b) * t;
}

/* ─────────── Easing ─────────── */

function smoothstep(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

/* ─────────── Particle system ─────────── */

interface ParticleSystemProps {
  progress: MotionValue<number>;
  shapes: ShapeName[];
}

function ParticleSystem({ progress, shapes }: ParticleSystemProps) {
  const ref = useRef<THREE.Points>(null);
  const reduced = useReducedMotion();
  const stageCount = shapes.length;

  const stagePositions = useMemo(
    () => shapes.map((s) => GENERATORS[s](PARTICLE_COUNT)),
    [shapes],
  );

  // Per-particle color group index + per-particle brightness/size variation
  const colorGroups = useMemo(() => {
    const arr = new Uint8Array(PARTICLE_COUNT);
    for (let i = 0; i < PARTICLE_COUNT; i++) arr[i] = i % COLOR_GROUPS;
    return arr;
  }, []);

  const brightnessJitter = useMemo(() => {
    const arr = new Float32Array(PARTICLE_COUNT);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      arr[i] = 0.75 + Math.random() * 0.5; // 0.75-1.25
    }
    return arr;
  }, []);

  const palettes = useMemo(
    () =>
      PALETTES.map((p) => p.map((c) => new THREE.Color(c))) as THREE.Color[][],
    [],
  );

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(stagePositions[0]!.slice(), 3),
    );
    const colors = new Float32Array(PARTICLE_COUNT * 3);
    const pal = palettes[0]!;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const c = pal[colorGroups[i]!]!;
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return g;
  }, [stagePositions, palettes, colorGroups]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.07,
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
        sizeAttenuation: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const current = useRef(0);
  const tmpColor = useRef(new THREE.Color());

  useFrame((state, delta) => {
    if (!ref.current) return;
    const p = progress.get();
    const target = reduced ? 0 : p * (stageCount - 1);
    current.current += (target - current.current) * Math.min(1, delta * 4);

    const stage = current.current;
    const i0 = Math.floor(stage);
    const i1 = Math.min(stageCount - 1, i0 + 1);
    const t = smoothstep(stage - i0);

    const positions = ref.current.geometry.attributes.position!.array as Float32Array;
    const colorsAttr = ref.current.geometry.attributes.color!.array as Float32Array;
    const from = stagePositions[i0]!;
    const to = stagePositions[i1]!;

    // Color interpolation - between palettes
    const paletteCount = palettes.length;
    const palT = p * (paletteCount - 1);
    const pi0 = Math.floor(palT);
    const pi1 = Math.min(paletteCount - 1, pi0 + 1);
    const palBlend = smoothstep(palT - pi0);
    const palA = palettes[pi0]!;
    const palB = palettes[pi1]!;
    const time = state.clock.elapsedTime;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Positions: morph between stages + slight breathing
      const breathe = Math.sin(time * 0.6 + i * 0.05) * 0.02;
      const ix = i * 3;
      positions[ix] = from[ix]! + (to[ix]! - from[ix]!) * t + breathe;
      positions[ix + 1] =
        from[ix + 1]! + (to[ix + 1]! - from[ix + 1]!) * t + breathe * 0.7;
      positions[ix + 2] = from[ix + 2]! + (to[ix + 2]! - from[ix + 2]!) * t;

      // Color: blend between palettes + per-particle brightness jitter + sparkle
      const cg = colorGroups[i]!;
      lerpColor(palA[cg]!, palB[cg]!, palBlend, tmpColor.current);
      const sparkle = 1 + Math.sin(time * 1.5 + i * 0.3) * 0.15;
      const bright = brightnessJitter[i]! * sparkle;
      colorsAttr[ix] = tmpColor.current.r * bright;
      colorsAttr[ix + 1] = tmpColor.current.g * bright;
      colorsAttr[ix + 2] = tmpColor.current.b * bright;
    }

    ref.current.geometry.attributes.position!.needsUpdate = true;
    ref.current.geometry.attributes.color!.needsUpdate = true;

    if (!reduced) {
      ref.current.rotation.y = time * 0.06;
      ref.current.rotation.x = Math.sin(time * 0.08) * 0.18;
      ref.current.rotation.z = Math.cos(time * 0.04) * 0.05;
    }
  });

  return <points ref={ref} geometry={geometry} material={material} />;
}

/* ─────────── Wireframe sphere (color shifts with progress) ─────────── */

function ConnectionLines({ progress }: { progress: MotionValue<number> }) {
  const ref = useRef<THREE.Mesh>(null);
  const tmp = useRef(new THREE.Color());
  const palettes = useMemo(
    () => PALETTES.map((p) => new THREE.Color(p[0])),
    [],
  );

  useFrame(() => {
    if (!ref.current) return;
    const p = progress.get();
    const palT = p * (palettes.length - 1);
    const i0 = Math.floor(palT);
    const i1 = Math.min(palettes.length - 1, i0 + 1);
    const blend = smoothstep(palT - i0);
    tmp.current.copy(palettes[i0]!).lerp(palettes[i1]!, blend);
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.color.copy(tmp.current);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[3.9, 28, 28]} />
      <meshBasicMaterial color="#38bdf8" wireframe transparent opacity={0.05} />
    </mesh>
  );
}

/* ─────────── Public component ─────────── */

interface MorphingParticlesProps {
  progress: MotionValue<number>;
  shapes: ShapeName[];
}

export function MorphingParticles({ progress, shapes }: MorphingParticlesProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 8], fov: 50 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      aria-hidden
    >
      <Suspense fallback={null}>
        <ConnectionLines progress={progress} />
        <ParticleSystem progress={progress} shapes={shapes} />
      </Suspense>
    </Canvas>
  );
}
