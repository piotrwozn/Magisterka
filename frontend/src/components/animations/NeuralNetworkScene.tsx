import { useFrame } from "@react-three/fiber";
import { Canvas } from "@react-three/fiber";
import { Suspense, useMemo, useRef } from "react";
import * as THREE from "three";

import { useReducedMotion } from "@/hooks/useReducedMotion";

function NeuralNetwork() {
  const groupRef = useRef<THREE.Group>(null);
  const reduced = useReducedMotion();

  const { nodes, edges } = useMemo(() => {
    const layers = [4, 8, 8, 5];
    const xs = [-3, -1, 1, 3];
    const nodes: { position: THREE.Vector3; layer: number }[] = [];

    for (let l = 0; l < layers.length; l++) {
      const n = layers[l]!;
      for (let i = 0; i < n; i++) {
        const y = (i - (n - 1) / 2) * 0.65;
        nodes.push({ position: new THREE.Vector3(xs[l]!, y, 0), layer: l });
      }
    }

    const edges: [number, number][] = [];
    let offset = 0;
    for (let l = 0; l < layers.length - 1; l++) {
      const a = layers[l]!;
      const b = layers[l + 1]!;
      for (let i = 0; i < a; i++) {
        for (let j = 0; j < b; j++) {
          edges.push([offset + i, offset + a + j]);
        }
      }
      offset += a;
    }

    return { nodes, edges };
  }, []);

  useFrame((state) => {
    if (!groupRef.current || reduced) return;
    const t = state.clock.elapsedTime;
    groupRef.current.rotation.y = Math.sin(t * 0.12) * 0.15;
    groupRef.current.rotation.x = Math.sin(t * 0.08) * 0.08;
  });

  return (
    <group ref={groupRef}>
      {edges.map(([i, j], idx) => {
        const a = nodes[i]!.position;
        const b = nodes[j]!.position;
        const positions = new Float32Array([a.x, a.y, a.z, b.x, b.y, b.z]);
        return (
          <line key={idx}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[positions, 3]}
                count={2}
                array={positions}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#38bdf8" transparent opacity={0.15} />
          </line>
        );
      })}

      {nodes.map((n, idx) => (
        <PulsatingSphere key={idx} position={n.position} layer={n.layer} reduced={reduced} />
      ))}
    </group>
  );
}

function PulsatingSphere({
  position,
  layer,
  reduced,
}: {
  position: THREE.Vector3;
  layer: number;
  reduced: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const seed = useMemo(() => Math.random() * Math.PI * 2, []);

  useFrame((state) => {
    if (!meshRef.current || reduced) return;
    const t = state.clock.elapsedTime;
    const pulse = 1 + Math.sin(t * 1.5 + seed) * 0.15;
    meshRef.current.scale.setScalar(pulse);
  });

  const color = layer === 0 ? "#38bdf8" : layer === 3 ? "#a855f7" : "#06b6d4";
  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

export function NeuralNetworkScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 50 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
      aria-hidden
    >
      <Suspense fallback={null}>
        <NeuralNetwork />
      </Suspense>
    </Canvas>
  );
}
