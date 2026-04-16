import { useFrame } from '@react-three/fiber'
import { useRef, type MutableRefObject } from 'react'
import type { Mesh, MeshStandardMaterial } from 'three'
import type { RoiValues } from '../../types/artifact'

type RoiKey = keyof RoiValues

const scale = 2.5

/** v is normalized to ~[0, 1] per ROI (see useVideoSync + roiBounds). */
function intensityFromValue(v: number): number {
  return 0.25 + v * 2.2
}

export function RoiMesh({
  position,
  color,
  roiKey,
  roiValuesRef,
}: {
  position: [number, number, number]
  color: string
  roiKey: RoiKey
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  const meshRef = useRef<Mesh>(null)
  const matRef = useRef<MeshStandardMaterial>(null)

  useFrame(() => {
    const mat = matRef.current
    if (!mat) {
      return
    }
    const v = roiValuesRef.current[roiKey]
    mat.emissiveIntensity = intensityFromValue(Math.max(0, Math.min(1, v)))
  })

  return (
    <mesh ref={meshRef} position={position.map((p) => p * scale) as [number, number, number]}>
      <sphereGeometry args={[0.42, 32, 32]} />
      <meshStandardMaterial
        ref={matRef}
        color={color}
        emissive={color}
        emissiveIntensity={0.2}
        metalness={0.2}
        roughness={0.6}
      />
    </mesh>
  )
}
