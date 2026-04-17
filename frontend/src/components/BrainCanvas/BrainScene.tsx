import { Environment, OrbitControls } from '@react-three/drei'
import { useState, type MutableRefObject } from 'react'
import type { RoiValues } from '../../types/artifact'
import { BrainMesh } from './BrainMesh'

export function BrainScene({
  roiValuesRef,
}: {
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  const [autoRotate, setAutoRotate] = useState(true)

  return (
    <>
      <Environment preset="studio" environmentIntensity={0.72} resolution={128} />
      <ambientLight intensity={0.1} />
      <hemisphereLight color="#c2cfe0" groundColor="#080a0d" intensity={0.32} />
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.35}
        onStart={() => setAutoRotate(false)}
      />
      <BrainMesh roiValuesRef={roiValuesRef} />
    </>
  )
}
