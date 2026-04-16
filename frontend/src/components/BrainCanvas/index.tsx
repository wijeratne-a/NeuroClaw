import { Canvas } from '@react-three/fiber'
import { Suspense, type MutableRefObject } from 'react'
import type { RoiValues } from '../../types/artifact'
import { BrainScene } from './BrainScene'
import './brain-canvas.css'

export function BrainCanvas({
  roiValuesRef,
}: {
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  return (
    <div className="brain-canvas-wrap">
      <div className="brain-label">Cortical Marketing Four — ROI glow (sample mesh)</div>
      <Canvas className="brain-canvas" camera={{ position: [0, 0, 7], fov: 42 }}>
        <color attach="background" args={['#0a0a0c']} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[5, 8, 6]} intensity={1.1} />
        <Suspense fallback={null}>
          <BrainScene roiValuesRef={roiValuesRef} />
        </Suspense>
      </Canvas>
    </div>
  )
}
