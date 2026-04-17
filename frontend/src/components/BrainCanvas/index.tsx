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
      <div className="brain-label">fsaverage5 — clear glass · live ROI glow (video sync)</div>
      <Canvas
        className="brain-canvas"
        camera={{ position: [0, 0, 5.6], fov: 36 }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          stencil: false,
          depth: true,
          preserveDrawingBuffer: false,
        }}
        dpr={[1, 1.5]}
        frameloop="always"
      >
        <color attach="background" args={['#0f1218']} />
        <Suspense fallback={null}>
          <BrainScene roiValuesRef={roiValuesRef} />
        </Suspense>
      </Canvas>
    </div>
  )
}
