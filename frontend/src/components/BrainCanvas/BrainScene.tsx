import { OrbitControls } from '@react-three/drei'
import type { MutableRefObject } from 'react'
import type { RoiValues } from '../../types/artifact'
import { RoiMesh } from './RoiMesh'

export function BrainScene({
  roiValuesRef,
}: {
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  return (
    <>
      <OrbitControls enableDamping dampingFactor={0.08} />
      <RoiMesh roiKey="ffa" roiValuesRef={roiValuesRef} position={[-1.1, 0.2, 0.6]} color="#6cf" />
      <RoiMesh roiKey="vmpfc" roiValuesRef={roiValuesRef} position={[0.2, 1.2, -0.4]} color="#f8c" />
      <RoiMesh roiKey="ifg" roiValuesRef={roiValuesRef} position={[1.4, 0.1, 0.5]} color="#fc6" />
      <RoiMesh roiKey="insula" roiValuesRef={roiValuesRef} position={[1.5, -0.6, 0.2]} color="#9f6" />
    </>
  )
}
