import { OrbitControls } from '@react-three/drei'
import type { MutableRefObject } from 'react'
import type { RoiValues } from '../../types/artifact'
import { BrainMesh } from './BrainMesh'

export function BrainScene({
  roiValuesRef,
}: {
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  return (
    <>
      <OrbitControls enableDamping dampingFactor={0.08} />
      <BrainMesh roiValuesRef={roiValuesRef} />
    </>
  )
}
