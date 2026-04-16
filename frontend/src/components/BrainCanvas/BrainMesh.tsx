import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import type { MutableRefObject } from 'react'
import * as THREE from 'three'
import { ROI_BBOX_FRACTION, type RoiKey } from '../../lib/roiCentroids'
import type { RoiValues } from '../../types/artifact'
import { createBrainGlowMaterial } from './brainShaderMaterial'

const ROI_KEYS: RoiKey[] = ['ffa', 'vmpfc', 'ifg', 'insula']

export function BrainMesh({ roiValuesRef }: { roiValuesRef: MutableRefObject<RoiValues> }) {
  const { scene } = useGLTF('/brain.glb')
  const rootRef = useRef<THREE.Group>(null)
  const clone = useMemo(() => scene.clone(true), [scene])
  const material = useMemo(() => createBrainGlowMaterial(), [])
  const uniformsRef = useRef<Record<string, { value: THREE.Vector3 | number }> | null>(null)
  const localsRef = useRef<Record<RoiKey, THREE.Vector3> | null>(null)
  const tmp = useMemo(() => new THREE.Vector3(), [])

  useLayoutEffect(() => {
    clone.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.castShadow = true
        o.receiveShadow = true
        o.material = material
      }
    })
    const box = new THREE.Box3().setFromObject(clone)
    const c = box.getCenter(new THREE.Vector3())
    clone.position.sub(c)
    const box2 = new THREE.Box3().setFromObject(clone)
    const size = box2.getSize(new THREE.Vector3())
    const half = size.clone().multiplyScalar(0.5)
    const loc: Record<RoiKey, THREE.Vector3> = {
      ffa: new THREE.Vector3(),
      vmpfc: new THREE.Vector3(),
      ifg: new THREE.Vector3(),
      insula: new THREE.Vector3(),
    }
    for (const k of ROI_KEYS) {
      const f = ROI_BBOX_FRACTION[k]
      loc[k].set(f.x * half.x, f.y * half.y, f.z * half.z)
    }
    localsRef.current = loc
    uniformsRef.current = material.userData.neuroUniforms as Record<
      string,
      { value: THREE.Vector3 | number }
    >
  }, [clone, material])

  useFrame(() => {
    const locs = localsRef.current
    const u = uniformsRef.current
    if (!locs || !u) {
      return
    }
    clone.updateMatrixWorld(true)
    const roi = roiValuesRef.current
    for (const k of ROI_KEYS) {
      tmp.copy(locs[k]).applyMatrix4(clone.matrixWorld)
      const uc = u[`uCenter_${k}`]
      if (uc && uc.value instanceof THREE.Vector3) {
        uc.value.copy(tmp)
      }
      const ui = u[`uIntensity_${k}`]
      if (ui && typeof ui.value === 'number') {
        const n = ui as { value: number }
        n.value = roi[k]
      }
    }
  })

  return (
    <group ref={rootRef} scale={1.45}>
      <primitive object={clone} />
    </group>
  )
}

useGLTF.preload('/brain.glb')
