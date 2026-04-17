import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useMemo, useRef, useEffect } from 'react'
import type { MutableRefObject, RefObject } from 'react'
import * as THREE from 'three'
import { ROI_BBOX_FRACTION, type RoiKey } from '../../lib/roiCentroids'
import type { RoiValues } from '../../types/artifact'
import { createBrainGlowMaterial } from './brainShaderMaterial'

const ROI_KEYS: RoiKey[] = ['ffa', 'vmpfc', 'ifg', 'insula']

/** Larger value = brain fills more of the camera view (world units for max bbox edge). */
const FIT_TARGET_EXTENT = 2.95

const ROI_LIGHT_COLORS: Record<RoiKey, THREE.Color> = {
  ffa: new THREE.Color('#3b82f6'),
  vmpfc: new THREE.Color('#4ade80'),
  ifg: new THREE.Color('#a855f7'),
  insula: new THREE.Color('#f59e0b'),
}

type PreparedMesh = {
  clone: THREE.Object3D
  fitScale: number
  roiLocalPos: Record<RoiKey, [number, number, number]>
  locals: Record<RoiKey, THREE.Vector3>
  uniforms: Record<string, { value: THREE.Vector3 | number }>
}

export function BrainMesh({ roiValuesRef }: { roiValuesRef: MutableRefObject<RoiValues> }) {
  const { scene } = useGLTF('/brain.glb')
  const material = useMemo(() => createBrainGlowMaterial(), [])

  const prepared = useMemo((): PreparedMesh => {
    const clone = scene.clone(true)
    clone.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.castShadow = false
        o.receiveShadow = false
        o.material = material
      }
    })
    const box = new THREE.Box3().setFromObject(clone)
    const c = box.getCenter(new THREE.Vector3())
    clone.position.sub(c)
    const box2 = new THREE.Box3().setFromObject(clone)
    const size = box2.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z)
    const fitScale = maxDim > 1e-9 ? FIT_TARGET_EXTENT / maxDim : 0.012

    const half = size.clone().multiplyScalar(0.5)
    const locals: Record<RoiKey, THREE.Vector3> = {
      ffa: new THREE.Vector3(),
      vmpfc: new THREE.Vector3(),
      ifg: new THREE.Vector3(),
      insula: new THREE.Vector3(),
    }
    for (const k of ROI_KEYS) {
      const f = ROI_BBOX_FRACTION[k]
      locals[k].set(f.x * half.x, f.y * half.y, f.z * half.z)
    }

    const roiLocalPos: Record<RoiKey, [number, number, number]> = {
      ffa: [locals.ffa.x, locals.ffa.y, locals.ffa.z],
      vmpfc: [locals.vmpfc.x, locals.vmpfc.y, locals.vmpfc.z],
      ifg: [locals.ifg.x, locals.ifg.y, locals.ifg.z],
      insula: [locals.insula.x, locals.insula.y, locals.insula.z],
    }

    const uniforms = material.userData.neuroUniforms as Record<
      string,
      { value: THREE.Vector3 | number }
    >

    return { clone, fitScale, roiLocalPos, locals, uniforms }
  }, [scene, material])

  const { clone, fitScale, roiLocalPos } = prepared
  const tmp = useMemo(() => new THREE.Vector3(), [])

  const preparedRef = useRef(prepared)
  useEffect(() => {
    preparedRef.current = prepared
  }, [prepared])

  const ffaLRef = useRef<THREE.PointLight | null>(null)
  const vmpfcLRef = useRef<THREE.PointLight | null>(null)
  const ifgLRef = useRef<THREE.PointLight | null>(null)
  const insulaLRef = useRef<THREE.PointLight | null>(null)

  const lightRefs: Record<RoiKey, RefObject<THREE.PointLight | null>> = {
    ffa: ffaLRef,
    vmpfc: vmpfcLRef,
    ifg: ifgLRef,
    insula: insulaLRef,
  }

  useFrame(() => {
    const { clone: c, locals: loc, uniforms: u } = preparedRef.current
    c.updateMatrixWorld(true)
    const roi = roiValuesRef.current
    for (const k of ROI_KEYS) {
      tmp.copy(loc[k]).applyMatrix4(c.matrixWorld)
      const uc = u[`uCenter_${k}`]
      if (uc && uc.value instanceof THREE.Vector3) {
        uc.value.copy(tmp)
      }
      const ui = u[`uIntensity_${k}`]
      if (ui && typeof ui.value === 'number') {
        ;(ui as { value: number }).value = roi[k]
      }
    }
    const ffaL = ffaLRef.current
    const vmpfcL = vmpfcLRef.current
    const ifgL = ifgLRef.current
    const insulaL = insulaLRef.current
    /** Keep interior lights tightly localized to ROI cores. */
    const lightGain = 3.2
    const falloff = 1.45
    if (ffaL) {
      ffaL.intensity = Math.pow(Math.max(0, Math.min(1, roi.ffa)), falloff) * lightGain
    }
    if (vmpfcL) {
      vmpfcL.intensity = Math.pow(Math.max(0, Math.min(1, roi.vmpfc)), falloff) * lightGain
    }
    if (ifgL) {
      ifgL.intensity = Math.pow(Math.max(0, Math.min(1, roi.ifg)), falloff) * lightGain
    }
    if (insulaL) {
      insulaL.intensity = Math.pow(Math.max(0, Math.min(1, roi.insula)), falloff) * lightGain
    }
  })

  return (
    <group scale={fitScale}>
      <primitive object={clone} />
      {ROI_KEYS.map((k) => (
        <pointLight
          key={k}
          ref={lightRefs[k]}
          position={roiLocalPos[k]}
          color={ROI_LIGHT_COLORS[k]}
          intensity={0}
          distance={4.8}
          decay={2.8}
        />
      ))}
    </group>
  )
}

