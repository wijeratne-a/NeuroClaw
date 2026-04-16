/**
 * ROI glow epicentres as fractional offsets from the mesh bounding-box center.
 * Each component is multiplied by half the corresponding axis size (local space, model centered at origin).
 * Tuned for a vertically oriented brain stem / cortical preview mesh.
 */
export const ROI_BBOX_FRACTION = {
  ffa: { x: -0.35, y: -0.25, z: 0.28 },
  vmpfc: { x: 0.05, y: 0.42, z: 0.18 },
  ifg: { x: 0.42, y: 0.12, z: 0.12 },
  insula: { x: 0.38, y: 0.02, z: 0.22 },
} as const

export type RoiKey = keyof typeof ROI_BBOX_FRACTION
