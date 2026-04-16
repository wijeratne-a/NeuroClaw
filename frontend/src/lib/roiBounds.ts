import type { ParsedArtifact, RoiValues } from '../types/artifact'

export type RoiBounds = {
  ffa: [number, number]
  vmpfc: [number, number]
  ifg: [number, number]
  insula: [number, number]
}

function minMax(arr: Float32Array): [number, number] {
  if (arr.length === 0) {
    return [0, 1]
  }
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (v < lo) {
      lo = v
    }
    if (v > hi) {
      hi = v
    }
  }
  if (!Number.isFinite(lo) || lo === hi) {
    return [lo, lo + 1e-6]
  }
  return [lo, hi]
}

export function computeRoiBounds(art: ParsedArtifact): RoiBounds {
  return {
    ffa: minMax(art.ffa),
    vmpfc: minMax(art.vmpfc),
    ifg: minMax(art.ifg),
    insula: minMax(art.insula),
  }
}

export function normalizeRoiValue(v: number, bounds: [number, number]): number {
  const [lo, hi] = bounds
  if (hi <= lo) {
    return 0.5
  }
  return Math.min(1, Math.max(0, (v - lo) / (hi - lo)))
}

export function normalizeRois(raw: RoiValues, b: RoiBounds): RoiValues {
  return {
    ffa: normalizeRoiValue(raw.ffa, b.ffa),
    vmpfc: normalizeRoiValue(raw.vmpfc, b.vmpfc),
    ifg: normalizeRoiValue(raw.ifg, b.ifg),
    insula: normalizeRoiValue(raw.insula, b.insula),
  }
}
