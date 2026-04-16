import type { ParsedArtifact, RoiValues, TranscriptSegment } from '../types/artifact'

/** Largest index i such that timestamps[i] <= t (sorted ascending). */
export function floorBinIndex(timestamps: Float64Array, t: number): number {
  const n = timestamps.length
  if (n === 0) {
    return 0
  }
  if (t < timestamps[0]) {
    return 0
  }
  if (t >= timestamps[n - 1]) {
    return n - 1
  }
  let lo = 0
  let hi = n - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (timestamps[mid] <= t) {
      lo = mid
    } else {
      hi = mid - 1
    }
  }
  return lo
}

function lerp(a: number, b: number, alpha: number): number {
  return a + (b - a) * alpha
}

export function interpolateRois(artifact: ParsedArtifact, displayTimeS: number): RoiValues {
  const { timestamps, ffa, vmpfc, ifg, insula } = artifact
  const n = timestamps.length
  if (n === 0) {
    return { ffa: 0, vmpfc: 0, ifg: 0, insula: 0 }
  }
  const i = floorBinIndex(timestamps, displayTimeS)
  const i2 = Math.min(i + 1, n - 1)
  const t0 = timestamps[i]
  const t1 = timestamps[i2]
  const denom = t1 - t0 || 1
  const alpha = t1 > t0 ? (displayTimeS - t0) / denom : 0
  return {
    ffa: lerp(ffa[i], ffa[i2], alpha),
    vmpfc: lerp(vmpfc[i], vmpfc[i2], alpha),
    ifg: lerp(ifg[i], ifg[i2], alpha),
    insula: lerp(insula[i], insula[i2], alpha),
  }
}

export function findActiveSegmentIndex(
  segments: TranscriptSegment[],
  t: number,
): number {
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i]
    if (t >= s.start && t < s.end) {
      return i
    }
  }
  return -1
}
