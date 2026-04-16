import { useEffect, useRef, useState } from 'react'
import type { ParsedArtifact, RoiValues, TranscriptSegment } from '../types/artifact'
import { computeRoiBounds, normalizeRois, type RoiBounds } from '../lib/roiBounds'
import { findActiveSegmentIndex, interpolateRois } from '../lib/syncMath'

export type VideoSyncRefs = {
  roiValuesRef: React.MutableRefObject<RoiValues>
  pulseOpacityRef: React.MutableRefObject<number>
}

export type UseVideoSyncOptions = {
  /** When true, neural time = max(0, videoTime - hemodynamicOffset). When false, neural time = videoTime (scrub-aligned). Default false for clearer UX. */
  applyHemodynamicDelay: boolean
}

/**
 * Drives 60Hz sync from video.currentTime. Mutates refs only (no per-frame React state).
 * ROI values in roiValuesRef are normalized to ~[0,1] per ROI for display (brain emissive).
 */
export function useVideoSync(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  artifact: ParsedArtifact | null,
  fallbackArtifact: ParsedArtifact | null,
  segments: TranscriptSegment[],
  options: UseVideoSyncOptions,
): VideoSyncRefs & { activeSegmentIndex: number } {
  const { applyHemodynamicDelay } = options

  const roiValuesRef = useRef<RoiValues>({
    ffa: 0,
    vmpfc: 0,
    ifg: 0,
    insula: 0,
  })
  const pulseOpacityRef = useRef(0)
  const boundsRef = useRef<RoiBounds | null>(null)
  const artifactIdRef = useRef<ParsedArtifact | null>(null)

  const [activeSegmentIndex, setActiveSegmentIndex] = useState(-1)
  const lastSegRef = useRef(-2)

  const effectiveArtifact = artifact ?? fallbackArtifact

  useEffect(() => {
    let id: number
    const tick = () => {
      const v = videoRef.current
      const art = effectiveArtifact
      if (!v || !art) {
        id = requestAnimationFrame(tick)
        return
      }
      if (artifactIdRef.current !== art) {
        artifactIdRef.current = art
        boundsRef.current = computeRoiBounds(art)
      }
      const hem = art.hemodynamicOffsetS
      const displayT = applyHemodynamicDelay
        ? Math.max(0, v.currentTime - hem)
        : v.currentTime
      const raw = interpolateRois(art, displayT)
      const b = boundsRef.current
      const norm = b ? normalizeRois(raw, b) : raw
      roiValuesRef.current = norm
      const m =
        Math.max(
          Math.abs(norm.ffa),
          Math.abs(norm.vmpfc),
          Math.abs(norm.ifg),
          Math.abs(norm.insula),
        ) * 0.8
      pulseOpacityRef.current = Math.min(1, Math.max(0, m))

      const seg = findActiveSegmentIndex(segments, v.currentTime)
      if (seg !== lastSegRef.current) {
        lastSegRef.current = seg
        setActiveSegmentIndex(seg)
      }

      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [videoRef, effectiveArtifact, segments, applyHemodynamicDelay])

  return { roiValuesRef, pulseOpacityRef, activeSegmentIndex }
}
