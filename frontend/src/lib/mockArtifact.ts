import type { ParsedArtifact } from '../types/artifact'

export function makeMockArtifact(seconds: number): ParsedArtifact {
  const n = Math.max(2, Math.ceil(seconds))
  const timestamps = new Float64Array(n)
  const ffa = new Float32Array(n)
  const vmpfc = new Float32Array(n)
  const ifg = new Float32Array(n)
  const insula = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    timestamps[i] = i
    const phase = i * 0.3
    ffa[i] = Math.sin(phase) * 0.5
    vmpfc[i] = Math.cos(phase * 0.7) * 0.5
    ifg[i] = Math.sin(phase * 1.1) * 0.4
    insula[i] = Math.cos(phase * 0.9) * 0.4
  }
  return {
    timestamps,
    ffa,
    vmpfc,
    ifg,
    insula,
    hemodynamicOffsetS: 5,
    metadata: {},
    transcriptSegments: [],
  }
}
