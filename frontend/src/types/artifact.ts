export interface TranscriptSegment {
  start: number
  end: number
  text: string
}

export interface ParsedArtifact {
  timestamps: Float64Array
  ffa: Float32Array
  vmpfc: Float32Array
  ifg: Float32Array
  insula: Float32Array
  hemodynamicOffsetS: number
  metadata: Record<string, unknown>
  transcriptSegments: TranscriptSegment[]
}

export interface RoiValues {
  ffa: number
  vmpfc: number
  ifg: number
  insula: number
}

export type TranscriptJson = {
  text?: string
  segments?: Array<{ start?: number; end?: number; text?: string }>
  asr_model?: string
  language?: string
  temperature?: number
}
