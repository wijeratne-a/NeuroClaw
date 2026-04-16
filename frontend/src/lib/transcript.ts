import type { TranscriptJson, TranscriptSegment } from '../types/artifact'

export function parseTranscriptJson(text: string): TranscriptSegment[] {
  if (!text.trim()) {
    return []
  }
  const j = JSON.parse(text) as TranscriptJson
  const segs = j.segments ?? []
  return segs
    .filter((s) => s.start != null && s.end != null)
    .map((s) => ({
      start: Number(s.start),
      end: Number(s.end),
      text: String(s.text ?? ''),
    }))
}
