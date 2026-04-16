import './transcript-panel.css'
import type { TranscriptSegment } from '../../types/artifact'

type Props = {
  segments: TranscriptSegment[]
  activeIndex: number
  videoRef: React.RefObject<HTMLVideoElement | null>
}

export function TranscriptPanel({ segments, activeIndex, videoRef }: Props) {
  return (
    <div className="transcript-panel">
      <h3>Transcript</h3>
      {segments.length === 0 ? (
        <p className="empty">No segments (silent video or empty ASR).</p>
      ) : (
        <ul className="segments">
          {segments.map((s, i) => (
            <li key={`${s.start}-${s.end}-${i}`}>
              <button
                type="button"
                className={i === activeIndex ? 'seg active' : 'seg'}
                onClick={() => {
                  const v = videoRef.current
                  if (v) {
                    v.currentTime = s.start
                  }
                }}
              >
                <span className="time">
                  {s.start.toFixed(2)}s – {s.end.toFixed(2)}s
                </span>
                <span className="text">{s.text}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
