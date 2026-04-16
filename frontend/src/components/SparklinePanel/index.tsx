import type { ParsedArtifact } from '../../types/artifact'
import { Sparkline } from './Sparkline'
import './sparkline-panel.css'

export function SparklinePanel({
  artifact,
  videoRef,
  applyHemodynamicDelay,
}: {
  artifact: ParsedArtifact | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  applyHemodynamicDelay: boolean
}) {
  if (!artifact) {
    return <div className="sparkline-panel empty">Load an artifact to see ROI sparklines.</div>
  }

  return (
    <div className="sparkline-panel">
      <h3 className="sparkline-title">
        ROI series —{' '}
        <strong>{applyHemodynamicDelay ? 'HRF-delayed' : 'Stimulus-aligned'}</strong> (30s window)
      </h3>
      <div className="grid">
        <Sparkline
          artifact={artifact}
          videoRef={videoRef}
          roiKey="ffa"
          applyHemodynamicDelay={applyHemodynamicDelay}
        />
        <Sparkline
          artifact={artifact}
          videoRef={videoRef}
          roiKey="vmpfc"
          applyHemodynamicDelay={applyHemodynamicDelay}
        />
        <Sparkline
          artifact={artifact}
          videoRef={videoRef}
          roiKey="ifg"
          applyHemodynamicDelay={applyHemodynamicDelay}
        />
        <Sparkline
          artifact={artifact}
          videoRef={videoRef}
          roiKey="insula"
          applyHemodynamicDelay={applyHemodynamicDelay}
        />
      </div>
    </div>
  )
}
