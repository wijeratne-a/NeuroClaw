import { useMemo, useRef, useState } from 'react'
import './App.css'
import { ArtifactLoader } from './components/ArtifactLoader'
import { BrainCanvas } from './components/BrainCanvas'
import { PulseSmoke } from './components/PulseSmoke'
import { SparklinePanel } from './components/SparklinePanel'
import { TranscriptPanel } from './components/TranscriptPanel'
import { VideoPlayer } from './components/VideoPlayer'
import { useArtifact } from './hooks/useArtifact'
import { useVideoSync } from './hooks/useVideoSync'
import { makeMockArtifact } from './lib/mockArtifact'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [useDemoMetrics, setUseDemoMetrics] = useState(false)
  const [applyHemodynamicDelay, setApplyHemodynamicDelay] = useState(false)

  const demoArtifact = useMemo(() => makeMockArtifact(120), [])
  const { artifact, status, error, progressPhase, parseFiles } = useArtifact()

  const fallbackArtifact = useDemoMetrics ? demoArtifact : null
  const segments = artifact?.transcriptSegments ?? []

  const { roiValuesRef, pulseOpacityRef, activeSegmentIndex } = useVideoSync(
    videoRef,
    artifact,
    fallbackArtifact,
    segments,
    { applyHemodynamicDelay },
  )

  return (
    <>
      <header className="app-header">
        <h1>NeuroClaw — Marketing Four</h1>
        <p className="sub">
          Upload cortical-four <code>.safetensors.zst</code>, transcript, and a video. By default the brain
          tracks <strong>video time</strong> (stimulus-aligned). Enable hemodynamic delay to use{' '}
          <code>currentTime − hemodynamic_offset_s</code> from the artifact metadata.
        </p>
      </header>

      <ArtifactLoader
        parseFiles={parseFiles}
        status={status}
        error={error}
        progressPhase={progressPhase}
      />

      <section className="video-row">
        <label className="video-file">
          Video (.mp4 / .mkv)
          <input
            type="file"
            accept="video/*"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (videoUrl) {
                URL.revokeObjectURL(videoUrl)
              }
              setVideoUrl(f ? URL.createObjectURL(f) : null)
            }}
          />
        </label>
        <label className="demo-toggle">
          <input
            type="checkbox"
            checked={useDemoMetrics}
            onChange={(e) => setUseDemoMetrics(e.target.checked)}
          />{' '}
          Demo 1Hz mock metrics (no artifact required for pulse)
        </label>
        <label className="demo-toggle">
          <input
            type="checkbox"
            checked={applyHemodynamicDelay}
            onChange={(e) => setApplyHemodynamicDelay(e.target.checked)}
          />{' '}
          Apply hemodynamic delay (HRF) to neural timeline
        </label>
      </section>

      <div className="layout">
        <div className="col main">
          <VideoPlayer ref={videoRef} videoUrl={videoUrl} />
          <div className="brain-row">
            <BrainCanvas roiValuesRef={roiValuesRef} />
            <PulseSmoke pulseOpacityRef={pulseOpacityRef} />
          </div>
          <SparklinePanel
            artifact={artifact ?? fallbackArtifact}
            videoRef={videoRef}
            applyHemodynamicDelay={applyHemodynamicDelay}
          />
        </div>
        <div className="col side">
          <TranscriptPanel
            segments={segments}
            activeIndex={activeSegmentIndex}
            videoRef={videoRef}
          />
        </div>
      </div>
    </>
  )
}

export default App
