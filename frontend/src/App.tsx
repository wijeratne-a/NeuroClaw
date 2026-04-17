import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import './components/BrainCanvas/brain-canvas.css'
import { ArtifactLoader } from './components/ArtifactLoader'

const BrainCanvas = lazy(async () => {
  const mod = await import('./components/BrainCanvas')
  return { default: mod.BrainCanvas }
})
import { SemanticHud } from './components/SemanticHud'
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
  const skipAutoDemoRef = useRef(false)

  const devMode = useMemo(
    () => typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('dev') === '1',
    [],
  )

  const demoArtifact = useMemo(() => makeMockArtifact(120), [])
  const { artifact, status, error, progressPhase, parseFiles: parseArtifact } = useArtifact()

  const parseFiles = useCallback(
    (zstFiles: File[], transcriptText: string) => {
      skipAutoDemoRef.current = true
      parseArtifact(zstFiles, transcriptText)
    },
    [parseArtifact],
  )

  /** Defer demo downloads so first paint + lazy WebGL chunk aren’t blocked on large binaries. */
  useEffect(() => {
    const id = window.setTimeout(() => {
      void (async () => {
        const zst = await fetch('/demo/demo_voxels.safetensors.zst')
        const tr = await fetch('/demo/demo_transcript.json')
        const vid = await fetch('/demo/demo_clip.mp4')
        if (!zst.ok || !tr.ok || !vid.ok) {
          return
        }
        if (skipAutoDemoRef.current) {
          return
        }
        const zstFile = new File([await zst.blob()], 'demo_voxels.safetensors.zst', {
          type: 'application/octet-stream',
        })
        const transcriptText = await tr.text()
        if (skipAutoDemoRef.current) {
          return
        }
        parseArtifact([zstFile], transcriptText)
        if (skipAutoDemoRef.current) {
          return
        }
        setVideoUrl(URL.createObjectURL(await vid.blob()))
      })()
    }, 250)
    return () => window.clearTimeout(id)
  }, [parseArtifact])

  const fallbackArtifact = useDemoMetrics ? demoArtifact : null
  const segments = artifact?.transcriptSegments ?? []

  const { roiValuesRef, activeSegmentIndex } = useVideoSync(
    videoRef,
    artifact,
    fallbackArtifact,
    segments,
    { applyHemodynamicDelay },
  )

  return (
    <>
      <header className="app-header">
        <h1>NeuroClaw — Cortical Marketing Intelligence</h1>
        <p className="sub">
          Four Destrieux ROIs — FFA, vmPFC, IFG, Insula — driven from cortical-four{' '}
          <code>.safetensors.zst</code>. Default timeline is <strong>stimulus-aligned</strong> with the
          video; enable hemodynamic delay to match HRF-shifted lookup.
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
              skipAutoDemoRef.current = true
              const f = e.target.files?.[0]
              if (videoUrl) {
                URL.revokeObjectURL(videoUrl)
              }
              setVideoUrl(f ? URL.createObjectURL(f) : null)
            }}
          />
        </label>
        {devMode ? (
          <label className="demo-toggle">
            <input
              type="checkbox"
              checked={useDemoMetrics}
              onChange={(e) => setUseDemoMetrics(e.target.checked)}
            />{' '}
            Dev: mock 1Hz metrics (<code>?dev=1</code>)
          </label>
        ) : null}
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
            <Suspense fallback={<div className="brain-canvas-fallback" aria-hidden />}>
              <BrainCanvas roiValuesRef={roiValuesRef} />
            </Suspense>
            <SemanticHud roiValuesRef={roiValuesRef} />
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
