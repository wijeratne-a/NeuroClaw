import { useCallback, useRef } from 'react'
import './artifact-loader.css'

type Status = 'idle' | 'loading' | 'ready' | 'error'

type Props = {
  parseFiles: (zstFiles: File[], transcriptText: string) => void
  status: Status
  error: string | null
  progressPhase: string | null
}

export function ArtifactLoader({ parseFiles, status, error, progressPhase }: Props) {
  const zstInputRef = useRef<HTMLInputElement>(null)
  const trInputRef = useRef<HTMLInputElement>(null)

  const onPick = useCallback(() => {
    const zst = zstInputRef.current?.files
    if (!zst?.length) {
      return
    }
    const zstFiles = Array.from(zst).filter((f) => f.name.endsWith('.zst'))
    if (zstFiles.length === 0) {
      return
    }
    const trFile = trInputRef.current?.files?.[0]
    void (async () => {
      const transcriptText = trFile
        ? await trFile.text()
        : '{"segments":[],"text":""}'
      parseFiles(zstFiles, transcriptText)
    })()
  }, [parseFiles])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const files = Array.from(e.dataTransfer.files)
      const zstFiles = files.filter(
        (f) => f.name.endsWith('.safetensors.zst') || f.name.endsWith('.zst'),
      )
      const tr = files.find(
        (f) => f.name.endsWith('_transcript.json') || f.name.endsWith('transcript.json'),
      )
      if (zstFiles.length === 0) {
        return
      }
      void (async () => {
        const transcriptText = tr
          ? await tr.text()
          : '{"segments":[],"text":""}'
        parseFiles(zstFiles, transcriptText)
      })()
    },
    [parseFiles],
  )

  return (
    <section className="artifact-loader" onDragOver={(e) => e.preventDefault()}>
      <h2>Artifacts</h2>
      <p className="hint">
        Drop <code>*.safetensors.zst</code> (one or more chunks) and optional{' '}
        <code>*_transcript.json</code>, or use file inputs.
      </p>
      <div
        className="dropzone"
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault()
        }}
      >
        Drop NeuroClaw outputs here
      </div>
      <div className="row">
        <label>
          .safetensors.zst (multi-select)
          <input ref={zstInputRef} type="file" accept=".zst,.safetensors.zst" multiple />
        </label>
        <label>
          transcript.json (optional)
          <input ref={trInputRef} type="file" accept=".json,application/json" />
        </label>
        <button type="button" onClick={onPick}>
          Parse selected files
        </button>
      </div>
      <div className="status">
        Status: <strong>{status}</strong>
        {progressPhase ? ` (${progressPhase})` : null}
      </div>
      {error ? <div className="err">{error}</div> : null}
    </section>
  )
}
