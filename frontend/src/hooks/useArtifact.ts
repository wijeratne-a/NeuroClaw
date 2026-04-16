import { useCallback, useRef, useState } from 'react'
import type { ParsedArtifact } from '../types/artifact'

type Status = 'idle' | 'loading' | 'ready' | 'error'

export function useArtifact() {
  const [artifact, setArtifact] = useState<ParsedArtifact | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [progressPhase, setProgressPhase] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)

  const parseFiles = useCallback((zstFiles: File[], transcriptText: string) => {
    setStatus('loading')
    setError(null)
    setProgressPhase('start')
    const w = new Worker(new URL('../workers/artifact.worker.ts', import.meta.url), {
      type: 'module',
    })
    workerRef.current = w

    const onMsg = (ev: MessageEvent) => {
      const d = ev.data as
        | { type: 'PROGRESS'; phase: string }
        | { type: 'DONE'; result: ParsedArtifact }
        | { type: 'ERROR'; message: string }
      if (d.type === 'PROGRESS') {
        setProgressPhase(d.phase)
        return
      }
      if (d.type === 'ERROR') {
        setError(d.message)
        setStatus('error')
        w.terminate()
        workerRef.current = null
        return
      }
      if (d.type === 'DONE') {
        setArtifact(d.result)
        setStatus('ready')
        setProgressPhase(null)
        w.terminate()
        workerRef.current = null
        console.log('Parsed artifact', {
          T: d.result.timestamps.length,
          FFA: d.result.ffa,
          vmPFC: d.result.vmpfc,
          timestamps: d.result.timestamps,
        })
      }
    }
    w.addEventListener('message', onMsg)
    w.addEventListener('error', (e) => {
      setError(e.message || 'worker error')
      setStatus('error')
    })

    void (async () => {
      const buffers = await Promise.all(zstFiles.map((f) => f.arrayBuffer()))
      w.postMessage({
        type: 'PARSE',
        zstBuffers: buffers,
        transcriptText,
      })
    })()
  }, [])

  const reset = useCallback(() => {
    workerRef.current?.terminate()
    workerRef.current = null
    setArtifact(null)
    setStatus('idle')
    setError(null)
    setProgressPhase(null)
  }, [])

  return { artifact, status, error, progressPhase, parseFiles, reset }
}
