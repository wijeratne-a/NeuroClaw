import { useEffect, useRef } from 'react'
import type { ParsedArtifact } from '../../types/artifact'
import { floorBinIndex, interpolateRois } from '../../lib/syncMath'

type RoiKey = 'ffa' | 'vmpfc' | 'ifg' | 'insula'

const LABELS: Record<RoiKey, string> = {
  ffa: 'FFA',
  vmpfc: 'vmPFC',
  ifg: 'IFG',
  insula: 'Insula',
}

const COLORS: Record<RoiKey, string> = {
  ffa: '#3b82f6',
  vmpfc: '#4ade80',
  ifg: '#a855f7',
  insula: '#f59e0b',
}

function seriesFor(artifact: ParsedArtifact, k: RoiKey): Float32Array {
  switch (k) {
    case 'ffa':
      return artifact.ffa
    case 'vmpfc':
      return artifact.vmpfc
    case 'ifg':
      return artifact.ifg
    case 'insula':
      return artifact.insula
    default:
      return artifact.ffa
  }
}

export function Sparkline({
  artifact,
  videoRef,
  roiKey,
  applyHemodynamicDelay,
}: {
  artifact: ParsedArtifact
  videoRef: React.RefObject<HTMLVideoElement | null>
  roiKey: RoiKey
  applyHemodynamicDelay: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const ts = artifact.timestamps
    const series = seriesFor(artifact, roiKey)
    const windowS = 30
    const targetFrameMs = 50 // ~20fps per chart to reduce UI/GPU load

    let id = 0
    let lastDrawMs = 0

    const draw = (nowMs: number) => {
      if (nowMs - lastDrawMs < targetFrameMs) {
        id = requestAnimationFrame(draw)
        return
      }
      lastDrawMs = nowMs

      const v = videoRef.current
      if (!v) {
        id = requestAnimationFrame(draw)
        return
      }

      const hem = artifact.hemodynamicOffsetS
      const t = applyHemodynamicDelay ? Math.max(0, v.currentTime - hem) : v.currentTime
      const t0 = Math.max(0, t - windowS)

      const start = floorBinIndex(ts, t0)
      const end = floorBinIndex(ts, t)

      let ymin = Infinity
      let ymax = -Infinity
      for (let i = start; i <= end && i < ts.length; i++) {
        const y = series[i]
        ymin = Math.min(ymin, y)
        ymax = Math.max(ymax, y)
      }
      const cur = interpolateRois(artifact, t)
      const yCur =
        roiKey === 'ffa' ? cur.ffa : roiKey === 'vmpfc' ? cur.vmpfc : roiKey === 'ifg' ? cur.ifg : cur.insula
      ymin = Math.min(ymin, yCur)
      ymax = Math.max(ymax, yCur)

      if (!Number.isFinite(ymin) || !Number.isFinite(ymax) || ymin === ymax) {
        ymin -= 1
        ymax += 1
      }
      const pad = (ymax - ymin) * 0.08

      const w = canvas.width
      const h = canvas.height
      ctx.fillStyle = '#121214'
      ctx.fillRect(0, 0, w, h)
      ctx.strokeStyle = '#333'
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1)

      const mapX = (tx: number) => ((tx - t0) / (windowS || 1)) * (w - 8) + 4
      const mapY = (ty: number) => h - 4 - ((ty - (ymin - pad)) / (ymax - ymin + 2 * pad)) * (h - 8)

      ctx.beginPath()
      ctx.strokeStyle = COLORS[roiKey]
      ctx.lineWidth = 1.5
      let started = false
      for (let i = start; i <= end && i < ts.length; i++) {
        const px = mapX(ts[i])
        const py = mapY(series[i])
        if (!started) {
          ctx.moveTo(px, py)
          started = true
        } else {
          ctx.lineTo(px, py)
        }
      }
      if (started) {
        const pxCur = mapX(t)
        const pyCur = mapY(yCur)
        ctx.lineTo(pxCur, pyCur)
      }
      ctx.stroke()

      const pxPlay = mapX(t)
      ctx.strokeStyle = '#fff'
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(pxPlay, 0)
      ctx.lineTo(pxPlay, h)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#888'
      ctx.font = '10px system-ui'
      ctx.fillText(LABELS[roiKey], 6, 14)

      id = requestAnimationFrame(draw)
    }

    id = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(id)
  }, [artifact, videoRef, roiKey, applyHemodynamicDelay])

  return <canvas ref={canvasRef} width={320} height={72} className="sparkline-canvas" />
}
