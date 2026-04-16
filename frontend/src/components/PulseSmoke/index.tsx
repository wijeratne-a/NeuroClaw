import { useEffect, useRef } from 'react'
import './pulse-smoke.css'

type Props = {
  pulseOpacityRef: React.MutableRefObject<number>
}

/**
 * Canvas smoke test: circle opacity driven by pulseOpacityRef at 60fps (no React state).
 */
export function PulseSmoke({ pulseOpacityRef }: Props) {
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
    let id: number
    const draw = () => {
      const w = canvas.width
      const h = canvas.height
      ctx.fillStyle = '#111'
      ctx.fillRect(0, 0, w, h)
      const o = pulseOpacityRef.current
      ctx.beginPath()
      ctx.arc(w / 2, h / 2, Math.min(w, h) * 0.25, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(100, 200, 255, ${0.15 + o * 0.85})`
      ctx.fill()
      id = requestAnimationFrame(draw)
    }
    id = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(id)
  }, [pulseOpacityRef])

  return (
    <div className="pulse-smoke">
      <span className="label">Sync smoke (canvas, rAF)</span>
      <canvas ref={canvasRef} width={200} height={200} className="pulse-canvas" />
    </div>
  )
}
