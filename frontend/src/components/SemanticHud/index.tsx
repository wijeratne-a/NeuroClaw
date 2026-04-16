import { useEffect, useRef, useState } from 'react'
import type { MutableRefObject } from 'react'
import type { RoiValues } from '../../types/artifact'
import './semantic-hud.css'

const THRESHOLDS: Array<{
  roiKey: keyof RoiValues
  level: number
  label: string
  color: string
}> = [
  { roiKey: 'ffa', level: 0.72, label: 'High facial engagement detected', color: '#66ccff' },
  { roiKey: 'vmpfc', level: 0.7, label: 'High subjective value / desire signal', color: '#ff88cc' },
  { roiKey: 'ifg', level: 0.68, label: 'Language / brand processing active', color: '#ffcc66' },
  { roiKey: 'insula', level: 0.65, label: 'Heightened arousal / salience', color: '#99ff66' },
]

type Toast = { id: number; label: string; color: string }

export function SemanticHud({
  roiValuesRef,
}: {
  roiValuesRef: MutableRefObject<RoiValues>
}) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const idRef = useRef(0)
  const activeRef = useRef<Record<string, boolean>>({})
  const timersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  useEffect(() => {
    const timers = timersRef.current
    const tick = () => {
      const v = roiValuesRef.current
      for (const t of THRESHOLDS) {
        const key = t.roiKey
        const on = v[key] >= t.level
        const was = activeRef.current[key]
        if (on && !was) {
          activeRef.current[key] = true
          const id = ++idRef.current
          setToasts((prev) => [...prev, { id, label: t.label, color: t.color }])
          const tid = setTimeout(() => {
            setToasts((prev) => prev.filter((x) => x.id !== id))
            activeRef.current[key] = false
            delete timers[key]
          }, 3000)
          timers[key] = tid
        } else if (!on && was) {
          activeRef.current[key] = false
        }
      }
    }
    const id = window.setInterval(tick, 250)
    return () => {
      window.clearInterval(id)
      for (const k of Object.keys(timers)) {
        clearTimeout(timers[k])
      }
    }
  }, [roiValuesRef])

  return (
    <aside className="semantic-hud" aria-live="polite">
      {toasts.map((t) => (
        <p
          key={t.id}
          className="toast"
          style={{ background: `linear-gradient(135deg, ${t.color}33, #1a1a22)` }}
        >
          {t.label}
        </p>
      ))}
    </aside>
  )
}
