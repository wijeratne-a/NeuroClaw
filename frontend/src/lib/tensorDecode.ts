import { Float16Array as Float16ArrayPolyfill } from '@petamoriken/float16'

/** Convert F16 tensor bytes to Float32Array (length N). */
export function f16BytesToFloat32(ab: ArrayBuffer): Float32Array {
  const n = ab.byteLength / 2
  const f16 = new Float16ArrayPolyfill(ab)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = f16[i] as number
  }
  return out
}

export function f64BytesToFloat64(ab: ArrayBuffer): Float64Array {
  if (ab.byteLength % 8 !== 0) {
    throw new Error('f64: byte length not multiple of 8')
  }
  return new Float64Array(ab)
}

export function u8BytesToUtf8Json(ab: ArrayBuffer): Record<string, unknown> {
  const u8 = new Uint8Array(ab)
  const s = new TextDecoder('utf-8').decode(u8)
  return JSON.parse(s) as Record<string, unknown>
}
