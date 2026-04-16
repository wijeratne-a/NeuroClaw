/**
 * Minimal safetensors file parser (header + tensor byte slices).
 * @see https://github.com/huggingface/safetensors
 */

export type TensorHeader = {
  dtype: string
  shape: number[]
  data_offsets: [number, number]
}

export type SafetensorsHeader = Record<string, TensorHeader | undefined> & {
  __metadata__?: Record<string, string>
}

export function parseSafetensorsHeader(buf: ArrayBuffer): SafetensorsHeader {
  const view = new DataView(buf)
  if (buf.byteLength < 8) {
    throw new Error('safetensors: buffer too small')
  }
  const headerLen = Number(view.getBigUint64(0, true))
  if (headerLen <= 0 || 8 + headerLen > buf.byteLength) {
    throw new Error('safetensors: invalid header length')
  }
  const jsonBytes = new Uint8Array(buf, 8, headerLen)
  const text = new TextDecoder('utf-8').decode(jsonBytes)
  return JSON.parse(text) as SafetensorsHeader
}

export function getTensorByteSlice(
  buf: ArrayBuffer,
  header: SafetensorsHeader,
  name: string,
): ArrayBuffer {
  const view = new DataView(buf)
  const headerLen = Number(view.getBigUint64(0, true))
  const info = header[name]
  if (!info || !('data_offsets' in info)) {
    throw new Error(`safetensors: missing tensor ${name}`)
  }
  const [a, b] = info.data_offsets
  const base = 8 + headerLen
  const start = base + a
  const end = base + b
  if (start < base || end > buf.byteLength || start > end) {
    throw new Error(`safetensors: bad offsets for ${name}`)
  }
  return buf.slice(start, end)
}
