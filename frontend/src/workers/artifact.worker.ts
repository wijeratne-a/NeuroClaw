/// <reference lib="webworker" />

import { decompress } from 'fzstd'
import { parseSafetensorsHeader, getTensorByteSlice } from '../lib/safetensors'
import { f16BytesToFloat32, f64BytesToFloat64, u8BytesToUtf8Json } from '../lib/tensorDecode'
import { parseTranscriptJson } from '../lib/transcript'
import type { ParsedArtifact } from '../types/artifact'

type WorkerIn = {
  type: 'PARSE'
  zstBuffers: ArrayBuffer[]
  transcriptText: string
}

type WorkerOutProgress = { type: 'PROGRESS'; phase: 'decompress' | 'parse' | 'convert' }
type WorkerOutDone = { type: 'DONE'; result: ParsedArtifact }
type WorkerOutErr = { type: 'ERROR'; message: string }

type WorkerOut = WorkerOutProgress | WorkerOutDone | WorkerOutErr

interface ChunkParsed {
  ffa: Float32Array
  vmpfc: Float32Array
  ifg: Float32Array
  insula: Float32Array
  timestamps: Float64Array
  metadata: Record<string, unknown>
}

function toArrayBuffer(u8: Uint8Array): ArrayBuffer {
  const out = new ArrayBuffer(u8.byteLength)
  new Uint8Array(out).set(u8)
  return out
}

function parseSafetensorsFromBuffer(buf: ArrayBuffer): ChunkParsed {
  const head = parseSafetensorsHeader(buf)
  const ffa = f16BytesToFloat32(getTensorByteSlice(buf, head, 'FFA'))
  const vmpfc = f16BytesToFloat32(getTensorByteSlice(buf, head, 'vmPFC'))
  const ifg = f16BytesToFloat32(getTensorByteSlice(buf, head, 'IFG'))
  const insula = f16BytesToFloat32(getTensorByteSlice(buf, head, 'Insula'))
  const timestamps = f64BytesToFloat64(getTensorByteSlice(buf, head, 'timestamps'))
  const metaRaw = getTensorByteSlice(buf, head, 'model_metadata')
  const metadata = u8BytesToUtf8Json(metaRaw)

  const n = timestamps.length
  if (
    ffa.length !== n ||
    vmpfc.length !== n ||
    ifg.length !== n ||
    insula.length !== n
  ) {
    throw new Error(
      `tensor length mismatch: T=${n} ffa=${ffa.length} vmPFC=${vmpfc.length} IFG=${ifg.length} Insula=${insula.length}`,
    )
  }

  return { ffa, vmpfc, ifg, insula, timestamps, metadata }
}

function mergeChunks(parts: ChunkParsed[]): Omit<ParsedArtifact, 'transcriptSegments'> {
  if (parts.length === 0) {
    throw new Error('no chunks to merge')
  }
  const sorted = [...parts].sort((a, b) => {
    const ca = Number(a.metadata.chunk_index ?? 0)
    const cb = Number(b.metadata.chunk_index ?? 0)
    if (ca !== cb) {
      return ca - cb
    }
    const ta = Number(a.metadata.chunk_t0_s ?? 0)
    const tb = Number(b.metadata.chunk_t0_s ?? 0)
    return ta - tb
  })

  const cat32 = (arrays: Float32Array[]) => {
    const len = arrays.reduce((s, a) => s + a.length, 0)
    const out = new Float32Array(len)
    let o = 0
    for (const a of arrays) {
      out.set(a, o)
      o += a.length
    }
    return out
  }
  const cat64 = (arrays: Float64Array[]) => {
    const len = arrays.reduce((s, a) => s + a.length, 0)
    const out = new Float64Array(len)
    let o = 0
    for (const a of arrays) {
      out.set(a, o)
      o += a.length
    }
    return out
  }

  const ffa = cat32(sorted.map((s) => s.ffa))
  const vmpfc = cat32(sorted.map((s) => s.vmpfc))
  const ifg = cat32(sorted.map((s) => s.ifg))
  const insula = cat32(sorted.map((s) => s.insula))
  const timestamps = cat64(sorted.map((s) => s.timestamps))

  const metadata = { ...sorted[0].metadata }
  const hemodynamicOffsetS = Number(metadata.hemodynamic_offset_s ?? 5)

  return {
    timestamps,
    ffa,
    vmpfc,
    ifg,
    insula,
    hemodynamicOffsetS,
    metadata,
  }
}

function post(o: WorkerOut) {
  self.postMessage(o)
}

self.onmessage = (ev: MessageEvent<WorkerIn>) => {
  const msg = ev.data
  if (msg.type !== 'PARSE') {
    return
  }
  try {
    post({ type: 'PROGRESS', phase: 'decompress' })
    const chunks: ChunkParsed[] = []
    for (const ab of msg.zstBuffers) {
      const u8 = new Uint8Array(ab)
      const dec = decompress(u8)
      const buf = toArrayBuffer(dec)
      post({ type: 'PROGRESS', phase: 'parse' })
      chunks.push(parseSafetensorsFromBuffer(buf))
      post({ type: 'PROGRESS', phase: 'convert' })
    }
    const merged = mergeChunks(chunks)
    const transcriptSegments = parseTranscriptJson(msg.transcriptText)
    const result: ParsedArtifact = {
      ...merged,
      transcriptSegments,
    }
    post({ type: 'DONE', result })
  } catch (e) {
    post({
      type: 'ERROR',
      message: e instanceof Error ? e.message : String(e),
    })
  }
}
