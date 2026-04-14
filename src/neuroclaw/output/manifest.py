"""Reproducibility manifest: header SHA-256 + full .safetensors.zst SHA-256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import zstandard


@dataclass
class ReproManifest:
    header_sha256: str
    full_file_sha256: str
    artifact_path: str
    extra: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_safetensors_header_bytes(uncompressed_path: Path) -> bytes:
    """First JSON header blob in a safetensors file."""
    raw = uncompressed_path.read_bytes()
    if len(raw) < 8:
        msg = "invalid safetensors file"
        raise ValueError(msg)
    n = int.from_bytes(raw[:8], "little")
    return raw[8 : 8 + n]


def header_sha256_from_safetensors(uncompressed_path: Path) -> str:
    return _sha256_bytes(read_safetensors_header_bytes(uncompressed_path))


def decompress_zst_to_temp(zst_path: Path, tmp_path: Path) -> None:
    dctx = zstandard.ZstdDecompressor()
    tmp_path.write_bytes(dctx.decompress(zst_path.read_bytes()))


def build_manifest(zst_path: Path, tmp_uncompressed: Path | None = None) -> ReproManifest:
    """
    tmp_uncompressed: if provided, use it as uncompressed .safetensors for header hash.
    Otherwise decompress zst to a sibling temp file.
    """
    zst_path = Path(zst_path)
    full_hash = _sha256_bytes(zst_path.read_bytes())
    if tmp_uncompressed is None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tf:
            tmp = Path(tf.name)
        try:
            decompress_zst_to_temp(zst_path, tmp)
            hdr = header_sha256_from_safetensors(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        hdr = header_sha256_from_safetensors(tmp_uncompressed)

    return ReproManifest(
        header_sha256=hdr,
        full_file_sha256=full_hash,
        artifact_path=str(zst_path.resolve()),
        extra={},
    )


def write_manifest_sidecar(manifest: ReproManifest, clip_id: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{clip_id}_manifest.json"
    payload = asdict(manifest)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p
