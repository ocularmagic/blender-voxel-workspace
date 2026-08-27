"""MagicaVoxel .vox file parser.

Parses the standard MagicaVoxel binary format: a ``VOX `` header followed by
``PACK``/chunk structure. Supports the ``SIZE``/``XYZI`` (default) chunk
pair plus the optional ``RGBA`` palette chunk. The newer raw/XBR "packed"
variant (``PACK`` with more than one model, or version 200 raw voxel data)
is deliberately rejected with a readable error rather than half-supported.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_MAGIC = b"VOX "
_VERSION_150 = 150
_VERSION_200 = 200
# Chunks that may wrap children.
_MAIN = b"MAIN"
# Content chunks.
_SIZE = b"SIZE"
_XYZI = b"XYZI"
_RGBA = b"RGBA"
_PACK = b"PACK"
# MagicaVoxel's default palette (used when no RGBA chunk is present).
_DEFAULT_PALETTE_MAGICAVOXEL: Tuple[Tuple[int, int, int, int], ...] = (
    (255, 255, 255, 255),
    (255, 255, 204, 255),
    (204, 255, 255, 255),
    (204, 255, 204, 255),
    (255, 255, 153, 255),
    (255, 204, 255, 255),
    (255, 204, 204, 255),
    (255, 204, 153, 255),
    (255, 204, 102, 255),
    (255, 153, 204, 255),
    (204, 204, 255, 255),
    (204, 255, 153, 255),
    (255, 153, 153, 255),
    (255, 153, 102, 255),
    (255, 102, 102, 255),
    (255, 102, 51, 255),
)


class VoxParseError(ValueError):
    """Raised when a .vox file cannot be parsed."""


@dataclass
class VoxModel:
    """Parsed voxel model in .vox coordinates: x, y horizontal; z up."""

    size: Tuple[int, int, int]
    # {(x, y, z): 1-based palette color index}
    voxels: Dict[Tuple[int, int, int], int] = field(default_factory=dict)


@dataclass
class VoxDocument:
    version: int
    models: List[VoxModel]
    # palette[i] is the color for 1-based index i+1; 256 entries of RGBA 0..255.
    palette: List[Tuple[int, int, int, int]]

    def used_color_indices(self) -> List[int]:
        """Sorted unique 1-based color indices referenced by any model."""
        used = set()
        for model in self.models:
            used.update(model.voxels.values())
        return sorted(used)


def _read_chunk_header(data: bytes, offset: int) -> Tuple[bytes, int, int, int]:
    """Return (chunk_id, content_size, children_size, new_offset)."""
    if offset + 12 > len(data):
        raise VoxParseError("Truncated chunk header at byte %d" % offset)
    chunk_id = data[offset:offset + 4]
    content_size = int.from_bytes(data[offset + 4:offset + 8], "little")
    children_size = int.from_bytes(data[offset + 8:offset + 12], "little")
    return chunk_id, content_size, children_size, offset + 12


def _parse_size_content(content: bytes) -> Tuple[int, int, int]:
    if len(content) < 12:
        raise VoxParseError("SIZE chunk too short")
    return (
        int.from_bytes(content[0:4], "little"),
        int.from_bytes(content[4:8], "little"),
        int.from_bytes(content[8:12], "little"),
    )


def _parse_xyzi_content(content: bytes) -> Dict[Tuple[int, int, int], int]:
    if len(content) < 4:
        raise VoxParseError("XYZI chunk too short")
    count = int.from_bytes(content[0:4], "little")
    expected = 4 + 4 * count
    if len(content) < expected:
        raise VoxParseError("XYZI chunk truncated: %d voxels declared" % count)
    raw = np.frombuffer(content[4:expected], dtype=np.uint8).reshape(count, 4)
    voxels: Dict[Tuple[int, int, int], int] = {}
    for x, y, z, color_index in raw.tolist():
        if color_index == 0:
            # Index 0 means "empty" in MagicaVoxel; skip it.
            continue
        voxels[(x, y, z)] = int(color_index)
    return voxels


def _parse_rgba_content(content: bytes) -> List[Tuple[int, int, int, int]]:
    if len(content) < 1024:
        raise VoxParseError("RGBA chunk too short")
    raw = np.frombuffer(content[:1024], dtype=np.uint8).reshape(256, 4).tolist()
    palette = [(int(r), int(g), int(b), int(a)) for r, g, b, a in raw]
    return palette


def _walk_chunks(data: bytes, start: int, end: int) -> List[Tuple[bytes, bytes]]:
    """Iterate top-level chunks in the byte range [start, end)."""
    chunks: List[Tuple[bytes, bytes]] = []
    offset = start
    while offset < end:
        chunk_id, content_size, children_size, next_offset = _read_chunk_header(data, offset)
        content_end = next_offset + content_size
        if content_end + children_size > end:
            raise VoxParseError("Chunk %r overruns file bounds" % chunk_id.decode("latin-1"))
        content = data[next_offset:content_end]
        if chunk_id != _MAIN:
            chunks.append((chunk_id, content))
        else:
            chunks.extend(_walk_chunks(data, content_end, content_end + children_size))
        offset = content_end + children_size
    return chunks


def parse_vox_bytes(data: bytes) -> VoxDocument:
    """Parse raw .vox bytes into a VoxDocument."""
    if len(data) < 8 or data[:4] != _MAGIC:
        raise VoxParseError("Not a .vox file (missing VOX magic)")
    version = int.from_bytes(data[4:8], "little")
    if version == _VERSION_200:
        # Version 200 raw packing (XBR-style) is unsupported by design.
        raise VoxParseError(
            "Raw-voxel .vox files (version 200) are not supported; "
            "re-save the model with MagicaVoxel's standard format"
        )
    if version != _VERSION_150:
        raise VoxParseError("Unsupported .vox version %d" % version)

    chunks = _walk_chunks(data, 8, len(data))
    ids = {chunk_id for chunk_id, _ in chunks}
    pack_chunks = [content for chunk_id, content in chunks if chunk_id == _PACK]
    if pack_chunks:
        if len(pack_chunks[0]) >= 4 and int.from_bytes(pack_chunks[0][:4], "little") > 1:
            raise VoxParseError(
                "Multi-model (packed) .vox files are not supported; "
                "save each model as its own .vox file"
            )
    size_chunks = [content for chunk_id, content in chunks if chunk_id == _SIZE]
    xyzi_chunks = [content for chunk_id, content in chunks if chunk_id == _XYZI]
    if not size_chunks or not xyzi_chunks:
        raise VoxParseError("Missing SIZE/XYZI chunks; file is empty or corrupt")
    if len(size_chunks) != len(xyzi_chunks):
        raise VoxParseError("Mismatched SIZE/XYZI chunk counts")

    models: List[VoxModel] = []
    for size_content, xyzi_content in zip(size_chunks, xyzi_chunks):
        size = _parse_size_content(size_content)
        voxels = _parse_xyzi_content(xyzi_content)
        models.append(VoxModel(size=size, voxels=voxels))

    palette: List[Tuple[int, int, int, int]] = []
    for chunk_id, content in chunks:
        if chunk_id == _RGBA:
            palette = _parse_rgba_content(content)
            break
    if not palette:
        palette = list(_DEFAULT_PALETTE_MAGICAVOXEL) + [(255, 255, 255, 255)] * (256 - len(_DEFAULT_PALETTE_MAGICAVOXEL))

    if len(models) != 1:
        raise VoxParseError("Expected exactly one model, found %d" % len(models))
    return VoxDocument(version=version, models=models, palette=palette)


def parse_vox_file(path: str) -> VoxDocument:
    """Parse a .vox file from disk."""
    file_path = Path(path)
    if not file_path.is_file() or file_path.suffix.lower() != ".vox":
        raise VoxParseError("Choose an existing .vox file")
    return parse_vox_bytes(file_path.read_bytes())


def srgb_bytes_to_linear(color_bytes: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """Convert 0..255 sRGB RGBA to 0..1 linear RGBA (same regime as GLB import)."""
    out = []
    for channel in color_bytes[:3]:
        value = float(channel) / 255.0
        if value <= 0.04045:
            out.append(value / 12.92)
        else:
            out.append(((value + 0.055) / 1.055) ** 2.4)
    out.append(float(color_bytes[3]) / 255.0)
    return (out[0], out[1], out[2], out[3])
