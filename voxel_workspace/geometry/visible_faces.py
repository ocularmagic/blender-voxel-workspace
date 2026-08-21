from typing import Collection, Optional, Tuple
import numpy as np

from .buffers import MeshBuffers
from .material_filter import filter_brick_and_apron

FACE_SPECS: list[Tuple[int, int, np.ndarray]] = [
    # (axis, sgn, quad_template_ccw_outward)
    (0, 1, np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.float32)),
    (0, -1, np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.float32)),
    (1, 1, np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float32)),
    (1, -1, np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32)),
    (2, 1, np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)),
    (2, -1, np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)),
]


def mesh_visible_faces(
    apron: np.ndarray,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
    brick: Optional[np.ndarray] = None,
    exclude_indices: Optional[Collection[int]] = None,
    only_index: Optional[int] = None,
) -> MeshBuffers:
    """Generate mesh buffers from a brick's 1-voxel padded apron using vectorized visible-face extraction.

    Args:
        apron: (sx+2, sy+2, sz+2) uint8 array containing the brick and 1-voxel neighbor padding.
        origin: World/object space offset of the brick's (0, 0, 0) local coordinate.
        voxel_size: Edge length of a voxel in 3D units.
        brick: Optional owning brick array. If omitted, apron[1:-1, 1:-1, 1:-1] is used.
        exclude_indices: Optional indices to exclude from meshing (e.g. VOLUME domain).
        only_index: Optional index to isolate (e.g. single VOLUME proxy domain).

    Returns:
        MeshBuffers with float32 positions, int32 triangle indices, int32 corner palette indices, and quad_count.
    """
    raw_core = brick if brick is not None else apron[1:-1, 1:-1, 1:-1]
    core, apron = filter_brick_and_apron(
        raw_core,
        apron,
        exclude_indices=exclude_indices,
        only_index=only_index,
    )

    if not np.any(core):
        return MeshBuffers(
            positions=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0, 3), dtype=np.int32),
            palette_indices=np.empty((0,), dtype=np.int32),
            quad_count=0,
        )

    # Apron slices for neighbor lookups
    neighbor_slices = {
        (0, 1): (slice(2, None), slice(1, -1), slice(1, -1)),
        (0, -1): (slice(0, -2), slice(1, -1), slice(1, -1)),
        (1, 1): (slice(1, -1), slice(2, None), slice(1, -1)),
        (1, -1): (slice(1, -1), slice(0, -2), slice(1, -1)),
        (2, 1): (slice(1, -1), slice(1, -1), slice(2, None)),
        (2, -1): (slice(1, -1), slice(1, -1), slice(0, -2)),
    }

    vs: list[np.ndarray] = []
    ms: list[np.ndarray] = []

    origin_arr = np.asarray(origin, dtype=np.float32)

    for axis, sgn, tpl in FACE_SPECS:
        neigh = apron[neighbor_slices[(axis, sgn)]]
        mask = (core != 0) & (neigh == 0)
        coords = np.argwhere(mask).astype(np.float32)
        if len(coords) > 0:
            face_verts = (coords[:, None, :] + tpl[None, :, :]) * voxel_size + origin_arr
            vs.append(face_verts.reshape(-1, 3))
            ms.append(core[mask])

    if not vs:
        return MeshBuffers(
            positions=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0, 3), dtype=np.int32),
            palette_indices=np.empty((0,), dtype=np.int32),
            quad_count=0,
        )

    positions = np.concatenate(vs, axis=0)
    quad_mats = np.concatenate(ms, axis=0)
    quad_count = len(quad_mats)

    palette_indices = np.repeat(quad_mats.astype(np.int32), 4)

    quad_bases = np.arange(0, quad_count * 4, 4, dtype=np.int32)[:, None]
    t1 = quad_bases + np.array([0, 1, 2], dtype=np.int32)
    t2 = quad_bases + np.array([0, 2, 3], dtype=np.int32)

    indices = np.empty((quad_count * 2, 3), dtype=np.int32)
    indices[0::2] = t1
    indices[1::2] = t2

    return MeshBuffers(
        positions=positions,
        indices=indices,
        palette_indices=palette_indices,
        quad_count=quad_count,
    )
