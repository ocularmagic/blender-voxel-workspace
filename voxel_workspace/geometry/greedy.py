from typing import Iterator, Optional, Tuple
import numpy as np

from voxel_workspace.geometry.buffers import MeshBuffers


def _greedy_rectangles(mask: np.ndarray) -> Iterator[Tuple[int, int, int, int, int]]:
    """Find maximal rectangles of uniform non-zero material in a 2D slice.
    
    Yields:
        (u, v, width, height, material)
    """
    nu, nv = mask.shape
    used = np.zeros((nu, nv), dtype=np.bool_)
    for u in range(nu):
        for v in range(nv):
            m = int(mask[u, v])
            if m == 0 or used[u, v]:
                continue
            w = 1
            while u + w < nu and not used[u + w, v] and int(mask[u + w, v]) == m:
                w += 1
            h = 1
            while v + h < nv:
                row = mask[u : u + w, v + h]
                if np.any(used[u : u + w, v + h]) or not np.all(row == m):
                    break
                h += 1
            used[u : u + w, v : v + h] = True
            yield u, v, w, h, m


def mesh_greedy(
    apron: np.ndarray,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 1.0,
    brick: Optional[np.ndarray] = None,
) -> MeshBuffers:
    """Generate mesh buffers by merging coplanar faces of the same material into maximal rectangles.

    Args:
        apron: (sx+2, sy+2, sz+2) uint8 array containing the brick and 1-voxel neighbor padding.
        origin: World/object space offset of the brick's (0, 0, 0) local coordinate.
        voxel_size: Edge length of a voxel in 3D units.
        brick: Optional owning brick array. If omitted, apron[1:-1, 1:-1, 1:-1] is used.

    Returns:
        MeshBuffers with float32 positions, int32 triangle indices, int32 corner palette indices, and quad_count.
    """
    core = brick if brick is not None else apron[1:-1, 1:-1, 1:-1]
    if not np.any(core):
        return MeshBuffers(
            positions=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0, 3), dtype=np.int32),
            palette_indices=np.empty((0,), dtype=np.int32),
            quad_count=0,
        )

    dims = core.shape
    origin_arr = np.asarray(origin, dtype=np.float32)

    neighbor_slices = {
        (0, 1): (slice(2, None), slice(1, -1), slice(1, -1)),
        (0, -1): (slice(0, -2), slice(1, -1), slice(1, -1)),
        (1, 1): (slice(1, -1), slice(2, None), slice(1, -1)),
        (1, -1): (slice(1, -1), slice(0, -2), slice(1, -1)),
        (2, 1): (slice(1, -1), slice(1, -1), slice(2, None)),
        (2, -1): (slice(1, -1), slice(1, -1), slice(0, -2)),
    }

    verts: list[np.ndarray] = []
    mats: list[int] = []

    for d in range(3):
        u = (d + 1) % 3
        v = (d + 2) % 3

        plus = apron[neighbor_slices[(d, 1)]]
        minus = apron[neighbor_slices[(d, -1)]]

        for positive, exposed in ((True, (core != 0) & (plus == 0)), (False, (core != 0) & (minus == 0))):
            values = np.where(exposed, core, 0)
            for i in range(dims[d]):
                # Extract 2D slice with axes (u, v)
                if d == 0:
                    mask = values[i, :, :]  # (y, z) -> (u, v)
                elif d == 1:
                    mask = values[:, i, :].T  # (x, z).T -> (z, x) -> (u, v)
                else:
                    mask = values[:, :, i]  # (x, y) -> (u, v)

                for a, b, w, h, m in _greedy_rectangles(mask):
                    p0 = np.zeros(3, dtype=np.float32)
                    p1 = np.zeros(3, dtype=np.float32)
                    p2 = np.zeros(3, dtype=np.float32)
                    p3 = np.zeros(3, dtype=np.float32)

                    p0[d] = i + (1.0 if positive else 0.0)
                    p0[u] = float(a)
                    p0[v] = float(b)

                    p1[:] = p0[:]
                    p1[u] += float(w)

                    p2[:] = p0[:]
                    p2[u] += float(w)
                    p2[v] += float(h)

                    p3[:] = p0[:]
                    p3[v] += float(h)

                    if positive:
                        q_verts = np.stack([p0, p1, p2, p3], axis=0)
                    else:
                        q_verts = np.stack([p0, p3, p2, p1], axis=0)

                    q_verts = q_verts * voxel_size + origin_arr
                    verts.append(q_verts)
                    mats.append(m)

    if not verts:
        return MeshBuffers(
            positions=np.empty((0, 3), dtype=np.float32),
            indices=np.empty((0, 3), dtype=np.int32),
            palette_indices=np.empty((0,), dtype=np.int32),
            quad_count=0,
        )

    positions = np.concatenate(verts, axis=0)
    quad_mats = np.array(mats, dtype=np.int32)
    quad_count = len(quad_mats)

    palette_indices = np.repeat(quad_mats, 4)

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
