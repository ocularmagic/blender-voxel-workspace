"""Direct occupancy classification in voxel-index space (no Geometry Nodes).

Shell mode marks cells whose centers are within a conservative distance of a
source triangle (half the voxel cube diagonal). Solid mode flood-fills the
closed interior from outside; open meshes should stay in shell mode.
"""
from collections import deque
from typing import Sequence, Tuple
import numpy as np

VoxelCoord = Tuple[int, int, int]
# Distance from voxel center to a cube corner.
_HALF_DIAG = 0.5 * np.sqrt(3.0)
_HALF_DIAG_SQ = float(_HALF_DIAG * _HALF_DIAG)


def closest_point_on_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Return the closest point on triangle ABC to point P (Ericson)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    """Clamped barycentric coordinates of P with respect to triangle ABC."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (float(u), float(v), float(w))


def _extent_shape(extent_min: VoxelCoord, extent_max: VoxelCoord) -> Tuple[int, int, int]:
    sx = int(extent_max[0] - extent_min[0])
    sy = int(extent_max[1] - extent_min[1])
    sz = int(extent_max[2] - extent_min[2])
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError("Volume extent is empty")
    return (sx, sy, sz)


def closest_points_on_triangle(P: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on triangle ABC for each row of P (N, 3)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    P = np.asarray(P, dtype=np.float64).reshape((-1, 3))
    ab = b - a
    ac = c - a
    ap = P - a
    d1 = ap @ ab
    d2 = ap @ ac
    out = np.empty_like(P)
    assigned = np.zeros(len(P), dtype=bool)

    m = (d1 <= 0.0) & (d2 <= 0.0)
    out[m] = a
    assigned |= m

    bp = P - b
    d3 = bp @ ab
    d4 = bp @ ac
    m = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    out[m] = b
    assigned |= m

    vc = d1 * d4 - d3 * d2
    m = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = np.zeros(len(P), dtype=np.float64)
    denom = d1 - d3
    sel = m & (np.abs(denom) > 1e-20)
    v[sel] = d1[sel] / denom[sel]
    out[m] = a + v[m, None] * ab
    assigned |= m

    cp = P - c
    d5 = cp @ ab
    d6 = cp @ ac
    m = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    out[m] = c
    assigned |= m

    vb = d5 * d2 - d1 * d6
    m = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = np.zeros(len(P), dtype=np.float64)
    denom = d2 - d6
    sel = m & (np.abs(denom) > 1e-20)
    w[sel] = d2[sel] / denom[sel]
    out[m] = a + w[m, None] * ac
    assigned |= m

    va = d3 * d6 - d5 * d4
    m = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w = np.zeros(len(P), dtype=np.float64)
    denom = (d4 - d3) + (d5 - d6)
    sel = m & (np.abs(denom) > 1e-20)
    w[sel] = (d4[sel] - d3[sel]) / denom[sel]
    out[m] = b + w[m, None] * (c - b)
    assigned |= m

    rest = ~assigned
    if np.any(rest):
        denom = va[rest] + vb[rest] + vc[rest]
        denom = np.where(np.abs(denom) < 1e-20, 1.0, denom)
        vv = vb[rest] / denom
        ww = vc[rest] / denom
        out[rest] = a + ab * vv[:, None] + ac * ww[:, None]
    return out


def occupancy_shell(
    triangles: np.ndarray,
    extent_min: VoxelCoord,
    extent_max_exclusive: VoxelCoord,
) -> np.ndarray:
    """Return a boolean volume of cells whose centers are near a triangle.

    Small-triangle AABBs (typical after contain-fit of dense meshes) are marked
    with a vectorized scatter. Large triangles fall back to a closest-point test.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    emin = (int(extent_min[0]), int(extent_min[1]), int(extent_min[2]))
    emax = (int(extent_max_exclusive[0]), int(extent_max_exclusive[1]), int(extent_max_exclusive[2]))
    shape = _extent_shape(emin, emax)
    shell = np.zeros(shape, dtype=bool)
    if tris.size == 0:
        return shell
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (T, 3, 3)")

    ab = tris[:, 1] - tris[:, 0]
    ac = tris[:, 2] - tris[:, 0]
    cross = np.cross(ab, ac)
    area2 = np.einsum("ij,ij->i", cross, cross)
    alive = area2 >= 1e-18
    if not np.any(alive):
        return shell
    tris = tris[alive]
    half = _HALF_DIAG
    tmin = tris.min(axis=1) - half
    tmax = tris.max(axis=1) + half
    i0 = np.maximum(emin[0], np.floor(tmin[:, 0]).astype(np.int32))
    j0 = np.maximum(emin[1], np.floor(tmin[:, 1]).astype(np.int32))
    k0 = np.maximum(emin[2], np.floor(tmin[:, 2]).astype(np.int32))
    i1 = np.minimum(emax[0], np.floor(tmax[:, 0]).astype(np.int32) + 1)
    j1 = np.minimum(emax[1], np.floor(tmax[:, 1]).astype(np.int32) + 1)
    k1 = np.minimum(emax[2], np.floor(tmax[:, 2]).astype(np.int32) + 1)
    valid = (i0 < i1) & (j0 < j1) & (k0 < k1)
    if not np.any(valid):
        return shell
    i0, j0, k0, i1, j1, k1 = i0[valid], j0[valid], k0[valid], i1[valid], j1[valid], k1[valid]
    tris = tris[valid]
    dx = i1 - i0
    dy = j1 - j0
    dz = k1 - k0
    max_side = np.maximum(np.maximum(dx, dy), dz)
    small = max_side <= 4
    if np.any(small):
        _scatter_aabb_shell(
            shell,
            i0[small], j0[small], k0[small],
            dx[small], dy[small], dz[small],
            emin,
        )
    large_idx = np.nonzero(~small)[0]
    for t in large_idx:
        _mark_large_triangle_shell(
            shell,
            tris[t, 0], tris[t, 1], tris[t, 2],
            int(i0[t]), int(i1[t]), int(j0[t]), int(j1[t]), int(k0[t]), int(k1[t]),
            emin,
        )
    return shell


def _scatter_aabb_shell(
    shell: np.ndarray,
    i0: np.ndarray,
    j0: np.ndarray,
    k0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    emin: VoxelCoord,
    chunk: int = 65536,
) -> None:
    """Mark voxels covered by compact triangle AABBs (max side 4)."""
    sx, sy, sz = shell.shape
    max_off = 4
    ox, oy, oz = np.meshgrid(
        np.arange(max_off, dtype=np.int32),
        np.arange(max_off, dtype=np.int32),
        np.arange(max_off, dtype=np.int32),
        indexing="ij",
    )
    ox = ox.ravel()
    oy = oy.ravel()
    oz = oz.ravel()
    n = len(i0)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        inside = (
            (ox[None, :] < dx[sl, None])
            & (oy[None, :] < dy[sl, None])
            & (oz[None, :] < dz[sl, None])
        )
        ix = (i0[sl, None] + ox[None, :])[inside] - emin[0]
        iy = (j0[sl, None] + oy[None, :])[inside] - emin[1]
        iz = (k0[sl, None] + oz[None, :])[inside] - emin[2]
        if ix.size == 0:
            continue
        inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < sx) & (iy < sy) & (iz < sz)
        shell[ix[inb], iy[inb], iz[inb]] = True


def _mark_large_triangle_shell(
    shell: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    k0: int,
    k1: int,
    emin: VoxelCoord,
) -> None:
    ii = np.arange(i0, i1, dtype=np.int32)
    jj = np.arange(j0, j1, dtype=np.int32)
    kk = np.arange(k0, k1, dtype=np.int32)
    if ii.size == 0 or jj.size == 0 or kk.size == 0:
        return
    xx, yy, zz = np.meshgrid(ii + 0.5, jj + 0.5, kk + 0.5, indexing="ij")
    P = np.stack((xx, yy, zz), axis=-1).reshape((-1, 3))
    Q = closest_points_on_triangle(P, a, b, c)
    delta = P - Q
    hit = np.einsum("ij,ij->i", delta, delta) <= _HALF_DIAG_SQ
    if not np.any(hit):
        return
    pts = P[hit]
    ix = np.floor(pts[:, 0]).astype(np.int32) - emin[0]
    iy = np.floor(pts[:, 1]).astype(np.int32) - emin[1]
    iz = np.floor(pts[:, 2]).astype(np.int32) - emin[2]
    sx, sy, sz = shell.shape
    inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < sx) & (iy < sy) & (iz < sz)
    shell[ix[inb], iy[inb], iz[inb]] = True


def occupancy_solid(
    shell: np.ndarray,
) -> np.ndarray:
    """Fill interior cells of a shell occupancy via outside flood-fill.

    Cells unreachable from the padded exterior and not already shell are
    treated as interior. A leaky (open) shell therefore yields approximately
    the shell itself.
    """
    if shell.ndim != 3:
        raise ValueError("shell must be a 3D boolean array")
    blocked = np.asarray(shell, dtype=bool)
    sx, sy, sz = blocked.shape
    pad = np.zeros((sx + 2, sy + 2, sz + 2), dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = blocked
    reached = np.zeros_like(pad)
    q: deque = deque()

    def _seed(x: int, y: int, z: int) -> None:
        if not reached[x, y, z] and not pad[x, y, z]:
            reached[x, y, z] = True
            q.append((x, y, z))

    nx, ny, nz = pad.shape
    for y in range(ny):
        for z in range(nz):
            _seed(0, y, z)
            _seed(nx - 1, y, z)
    for x in range(nx):
        for z in range(nz):
            _seed(x, 0, z)
            _seed(x, ny - 1, z)
    for x in range(nx):
        for y in range(ny):
            _seed(x, y, 0)
            _seed(x, y, nz - 1)

    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if nx_ < 0 or ny_ < 0 or nz_ < 0 or nx_ >= nx or ny_ >= ny or nz_ >= nz:
                continue
            if reached[nx_, ny_, nz_] or pad[nx_, ny_, nz_]:
                continue
            reached[nx_, ny_, nz_] = True
            q.append((nx_, ny_, nz_))

    inner_reached = reached[1:-1, 1:-1, 1:-1]
    interior = ~inner_reached & ~blocked
    return blocked | interior


def occupied_coords(mask: np.ndarray, extent_min: VoxelCoord) -> np.ndarray:
    """Return (N, 3) int32 voxel coordinates for True cells in mask."""
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.int32)
    origin = np.array(extent_min, dtype=np.int32)
    return (idx.astype(np.int32) + origin).astype(np.int32)


def axis_aligned_box_triangles(mn: Sequence[float], mx: Sequence[float]) -> np.ndarray:
    """Twelve triangles of an axis-aligned box (for tests and synthetic meshes)."""
    x0, y0, z0 = (float(v) for v in mn)
    x1, y1, z1 = (float(v) for v in mx)
    v = np.array(
        [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ],
        dtype=np.float64,
    )
    faces = (
        (0, 1, 2, 3),  # -Z
        (4, 7, 6, 5),  # +Z
        (0, 4, 5, 1),  # -Y
        (3, 2, 6, 7),  # +Y
        (0, 3, 7, 4),  # -X
        (1, 5, 6, 2),  # +X
    )
    tris = []
    for a, b, c, d in faces:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return np.asarray(tris, dtype=np.float64)
