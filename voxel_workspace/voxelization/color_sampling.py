"""Nearest-surface color sampling for occupied voxels."""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from .occupancy import _HALF_DIAG, barycentric, closest_point_on_triangle


@dataclass
class SampledMaterial:
    """Evaluated GLB material usable without bpy."""

    base_color: Tuple[float, float, float, float]
    image: Optional[np.ndarray] = None  # (H, W, 4) linear RGBA float32
    name: str = ""
    warnings: List[str] = field(default_factory=list)


def wrap01(u: float) -> float:
    """Repeat wrap into [0, 1)."""
    return u - np.floor(u)


def sample_image_nearest(image: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
    """Nearest-texel sample a (H, W, 4) image with repeat wrap. UV origin is bottom-left."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros(4, dtype=np.float32)
    u = wrap01(float(uv[0]))
    v = wrap01(float(uv[1]))
    x = int(np.floor(u * w)) % w
    y = int(np.floor(v * h)) % h
    return np.asarray(image[y, x], dtype=np.float32)


def sample_image_nearest_batch(image: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Nearest-texel sample a (H, W, 4) image at (N, 2) UVs with repeat wrap."""
    h, w = image.shape[:2]
    uv = np.asarray(uvs, dtype=np.float64).reshape((-1, 2))
    if h <= 0 or w <= 0 or len(uv) == 0:
        return np.zeros((len(uv), 4), dtype=np.float32)
    u = uv[:, 0] - np.floor(uv[:, 0])
    v = uv[:, 1] - np.floor(uv[:, 1])
    x = np.floor(u * w).astype(np.int32) % w
    y = np.floor(v * h).astype(np.int32) % h
    return np.asarray(image[y, x], dtype=np.float32)


def sample_image(image: np.ndarray, uv: Tuple[float, float]) -> np.ndarray:
    """Bilinear sample a (H, W, 4) linear image with repeat wrap. UV origin is bottom-left."""
    h, w = image.shape[:2]
    u = wrap01(float(uv[0])) * w - 0.5
    v = wrap01(float(uv[1])) * h - 0.5
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    tx = u - x0
    ty = v - y0
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    x0 %= w
    y0 %= h
    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    return (1.0 - ty) * ((1.0 - tx) * c00 + tx * c10) + ty * ((1.0 - tx) * c01 + tx * c11)


def sample_material(
    material: SampledMaterial,
    uv: Optional[Tuple[float, float]],
) -> np.ndarray:
    """Return linear RGBA for a material, using UV when an image is present."""
    if material.image is not None and uv is not None:
        return sample_image_nearest(material.image, uv)
    return np.asarray(material.base_color, dtype=np.float32)


def _color_from_hit(
    tri_index: int,
    hit_point: np.ndarray,
    triangles: np.ndarray,
    uvs: Optional[np.ndarray],
    material_indices: np.ndarray,
    materials: Sequence[SampledMaterial],
    fallback: np.ndarray,
) -> np.ndarray:
    """Nearest-texel (or base color) for a surface hit on triangle tri_index."""
    mi = int(material_indices[tri_index]) if tri_index < len(material_indices) else 0
    if mi < 0 or mi >= len(materials):
        return fallback
    mat = materials[mi]
    uv = None
    if uvs is not None and tri_index < len(uvs):
        a, b, c = triangles[tri_index]
        wu, wv, ww = barycentric(hit_point, a, b, c)
        uv_tri = uvs[tri_index]
        uv = (
            float(wu * uv_tri[0, 0] + wv * uv_tri[1, 0] + ww * uv_tri[2, 0]),
            float(wu * uv_tri[0, 1] + wv * uv_tri[1, 1] + ww * uv_tri[2, 1]),
        )
    if mat.image is not None and uv is not None:
        return sample_image_nearest(mat.image, uv)
    return np.asarray(mat.base_color, dtype=np.float32)


def _try_bvh_tree(tris: np.ndarray):
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return None
    n = int(len(tris))
    if n == 0:
        return None
    vertices = tris.reshape(-1, 3).tolist()
    polygons = [(i * 3, i * 3 + 1, i * 3 + 2) for i in range(n)]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=True)


def _nearest_hits_bvh(centers: np.ndarray, tree) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tri_index or -1, hit_point) for each center via BVH nearest-hit."""
    n = len(centers)
    indices = np.full(n, -1, dtype=np.int32)
    hits = np.zeros((n, 3), dtype=np.float64)
    for i, p in enumerate(centers):
        loc, _normal, index, _dist = tree.find_nearest((float(p[0]), float(p[1]), float(p[2])))
        if index is None:
            continue
        indices[i] = int(index)
        hits[i] = (float(loc[0]), float(loc[1]), float(loc[2]))
    return indices, hits


def _nearest_hits_numpy(centers: np.ndarray, tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Small-mesh fallback: closest triangle to each center. Not for million-tri GLBs."""
    n = len(centers)
    tcount = len(tris)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty((0, 3), dtype=np.float64)
    if tcount * n > 2_000_000:
        raise RuntimeError(
            "mathutils.bvhtree is required to sample large meshes; "
            "refusing a per-voxel triangle loop"
        )
    indices = np.full(n, -1, dtype=np.int32)
    hits = np.zeros((n, 3), dtype=np.float64)
    best = np.full(n, np.inf, dtype=np.float64)
    for t_idx, tri in enumerate(tris):
        a, b, c = tri[0], tri[1], tri[2]
        for i in range(n):
            q = closest_point_on_triangle(centers[i], a, b, c)
            delta = centers[i] - q
            d2 = float(np.dot(delta, delta))
            if d2 < best[i]:
                best[i] = d2
                indices[i] = t_idx
                hits[i] = q
    return indices, hits


def sample_occupied_nearest_surface(
    occupied: np.ndarray,
    triangles: np.ndarray,
    uvs: Optional[np.ndarray],
    material_indices: np.ndarray,
    materials: Sequence[SampledMaterial],
    extent_min: Tuple[int, int, int],
    fallback: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
) -> np.ndarray:
    """Color every occupied voxel from the nearest surface UV, nearest texel.

    Occupied voxel centers are queried with mathutils.bvhtree (or a small-mesh
    numpy fallback). Interior cells use the same nearest-surface sample.
    """
    from .occupancy import occupied_coords

    shape = occupied.shape
    colors = np.zeros(shape + (4,), dtype=np.float32)
    fb = np.asarray(fallback, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.size == 0 or not np.any(occupied):
        return colors
    coords = occupied_coords(occupied, extent_min)
    centers = coords.astype(np.float64) + 0.5
    tree = _try_bvh_tree(tris)
    if tree is not None:
        tri_ids, hits = _nearest_hits_bvh(centers, tree)
    else:
        tri_ids, hits = _nearest_hits_numpy(centers, tris)
    mat_i = np.asarray(material_indices, dtype=np.int32)
    uv_arr = None if uvs is None else np.asarray(uvs, dtype=np.float64)
    local = coords - np.array(extent_min, dtype=np.int32)
    for i in range(len(coords)):
        t_idx = int(tri_ids[i])
        if t_idx < 0:
            col = fb
        else:
            col = _color_from_hit(t_idx, hits[i], tris, uv_arr, mat_i, materials, fb)
        colors[int(local[i, 0]), int(local[i, 1]), int(local[i, 2])] = col
    return colors


def sample_image_batch(image: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Bilinear sample a (H, W, 4) image at (N, 2) UVs with repeat wrap."""
    h, w = image.shape[:2]
    uv = np.asarray(uvs, dtype=np.float64).reshape((-1, 2))
    u = (uv[:, 0] - np.floor(uv[:, 0])) * w - 0.5
    v = (uv[:, 1] - np.floor(uv[:, 1])) * h - 0.5
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    tx = (u - x0).astype(np.float32)
    ty = (v - y0).astype(np.float32)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    x0 %= w
    y0 %= h
    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    w0 = (1.0 - tx)[:, None]
    w1 = tx[:, None]
    return ((1.0 - ty)[:, None] * (w0 * c00 + w1 * c10) + ty[:, None] * (w0 * c01 + w1 * c11)).astype(np.float32)


def _triangle_colors(
    triangles: np.ndarray,
    uvs: Optional[np.ndarray],
    material_indices: np.ndarray,
    materials: Sequence[SampledMaterial],
    fallback: Tuple[float, float, float, float],
) -> np.ndarray:
    """Linear RGBA at each triangle centroid."""
    n = len(triangles)
    out = np.empty((n, 4), dtype=np.float32)
    out[:] = fallback
    uv_c = None if uvs is None else np.mean(np.asarray(uvs, dtype=np.float64), axis=1)
    mat_i = np.asarray(material_indices, dtype=np.int32)
    for mi, mat in enumerate(materials):
        sel = mat_i == mi
        if not np.any(sel):
            continue
        if mat.image is not None and uv_c is not None:
            out[sel] = sample_image_batch(mat.image, uv_c[sel])
        else:
            out[sel] = np.asarray(mat.base_color, dtype=np.float32)
    return out


def sample_shell_colors(
    shell: np.ndarray,
    triangles: np.ndarray,
    uvs: Optional[np.ndarray],
    material_indices: np.ndarray,
    materials: Sequence[SampledMaterial],
    extent_min: Tuple[int, int, int],
    fallback: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
) -> np.ndarray:
    """Assign nearest-surface linear RGBA to every True cell in shell.

    Dense meshes vote per triangle AABB (vectorized). Occupied cells that
    receive no vote keep `fallback` and can be filled by interior propagation.
    """
    shape = shell.shape
    colors = np.zeros(shape + (4,), dtype=np.float32)
    dist_sq = np.full(shape, np.inf, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.size == 0:
        return colors
    tri_cols = _triangle_colors(tris, uvs, material_indices, materials, fallback)
    emin = extent_min
    emax = (emin[0] + shape[0], emin[1] + shape[1], emin[2] + shape[2])
    half = float(_HALF_DIAG)
    centroids = tris.mean(axis=1)
    tmin = tris.min(axis=1) - half
    tmax = tris.max(axis=1) + half
    i0 = np.maximum(emin[0], np.floor(tmin[:, 0]).astype(np.int32))
    j0 = np.maximum(emin[1], np.floor(tmin[:, 1]).astype(np.int32))
    k0 = np.maximum(emin[2], np.floor(tmin[:, 2]).astype(np.int32))
    i1 = np.minimum(emax[0], np.floor(tmax[:, 0]).astype(np.int32) + 1)
    j1 = np.minimum(emax[1], np.floor(tmax[:, 1]).astype(np.int32) + 1)
    k1 = np.minimum(emax[2], np.floor(tmax[:, 2]).astype(np.int32) + 1)
    valid = (i0 < i1) & (j0 < j1) & (k0 < k1)
    i0, j0, k0, i1, j1, k1 = i0[valid], j0[valid], k0[valid], i1[valid], j1[valid], k1[valid]
    centroids = centroids[valid]
    tri_cols = tri_cols[valid]
    dx = i1 - i0
    dy = j1 - j0
    dz = k1 - k0
    max_side = np.maximum(np.maximum(dx, dy), dz)
    small = max_side <= 4
    if np.any(small):
        _scatter_aabb_colors(
            colors,
            dist_sq,
            shell,
            i0[small], j0[small], k0[small],
            dx[small], dy[small], dz[small],
            centroids[small],
            tri_cols[small],
            emin,
        )
    large_idx = np.nonzero(~small)[0]
    if large_idx.size:
        tris_v = tris[valid]
        for t in large_idx:
            _paint_large_triangle_color(
                colors, dist_sq, shell,
                tris_v[t, 0], tris_v[t, 1], tris_v[t, 2],
                int(i0[t]), int(i1[t]), int(j0[t]), int(j1[t]), int(k0[t]), int(k1[t]),
                tri_cols[t], emin,
            )
    missing = shell & ~np.isfinite(dist_sq)
    if np.any(missing):
        colors[missing] = np.asarray(fallback, dtype=np.float32)
    return colors


def _scatter_aabb_colors(
    colors: np.ndarray,
    dist_sq: np.ndarray,
    shell: np.ndarray,
    i0: np.ndarray,
    j0: np.ndarray,
    k0: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    centroids: np.ndarray,
    tri_cols: np.ndarray,
    emin: Tuple[int, int, int],
    chunk: int = 65536,
) -> None:
    sx, sy, sz = shell.shape
    max_off = 4
    ox, oy, oz = np.meshgrid(
        np.arange(max_off, dtype=np.int32),
        np.arange(max_off, dtype=np.int32),
        np.arange(max_off, dtype=np.int32),
        indexing="ij",
    )
    ox, oy, oz = ox.ravel(), oy.ravel(), oz.ravel()
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
        rows = np.nonzero(inside)[0] + start
        if ix.size == 0:
            continue
        inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < sx) & (iy < sy) & (iz < sz)
        ix, iy, iz, rows = ix[inb], iy[inb], iz[inb], rows[inb]
        on = shell[ix, iy, iz]
        if not np.any(on):
            continue
        ix, iy, iz, rows = ix[on], iy[on], iz[on], rows[on]
        centers = np.stack((ix + emin[0] + 0.5, iy + emin[1] + 0.5, iz + emin[2] + 0.5), axis=1)
        delta = centers - centroids[rows]
        d2 = np.einsum("ij,ij->i", delta, delta)
        lin = (ix.astype(np.int64) * sy + iy.astype(np.int64)) * sz + iz.astype(np.int64)
        order = np.lexsort((d2, lin))
        _, first = np.unique(lin[order], return_index=True)
        best = order[first]
        bx, by, bz = ix[best], iy[best], iz[best]
        better = d2[best] < dist_sq[bx, by, bz]
        if not np.any(better):
            continue
        bx, by, bz = bx[better], by[better], bz[better]
        dist_sq[bx, by, bz] = d2[best][better]
        colors[bx, by, bz] = tri_cols[rows[best][better]]


def _paint_large_triangle_color(
    colors: np.ndarray,
    dist_sq: np.ndarray,
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
    rgba: np.ndarray,
    emin: Tuple[int, int, int],
) -> None:
    from .occupancy import closest_points_on_triangle, _HALF_DIAG_SQ

    ii = np.arange(i0, i1, dtype=np.int32)
    jj = np.arange(j0, j1, dtype=np.int32)
    kk = np.arange(k0, k1, dtype=np.int32)
    if ii.size == 0 or jj.size == 0 or kk.size == 0:
        return
    xx, yy, zz = np.meshgrid(ii + 0.5, jj + 0.5, kk + 0.5, indexing="ij")
    P = np.stack((xx, yy, zz), axis=-1).reshape((-1, 3))
    Q = closest_points_on_triangle(P, a, b, c)
    delta = P - Q
    d2 = np.einsum("ij,ij->i", delta, delta)
    hit = d2 <= _HALF_DIAG_SQ
    if not np.any(hit):
        return
    pts = P[hit]
    d2 = d2[hit]
    ix = np.floor(pts[:, 0]).astype(np.int32) - emin[0]
    iy = np.floor(pts[:, 1]).astype(np.int32) - emin[1]
    iz = np.floor(pts[:, 2]).astype(np.int32) - emin[2]
    sx, sy, sz = shell.shape
    inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < sx) & (iy < sy) & (iz < sz)
    ix, iy, iz, d2 = ix[inb], iy[inb], iz[inb], d2[inb]
    on = shell[ix, iy, iz]
    ix, iy, iz, d2 = ix[on], iy[on], iz[on], d2[on]
    better = d2 < dist_sq[ix, iy, iz]
    if not np.any(better):
        return
    ix, iy, iz = ix[better], iy[better], iz[better]
    dist_sq[ix, iy, iz] = d2[better]
    colors[ix, iy, iz] = np.asarray(rgba, dtype=np.float32)


def propagate_interior_colors(occupied: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Copy nearest shell colors into interior cells with a 6-connected BFS."""
    from collections import deque

    out = np.array(colors, copy=True, dtype=np.float32)
    sx, sy, sz = occupied.shape
    has_color = occupied & (np.any(out[..., :3] > 0.0, axis=-1) | (out[..., 3] > 0.0))
    if not np.any(has_color):
        return out
    dist = np.full((sx, sy, sz), np.iinfo(np.int32).max, dtype=np.int32)
    q = deque()
    ys, xs, zs = np.nonzero(has_color)
    # np.nonzero returns axis0, axis1, axis2
    for i, j, k in zip(ys, xs, zs):
        dist[i, j, k] = 0
        q.append((int(i), int(j), int(k)))
    while q:
        i, j, k = q.popleft()
        nd = int(dist[i, j, k]) + 1
        col = out[i, j, k]
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or nj < 0 or nk < 0 or ni >= sx or nj >= sy or nk >= sz:
                continue
            if not occupied[ni, nj, nk]:
                continue
            if nd < dist[ni, nj, nk]:
                dist[ni, nj, nk] = nd
                out[ni, nj, nk] = col
                q.append((ni, nj, nk))
    return out
