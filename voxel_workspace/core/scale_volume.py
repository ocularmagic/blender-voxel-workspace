"""Scale (stretch/squash) the voxels inside a voxel root.

Unlike :mod:`operators.resize_volume` (which adds or removes empty space and
leaves voxel coordinates untouched), scaling remaps every occupied voxel's
coordinate together with the extent, so the shape itself stretches or
compresses.

Model: along the dragged axis every occupied source cell occupies the
half-open interval ``[c, c+1)`` inside the old extent length ``o_len``.
That interval scales proportionally into the new extent length ``n_len`` as
``[n_lo + floor(d * n_len / o_len), n_lo + ceil((d + 1) * n_len / o_len))``
and is painted completely — contiguous data therefore stays solid when
stretched (spans abut exactly), and an isolated voxel grows into as many
cells as its fractional width demands, mimicking real stretching.  When
squashing, several sources can overlap one target: the first non-empty
writer wins (cells are processed in sorted order), so nothing vanishes.

Each committed drag touches exactly one axis, so the remap is done per axis.
Works with both :class:`core.grid.VoxelGrid` (plain palette indices) and
:class:`core.tagged_grid.TaggedVoxelGrid` (index + domain pairs).
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OLD_EXTENT = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
NEW_EXTENT = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
COORD = Tuple[int, int, int]

MAX_DIM = 512


def validate_scale_extent(new_min: COORD, new_max: COORD) -> Optional[str]:
    """Return an error message if the requested scaled extents are invalid."""
    for i, name in enumerate(("X", "Y", "Z")):
        size = new_max[i] - new_min[i]
        if size < 1:
            return f"{name} must be at least 1"
        if size > MAX_DIM:
            return f"{name} cannot exceed {MAX_DIM}"
    return None


def _occupied_cells(grid: Any) -> List[Tuple[COORD, int, int]]:
    """Return ((x, y, z), domain, index) for every occupied cell."""
    from .coords import join_coord

    bs = int(grid.brick_size)
    cells: List[Tuple[COORD, int, int]] = []
    for bcoord, brick in grid.bricks.items():
        nz = np.argwhere(brick.indices > 0)
        domains = getattr(brick, "domains", None)
        for c in nz:
            lx, ly, lz = int(c[0]), int(c[1]), int(c[2])
            gc = join_coord(bcoord, (lx, ly, lz), bs)
            dom = int(domains[lx, ly, lz]) if domains is not None else 1
            cells.append(((int(gc[0]), int(gc[1]), int(gc[2])), dom,
                          int(brick.indices[lx, ly, lz])))
    return cells


def _target_span(src_pos: int, o_lo: int, o_len: int,
                 n_lo: int, n_len: int) -> Tuple[int, int]:
    """Return the inclusive target range for source cell ``src_pos``.

    The source cell occupies ``[d, d+1)`` of ``o_len`` fraction units; its
    image inside the new extent is that interval rescaled to ``n_len`` units.
    Never empty: degenerate images keep a single nearest cell.
    """
    d = src_pos - o_lo
    lo = n_lo + (d * n_len) // o_len
    hi_excl = n_lo + -((-(d + 1) * n_len) // o_len)  # ceil division
    if hi_excl <= lo:
        # Sub-cell image after shrinking: keep one representative cell.
        center = n_lo + int(round((d + 0.5) * n_len / o_len))
        lo = max(n_lo, min(n_lo + n_len - 1, center))
        hi_excl = lo + 1
    return lo, hi_excl - 1


def compute_scale_writes(grid: Any, old_extent: OLD_EXTENT,
                         new_extent: NEW_EXTENT,
                         axis: int) -> Dict[Tuple[COORD, int], int]:
    """Return writes that rescale ``axis`` from ``old_extent`` to ``new_extent``.

    Keys are (coordinate, domain); values are palette indices.  See the module
    docstring for the stretch/squash semantics.
    """
    assert 0 <= axis <= 2
    o_lo, o_hi = old_extent[0][axis], old_extent[1][axis]
    n_lo, n_hi = new_extent[0][axis], new_extent[1][axis]
    o_len = max(1, o_hi - o_lo)
    n_len = max(1, n_hi - n_lo)

    others = tuple(i for i in range(3) if i != axis)
    cells = sorted(_occupied_cells(grid), key=lambda cell: (
        tuple(cell[0][i] for i in others) + (cell[0][axis],)))

    writes: Dict[Tuple[COORD, int], int] = {}

    def put(coord: COORD, dom: int, index: int) -> None:
        key = (coord, dom)
        existing = writes.get(key)
        # Squash overlap resolution: first non-empty writer wins, and EMPTY
        # never erases a previously-written value.
        if existing is None or (existing == 0 and index != 0):
            writes[key] = index

    for coord, dom, index in cells:
        t_lo, t_hi = _target_span(coord[axis], o_lo, o_len, n_lo, n_len)
        for t in range(t_lo, t_hi + 1):
            dst = list(coord)
            dst[axis] = t
            put(tuple(dst), dom, index)
    return writes


def commit_writes(grid: Any, writes: Dict[Tuple[COORD, int], int],
                  new_extent: NEW_EXTENT) -> None:
    """Replace grid contents with ``writes``, clamped to the new extent.

    The caller must already have moved ``grid.extent_min`` /
    ``extent_max_exclusive`` to ``new_extent`` (as the operator does before
    committing), because ``set``/``set_cell`` reject out-of-extent coords.
    """
    grid.bricks.clear()
    grid.dirty_bricks.clear()
    emin, emax = new_extent
    tagged = hasattr(grid, "set_cell")
    for (coord, dom), index in writes.items():
        if all(emin[i] <= coord[i] < emax[i] for i in range(3)):
            if tagged:
                grid.set_cell(coord, dom, index)
            else:
                grid.set(coord, index)


def apply_scaled_axis(mesh: Any, grid: Any, entry: Any,
                      old_extent: OLD_EXTENT, new_extent: NEW_EXTENT,
                      axis: int) -> None:
    """Rescale ``axis`` on the grid and rebuild the derived volume state."""
    writes = compute_scale_writes(grid, old_extent, new_extent, axis)
    grid.extent_min = tuple(new_extent[0])
    grid.extent_max_exclusive = tuple(new_extent[1])
    commit_writes(grid, writes, new_extent)
    from ..blender.persistence import serialize_volume
    serialize_volume(mesh, grid, dirty_only=False)
    from ..blender.mesh_sync import sync_volume_mesh
    entry.cpu_buffers.clear()
    sync_volume_mesh(mesh, grid=grid, entry=entry, dirty_only=False,
                     ensure_material=False, voxel_size=float(entry.voxel_size))
