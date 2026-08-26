"""Axis-mirror operations on the authoritative voxel grid.

Mirroring is a pure data-level operation: copy each occupied cell from the
source half onto its mirrored partner in the target half (overwrite
semantics). Odd extents are handled by copying/clipping against the extent,
never by warning or failing.
"""
from typing import Dict, Set, Tuple

from ..constants import BrickCoord, VoxelCoord
from .coords import split_coord
from .tagged_grid import TaggedVoxelGrid, CELL_EMPTY

AXES = {"X": 0, "Y": 1, "Z": 2}


def mirror_coord(
    extent_min: VoxelCoord,
    extent_max_exclusive: VoxelCoord,
    coord: VoxelCoord,
    axis_index: int,
) -> Tuple[VoxelCoord, bool]:
    """Return (mirrored coord, is_self) for one coordinate about the extent's
    center plane along ``axis_index``. The result may fall outside the extent
    for odd sizes; callers clip."""
    lo = int(extent_min[axis_index])
    hi = int(extent_max_exclusive[axis_index])
    # Mirror about the extent's center plane: cell i maps to lo+hi-1-i.
    # The middle cell maps to itself when the size along the axis is odd.
    mirrored = lo + hi - 1 - int(coord[axis_index])
    out = list(coord)
    out[axis_index] = mirrored
    return tuple(out), mirrored == int(coord[axis_index])


def mirror_half_to_half(
    grid: TaggedVoxelGrid,
    axis: str,
    direction: str,
) -> Tuple[int, Set[BrickCoord]]:
    """Copy the source half onto the target half along ``axis``.

    ``direction``: 'NEG_TO_POS' copies the low side onto the high side;
    'POS_TO_NEG' copies the high side onto the low side. Overwrites whatever
    the target half had. Cells whose mirror partner falls outside the extent
    are clipped. Returns (changed cell count, changed brick coords).
    """
    if axis not in AXES:
        raise ValueError(f"Unknown mirror axis: {axis}")
    if direction not in ("NEG_TO_POS", "POS_TO_NEG"):
        raise ValueError(f"Unknown mirror direction: {direction}")
    axis_index = AXES[axis]
    lo = int(grid.extent_min[axis_index])
    hi = int(grid.extent_max_exclusive[axis_index])
    center = (lo + hi) / 2.0

    neg_to_pos = direction == "NEG_TO_POS"
    changed = 0
    changed_bricks: Set[BrickCoord] = set()
    brick_size = int(grid.brick_size)

    for brick_coord, brick in list(grid.bricks.items()):
        bx, by, bz = brick_coord
        base = (bx * brick_size, by * brick_size, bz * brick_size)
        coords = [
            (base[0] + lx, base[1] + ly, base[2] + lz)
            for lx in range(brick_size)
            for ly in range(brick_size)
            for lz in range(brick_size)
            if int(brick.indices[lx, ly, lz]) > 0
        ]
        for src in coords:
            s_val = int(src[axis_index])
            if (s_val < center) != neg_to_pos:
                continue  # not in the source half
            dst, _self = mirror_coord(
                grid.extent_min, grid.extent_max_exclusive, src, axis_index
            )
            if not grid.in_extent(dst):
                continue  # clip: shorter target keeps what it has
            cell = grid.get_cell(src)
            before = grid.get_cell(dst)
            grid.set_cell(dst, cell.domain, cell.index)
            if before != cell:
                changed += 1
                changed_bricks.add(split_coord(dst, brick_size)[0])
        # Source bricks can become empty after their content moved over.
        if brick.is_empty():
            del grid.bricks[brick_coord]
            changed_bricks.add(brick_coord)

    # True copy semantics: a target voxel whose source partner is EMPTY must
    # be removed. Walk occupied target-half cells and erase those without an
    # occupied mirror partner.
    for brick_coord, brick in list(grid.bricks.items()):
        bx, by, bz = brick_coord
        base = (bx * brick_size, by * brick_size, bz * brick_size)
        coords = [
            (base[0] + lx, base[1] + ly, base[2] + lz)
            for lx in range(brick_size)
            for ly in range(brick_size)
            for lz in range(brick_size)
            if int(brick.indices[lx, ly, lz]) > 0
        ]
        for dst in coords:
            d_val = int(dst[axis_index])
            if (d_val < center) == neg_to_pos:
                continue  # not in the target half
            src, is_self = mirror_coord(
                grid.extent_min, grid.extent_max_exclusive, dst, axis_index
            )
            if is_self or not grid.in_extent(src):
                continue
            if grid.get_cell(src) == CELL_EMPTY:
                before = grid.get_cell(dst)
                grid.erase(dst)
                changed += 1
                changed_bricks.add(split_coord(dst, brick_size)[0])
        if brick.is_empty():
            del grid.bricks[brick_coord]
            changed_bricks.add(brick_coord)

    return changed, changed_bricks


def mirrored_cells_for_axes(
    grid: TaggedVoxelGrid,
    coord: VoxelCoord,
    axes: Dict[str, bool],
) -> list:
    """Return the live-mirror targets for one brush touch given active axes.

    Mirrors compose across all selected axes so painting in a corner with X+Y
    active produces the full symmetric set. Coordinates outside the extent are
    dropped.
    """
    targets = []
    current = [coord]
    for axis_name in ("X", "Y", "Z"):
        if not axes.get(axis_name):
            continue
        axis_index = AXES[axis_name]
        next_round = []
        for c in current:
            mirrored, is_self = mirror_coord(
                grid.extent_min, grid.extent_max_exclusive, c, axis_index
            )
            if not is_self and grid.in_extent(mirrored):
                next_round.append(mirrored)
        current.extend(next_round)
    for c in current[1:]:
        if c not in targets:
            targets.append(c)
    return targets


def mirror_half_paint_only(
    grid: TaggedVoxelGrid,
    axis: str,
    direction: str,
) -> Tuple[int, Set[BrickCoord]]:
    """Recolor-only mirror along ``axis``.

    Where both the source and its mirrored target cell hold a voxel **of the
    same domain**, the target adopts the source's palette index. Mismatched
    domains, empty cells, and self-mirroring cells are left untouched — no
    voxel is added or removed.
    """
    if axis not in AXES:
        raise ValueError(f"Unknown mirror axis: {axis}")
    if direction not in ("NEG_TO_POS", "POS_TO_NEG"):
        raise ValueError(f"Unknown mirror direction: {direction}")
    axis_index = AXES[axis]
    lo = int(grid.extent_min[axis_index])
    hi = int(grid.extent_max_exclusive[axis_index])
    center = (lo + hi) / 2.0

    neg_to_pos = direction == "NEG_TO_POS"
    changed = 0
    changed_bricks: Set[BrickCoord] = set()
    brick_size = int(grid.brick_size)

    for brick_coord, brick in list(grid.bricks.items()):
        bx, by, bz = brick_coord
        base = (bx * brick_size, by * brick_size, bz * brick_size)
        coords = [
            (base[0] + lx, base[1] + ly, base[2] + lz)
            for lx in range(brick_size)
            for ly in range(brick_size)
            for lz in range(brick_size)
            if int(brick.indices[lx, ly, lz]) > 0
        ]
        for src in coords:
            s_val = int(src[axis_index])
            if (s_val < center) != neg_to_pos:
                continue  # not in the source half
            dst, is_self = mirror_coord(
                grid.extent_min, grid.extent_max_exclusive, src, axis_index
            )
            if is_self or not grid.in_extent(dst):
                continue
            src_cell = grid.get_cell(src)
            dst_cell = grid.get_cell(dst)
            # Same domain only; never add/remove voxels cross-domain.
            if (
                dst_cell == CELL_EMPTY
                or src_cell.domain != dst_cell.domain
                or src_cell.index == dst_cell.index
            ):
                continue
            grid.set_cell(dst, dst_cell.domain, src_cell.index)
            changed += 1
            changed_bricks.add(split_coord(dst, brick_size)[0])

    return changed, changed_bricks
