"""Multi-cell brush stamp geometry: sphere and cube footprints.

A brush anchor cell (from ray picking) expands into a footprint of voxel
offsets. All shapes exclude their exact-radius boundary ring (distance ==
radius is NOT included) so ``radius == 1`` yields exactly one voxel,
matching the historical single-voxel brush behavior.

The UI exposes the brush size as a DIAMETER (footprint width in voxels).
Internally that maps to a radius: ``R = (D + 1) // 2``, giving footprint
widths of odd sizes ``2R - 1``; even diameters snap down to the previous
odd size.
"""
from typing import Dict, FrozenSet, List, Sequence, Tuple

from ..constants import VoxelCoord

SHAPE_SPHERE = "SPHERE"
SHAPE_CUBE = "CUBE"

_BRUSH_SHAPES = (SHAPE_SPHERE, SHAPE_CUBE)

_OFFSET_CACHE: Dict[Tuple[str, int], Tuple[VoxelCoord, ...]] = {}


def radius_from_diameter(diameter: int) -> int:
    """Map a UI diameter to the internal stamp radius."""
    return max(1, (int(diameter) + 1) // 2)


def normalize_shape(shape: str) -> str:
    normalized = str(shape).upper()
    return normalized if normalized in _BRUSH_SHAPES else SHAPE_SPHERE


def stamp_offsets(shape: str, radius: int) -> Tuple[VoxelCoord, ...]:
    """Return the voxel offset set for a brush shape, center-ordered.

    Cube: Chebyshev footprint, side length ``2R - 1``.
    Sphere: Euclidean footprint, cell centers strictly within radius ``R``.
    """
    r = max(1, int(radius))
    key = (normalize_shape(shape), r)
    cached = _OFFSET_CACHE.get(key)
    if cached is not None:
        return cached

    r_squared = r * r
    offsets: List[VoxelCoord] = []
    for dx in range(-(r - 1), r):
        for dy in range(-(r - 1), r):
            for dz in range(-(r - 1), r):
                if key[0] == SHAPE_CUBE or (dx * dx + dy * dy + dz * dz) < r_squared:
                    offsets.append((dx, dy, dz))
    # Near-anchor cells first: cheap early-out for heavily overlapped drags.
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
    result = tuple(offsets)
    _OFFSET_CACHE[key] = result
    return result


def stamp_cells(
    grid: Sequence,
    center: VoxelCoord,
    shape: str,
    radius: int,
) -> List[VoxelCoord]:
    """Expand the brush footprint around ``center``, clipped to the grid extent.

    Clipping is intentional (v0.19.0): a large brush anchored outside the
    root still paints its overlap with the root. Cells beyond the extent are
    simply skipped — never clamped back onto the anchor, so a fully out-of-
    reach footprint paints nothing rather than teleporting into bounds.
    """
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    cells: List[VoxelCoord] = []
    for ox, oy, oz in stamp_offsets(shape, radius):
        candidate = (cx + ox, cy + oy, cz + oz)
        if grid.in_extent(candidate):
            cells.append(candidate)
    return cells
