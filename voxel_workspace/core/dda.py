"""Object-local voxel picking with a grid-space 3D DDA."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

from ..constants import VoxelCoord
from .grid import VoxelGrid


@dataclass(frozen=True)
class VoxelHit:
    cell: VoxelCoord
    normal: VoxelCoord
    previous_empty: VoxelCoord
    distance: float


def _normalized(direction: Sequence[float]) -> tuple[float, float, float] | None:
    length = math.sqrt(sum(float(value) ** 2 for value in direction))
    if not math.isfinite(length) or length == 0.0:
        return None
    return tuple(float(value) / length for value in direction)  # type: ignore[return-value]


def _aabb_interval(
    grid: VoxelGrid,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[float, float, VoxelCoord] | None:
    t_enter = -math.inf
    t_exit = math.inf
    entry_axis = -1
    entry_normal = [0, 0, 0]

    for axis in range(3):
        lower = float(grid.extent_min[axis])
        upper = float(grid.extent_max_exclusive[axis])
        component = direction[axis]
        if component == 0.0:
            if origin[axis] < lower or origin[axis] >= upper:
                return None
            continue

        near = (lower - origin[axis]) / component
        far = (upper - origin[axis]) / component
        normal = -1
        if near > far:
            near, far = far, near
            normal = 1

        if near > t_enter:
            t_enter = near
            entry_axis = axis
            entry_normal = [0, 0, 0]
            entry_normal[axis] = normal
        if far < t_exit:
            t_exit = far
        if t_enter > t_exit:
            return None

    if entry_axis < 0:
        entry_normal = [0, 0, 0]
    return t_enter, t_exit, tuple(entry_normal)  # type: ignore[return-value]


def trace_grid(
    grid: VoxelGrid,
    origin_grid: Sequence[float],
    direction_grid: Sequence[float],
    max_distance: float,
) -> VoxelHit | None:
    """Return the first occupied voxel touched by a ray.

    The direction is normalized internally, so ``distance`` and
    ``max_distance`` are Euclidean distances in grid units. Simultaneous
    boundary crossings use a deterministic supercover (X, then Y, then Z)
    so face- and edge-touching cells are not skipped.
    """
    if max_distance < 0.0:
        return None
    direction = _normalized(direction_grid)
    if direction is None:
        return None
    origin = tuple(float(value) for value in origin_grid)
    if len(origin) != 3:
        raise ValueError("origin_grid must have three components")

    origin_cell = tuple(math.floor(value) for value in origin)
    if grid.in_extent(origin_cell) and grid.get(origin_cell) != 0:
        return VoxelHit(origin_cell, (0, 0, 0), origin_cell, 0.0)

    interval = _aabb_interval(grid, origin, direction)
    if interval is None:
        return None
    t_enter, t_exit, entry_normal = interval
    if t_exit < 0.0:
        return None
    start_t = max(0.0, t_enter)
    end_t = min(float(max_distance), t_exit)
    if start_t > end_t:
        return None

    # Nudge only rays entering from outside so a point on the maximum AABB
    # face lands in the first interior cell rather than the exclusive bound.
    sample_t = start_t + (1e-9 if t_enter > 0.0 else 0.0)
    point = tuple(origin[i] + direction[i] * sample_t for i in range(3))
    current = tuple(math.floor(value) for value in point)
    if not grid.in_extent(current):
        return None

    if grid.get(current) != 0:
        previous = tuple(current[i] + entry_normal[i] for i in range(3))
        return VoxelHit(current, entry_normal, previous, start_t)

    step = tuple(1 if value > 0.0 else -1 if value < 0.0 else 0 for value in direction)
    t_delta = tuple(abs(1.0 / value) if value != 0.0 else math.inf for value in direction)
    t_max_list: list[float] = []
    for axis in range(3):
        if step[axis] > 0:
            boundary = current[axis] + 1.0
        elif step[axis] < 0:
            boundary = float(current[axis])
        else:
            t_max_list.append(math.inf)
            continue
        t_max_list.append(sample_t + (boundary - point[axis]) / direction[axis])

    previous_empty = current
    epsilon = 1e-10
    while True:
        next_t = min(t_max_list)
        if next_t > end_t + epsilon:
            return None
        tied = [axis for axis, value in enumerate(t_max_list) if abs(value - next_t) <= epsilon]

        # Test every cell touched at this simultaneous crossing. Subsets are
        # ordered by cardinality then axis tuple for deterministic normals.
        for size in range(1, len(tied) + 1):
            for axes in itertools.combinations(tied, size):
                candidate = list(current)
                for axis in axes:
                    candidate[axis] += step[axis]
                candidate_coord = tuple(candidate)
                if not grid.in_extent(candidate_coord):
                    continue
                if grid.get(candidate_coord) != 0:
                    normal = [0, 0, 0]
                    normal[axes[0]] = -step[axes[0]]
                    return VoxelHit(candidate_coord, tuple(normal), current, next_t)  # type: ignore[arg-type]

        old_current = current
        current_list = list(current)
        for axis in tied:
            current_list[axis] += step[axis]
            t_max_list[axis] += t_delta[axis]
        current = tuple(current_list)
        previous_empty = old_current

        if not grid.in_extent(current):
            return None


def intersect_work_plane(
    origin_grid: Sequence[float],
    direction_grid: Sequence[float],
    axis: int | str,
    slice_index: int,
) -> VoxelCoord | None:
    """Intersect a forward ray with an axis-aligned voxel work plane.

    ``axis`` accepts 0/1/2 or X/Y/Z (case-insensitive). The returned cell
    follows Python floor semantics, with its plane-axis coordinate fixed to
    ``slice_index``.
    """
    if isinstance(axis, str):
        try:
            axis_index = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
        except KeyError as exc:
            raise ValueError("axis must be X, Y, Z, 0, 1, or 2") from exc
    else:
        axis_index = int(axis)
    if axis_index not in (0, 1, 2):
        raise ValueError("axis must be X, Y, Z, 0, 1, or 2")
    if len(origin_grid) != 3 or len(direction_grid) != 3:
        raise ValueError("origin_grid and direction_grid must have three components")

    component = float(direction_grid[axis_index])
    if component == 0.0:
        return None
    t = (float(slice_index) - float(origin_grid[axis_index])) / component
    if t < 0.0 or not math.isfinite(t):
        return None

    point = [float(origin_grid[i]) + float(direction_grid[i]) * t for i in range(3)]
    cell = [math.floor(value) for value in point]
    cell[axis_index] = int(slice_index)
    return tuple(cell)  # type: ignore[return-value]


def intersect_exit_work_plane(
    origin_grid: Sequence[float],
    direction_grid: Sequence[float],
    extent_min: Sequence[int],
    extent_max_exclusive: Sequence[int],
) -> tuple[VoxelCoord, VoxelCoord] | None:
    """Intersect a forward ray with the interior work plane it EXITS through.

    Instead of pinning drawing to the Z=0 floor, this finds the bounding-box
    face the ray leaves through (its far-side exit face). That face is the
    wall whose interior surface fronts the viewer, so it is the correct
    drawable plane when looking into an empty volume. Faces seen from their
    backsides (entry faces) never win this test.

    Returns ``(cell, face_normal)`` where ``cell`` is the last grid cell
    inside the extent on that wall and ``face_normal`` is the OUTWARD normal
    of the wall face (matching the historical floor convention of
    ``(0, 0, -1)`` for the bottom plane). Returns ``None`` when the ray runs
    parallel outside the extent slab or has no forward intersection.
    """
    direction = _normalized(direction_grid)
    if direction is None:
        return None
    origin = tuple(float(value) for value in origin_grid)

    t_exit = math.inf
    exit_axis = -1
    exit_at_upper = False
    for axis in range(3):
        component = direction[axis]
        lower = float(extent_min[axis])
        upper = float(extent_max_exclusive[axis])
        if component == 0.0:
            if origin[axis] < lower or origin[axis] > upper:
                return None
            continue
        t_lo = (lower - origin[axis]) / component
        t_hi = (upper - origin[axis]) / component
        at_upper = True
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
            at_upper = False
        # t_hi is the FAR intersection (exit); the far plane sits at the
        # upper bound when travelling positively, at the lower otherwise.
        if t_hi < t_exit:
            t_exit = t_hi
            exit_axis = axis
            exit_at_upper = at_upper

    if exit_axis < 0:
        return None

    point = tuple(origin[i] + direction[i] * t_exit for i in range(3))
    cell = [
        int(min(max(math.floor(point[a]), int(extent_min[a])), int(extent_max_exclusive[a]) - 1))
        for a in range(3)
    ]
    if exit_at_upper:
        cell[exit_axis] = int(extent_max_exclusive[exit_axis]) - 1
    else:
        cell[exit_axis] = int(extent_min[exit_axis])

    face_normal = [0, 0, 0]
    face_normal[exit_axis] = 1 if exit_at_upper else -1
    return tuple(cell), tuple(face_normal)  # type: ignore[return-value]
