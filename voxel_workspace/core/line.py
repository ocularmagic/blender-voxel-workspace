"""3D integer line interpolation and brush target picking helpers."""
from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np

from ..constants import VoxelCoord
from .grid import VoxelGrid
from .dda import trace_grid, intersect_work_plane


def clamp_to_extent(grid: VoxelGrid, coord: VoxelCoord) -> VoxelCoord:
    """Clamp a voxel coordinate to the nearest valid cell in the volume."""
    return tuple(
        max(grid.extent_min[axis], min(grid.extent_max_exclusive[axis] - 1, coord[axis]))
        for axis in range(3)
    )  # type: ignore[return-value]


def line_3d(
    start: VoxelCoord,
    end: VoxelCoord,
) -> list[VoxelCoord]:
    """Generate an ordered list of 3D integer coordinates connecting start and end (inclusive)
    with no holes (Chebyshev step <= 1)."""
    x0, y0, z0 = int(start[0]), int(start[1]), int(start[2])
    x1, y1, z1 = int(end[0]), int(end[1]), int(end[2])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = 1 if x1 > x0 else -1
    ys = 1 if y1 > y0 else -1
    zs = 1 if z1 > z0 else -1

    points: list[VoxelCoord] = [(x0, y0, z0)]
    if (x0, y0, z0) == (x1, y1, z1):
        return points

    # Driving axis
    if dx >= dy and dx >= dz:
        # Driving axis is X
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        x, y, z = x0, y0, z0
        while x != x1:
            x += xs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x, y, z))
    elif dy >= dx and dy >= dz:
        # Driving axis is Y
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        x, y, z = x0, y0, z0
        while y != y1:
            y += ys
            if p1 >= 0:
                x += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x, y, z))
    else:
        # Driving axis is Z
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        x, y, z = x0, y0, z0
        while z != z1:
            z += zs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x, y, z))

    return points


def compute_brush_target(
    grid: VoxelGrid,
    origin_grid: Sequence[float],
    direction_grid: Sequence[float],
    mode: str = "PLACE",
    max_distance: float = 1000.0,
) -> tuple[VoxelCoord | None, VoxelCoord | None, VoxelCoord | None]:
    """Calculate the target voxel coordinate and hover face normal for Place/Erase.
    
    Returns:
        (target_cell, hover_cell, hover_normal)
    """
    mode_upper = mode.upper()
    hit = trace_grid(grid, origin_grid, direction_grid, max_distance=max_distance)
    if hit is not None:
        if mode_upper == "PLACE":
            target = (
                hit.cell[0] + hit.normal[0],
                hit.cell[1] + hit.normal[1],
                hit.cell[2] + hit.normal[2],
            )
            if not grid.in_extent(target):
                target = clamp_to_extent(grid, target)
            return target, hit.cell, hit.normal
        elif mode_upper == "ERASE":
            return hit.cell, hit.cell, hit.normal

    # Miss: for PLACE, intersect with local Z=0 work plane
    if mode_upper == "PLACE":
        cell = intersect_work_plane(origin_grid, direction_grid, axis=2, slice_index=0)
        if cell is not None:
            if not grid.in_extent(cell):
                cell = clamp_to_extent(grid, cell)
            return cell, cell, (0, 0, 1)

    return None, None, None


def world_ray_to_grid_ray(
    origin_world: Sequence[float],
    direction_world: Sequence[float],
    matrix_world: Any,
    voxel_size: float = 1.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a world-space ray (origin and direction) to grid-space coordinates.
    
    matrix_world: 4x4 matrix (mathutils.Matrix or 4x4 array-like).
    voxel_size: world-space scale of a single voxel.
    """
    if hasattr(matrix_world, "inverted"):
        # Blender mathutils.Matrix
        inv_mat = matrix_world.inverted()
        import mathutils
        ow = mathutils.Vector((float(origin_world[0]), float(origin_world[1]), float(origin_world[2])))
        dw = mathutils.Vector((float(direction_world[0]), float(direction_world[1]), float(direction_world[2])))
        ol = inv_mat @ ow
        # Direction transform: 3x3 rotation/scale inverted
        dl = inv_mat.to_3x3() @ dw
        dl_len = dl.length
        if dl_len > 1e-9:
            dl = dl / dl_len
        scale = 1.0 / max(float(voxel_size), 1e-9)
        og = (float(ol.x) * scale, float(ol.y) * scale, float(ol.z) * scale)
        dg = (float(dl.x), float(dl.y), float(dl.z))
        return og, dg
    else:
        # NumPy or nested list 4x4
        mat = np.array(matrix_world, dtype=np.float64)
        inv_mat = np.linalg.inv(mat)
        ow = np.array([origin_world[0], origin_world[1], origin_world[2], 1.0], dtype=np.float64)
        ol = inv_mat @ ow
        dw = np.array([direction_world[0], direction_world[1], direction_world[2], 0.0], dtype=np.float64)
        dl = inv_mat @ dw
        dl_vec = dl[:3]
        dl_len = np.linalg.norm(dl_vec)
        if dl_len > 1e-9:
            dl_vec = dl_vec / dl_len
        scale = 1.0 / max(float(voxel_size), 1e-9)
        og = (float(ol[0]) * scale, float(ol[1]) * scale, float(ol[2]) * scale)
        dg = (float(dl_vec[0]), float(dl_vec[1]), float(dl_vec[2]))
        return og, dg
