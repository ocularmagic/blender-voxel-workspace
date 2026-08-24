"""Exact per-voxel Surface geometry for vertex-colour OBJ export.

This module intentionally knows nothing about Blender.  The input grid is the
authoritative TaggedVoxelGrid and every visible Surface face is emitted as one
coloured inset plus four grey perimeter strips.  Vertices are never shared
between colour regions so OBJ colour interpolation cannot soften the line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core.tagged_grid import TaggedVoxelGrid, VoxelDomain
from .visible_faces import FACE_SPECS

GREY_EDGE_COLOR = (0.0, 0.0, 0.0)
# OBJ consumers commonly use Y-up coordinates while the voxel workspace is
# Z-up.  This is the right-handed Blender-Z-up -> OBJ-Y-up basis conversion:
# (x, y, z) becomes (x, z, -y).
OBJ_Y_UP_CONVERSION = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_EPSILON = 1.0e-8


@dataclass(frozen=True)
class LinedMesh:
    """Deterministic triangle mesh with one RGB colour per vertex."""

    vertices: tuple[tuple[float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]


def _transform_point(point: Sequence[float], transform: object | None) -> tuple[float, float, float]:
    if transform is None:
        return tuple(float(v) for v in point)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be a 4x4 matrix")
    homogeneous = matrix @ np.array([*point, 1.0], dtype=np.float64)
    if abs(float(homogeneous[3])) > _EPSILON:
        homogeneous = homogeneous / homogeneous[3]
    return tuple(float(v) for v in homogeneous[:3])


def iter_visible_surface_faces(grid: TaggedVoxelGrid) -> Iterable[tuple[tuple[int, int, int], int, int, int]]:
    """Yield ``(cell, palette_index, axis, sign)`` in stable brick/cell order."""
    for brick_coord in grid.sorted_brick_coords():
        brick = grid.bricks[brick_coord]
        bx, by, bz = brick_coord
        base = (bx * grid.brick_size, by * grid.brick_size, bz * grid.brick_size)
        for lx in range(grid.brick_size):
            for ly in range(grid.brick_size):
                for lz in range(grid.brick_size):
                    coord = (base[0] + lx, base[1] + ly, base[2] + lz)
                    if not grid.in_extent(coord) or int(brick.domains[lx, ly, lz]) != int(VoxelDomain.SURFACE):
                        continue
                    index = int(brick.indices[lx, ly, lz])
                    for axis, sign, _template in FACE_SPECS:
                        neighbor = list(coord)
                        neighbor[axis] += sign
                        if grid.get_domain(tuple(neighbor)) != VoxelDomain.SURFACE:
                            yield coord, index, axis, sign


def build_voxel_lined_mesh(
    grid: TaggedVoxelGrid,
    color_for_index: Callable[[int], Sequence[float]],
    voxel_size: float = 1.0,
    edge_width: float = 0.01,
    transform: object | None = None,
    edge_color: Sequence[float] = GREY_EDGE_COLOR,
) -> LinedMesh:
    """Build exact visible-face center/strip triangles from ``grid``.

    ``edge_width`` is the shared rendered-edge fraction of a voxel.  It is
    clamped just below half a voxel to keep all rectangles non-inverted.
    Coordinates are root-local when ``transform`` is omitted; Blender passes
    the Voxel Root world matrix followed by ``OBJ_Y_UP_CONVERSION`` for
    standalone OBJ output.
    """
    size = float(voxel_size)
    fraction = float(edge_width)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("edge_width must be finite and non-negative")
    if len(edge_color) < 3:
        raise ValueError("edge_color must contain RGB components")
    width = min(fraction * size, size * 0.5 - _EPSILON)
    if width < 0.0:
        raise ValueError("edge_width is too large for the voxel size")

    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_rectangle(points: Sequence[Sequence[float]], color: Sequence[float]) -> None:
        start = len(vertices)
        rgb = tuple(float(c) for c in color[:3])
        if len(rgb) != 3:
            raise ValueError("palette colors must contain RGB components")
        vertices.extend(_transform_point(point, transform) for point in points)
        colors.extend([rgb] * 4)
        faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))

    for coord, palette_index, axis, sign in iter_visible_surface_faces(grid):
        _axis, _sign, template = FACE_SPECS[(axis * 2) + (0 if sign > 0 else 1)]
        outer = (np.asarray(coord, dtype=np.float64) + template) * size
        center = outer.mean(axis=0)
        inner = outer.copy()
        for i in range(4):
            delta = inner[i] - center
            for plane_axis in range(3):
                if plane_axis != axis:
                    inner[i, plane_axis] -= np.sign(delta[plane_axis]) * width
        add_rectangle(inner, color_for_index(palette_index))
        for i in range(4):
            j = (i + 1) % 4
            add_rectangle((outer[i], outer[j], inner[j], inner[i]), edge_color)

    return LinedMesh(tuple(vertices), tuple(colors), tuple(faces))


def write_vertex_color_obj(
    path: str,
    mesh: LinedMesh,
    edge_width: float,
    transform_space: str = "WORLD",
) -> None:
    """Write the repository's vertex-colour OBJ contract without sidecars."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Voxel Workspace exact voxel-lined OBJ export",
        f"# edge_width={float(edge_width):.9g} voxel fraction",
        f"# coordinate_space={transform_space}",
    ]
    lines.extend(
        "v {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}".format(*point, *color)
        for point, color in zip(mesh.vertices, mesh.colors)
    )
    lines.extend("f {} {} {}".format(a + 1, b + 1, c + 1) for a, b, c in mesh.faces)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


__all__ = [
    "GREY_EDGE_COLOR",
    "OBJ_Y_UP_CONVERSION",
    "LinedMesh",
    "iter_visible_surface_faces",
    "build_voxel_lined_mesh",
    "write_vertex_color_obj",
]
