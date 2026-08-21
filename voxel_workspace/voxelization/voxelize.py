"""End-to-end voxelization: occupancy, color, quantization, grid fill."""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from ..core.grid import VoxelGrid
from ..core.quantize import quantize_colors_median_cut
from .color_sampling import (
    SampledMaterial,
    sample_occupied_nearest_surface,
)
from .occupancy import occupancy_shell, occupancy_solid, occupied_coords


VoxelCoord = Tuple[int, int, int]


@dataclass
class VoxelizeResult:
    grid: VoxelGrid
    palette: List[Tuple[float, float, float, float]]
    occupied_count: int
    palette_color_count: int
    warnings: List[str] = field(default_factory=list)


def voxelize_fitted_mesh(
    triangles: np.ndarray,
    extent_min: VoxelCoord,
    extent_max_exclusive: VoxelCoord,
    occupancy_mode: str = "SOLID",
    mesh_closed: bool = True,
    uvs: Optional[np.ndarray] = None,
    material_indices: Optional[np.ndarray] = None,
    materials: Optional[Sequence[SampledMaterial]] = None,
    palette_size: int = 64,
    alpha_cutoff: float = 0.1,
    brick_size: int = 32,
    warnings: Optional[List[str]] = None,
    occupy_min: Optional[VoxelCoord] = None,
    occupy_max: Optional[VoxelCoord] = None,
) -> VoxelizeResult:
    """Voxelize fitted triangles into a sparse VoxelGrid with a generated palette.

    `triangles` are in voxel-index space after contain-fit. Index 0 of the
    returned palette is empty; occupied cells map to 1..K.
    Occupancy is generated only inside [occupy_min, occupy_max) so padding
    around a contain-fit stays empty.
    """
    notes: List[str] = list(warnings or [])
    mode = occupancy_mode.upper()
    if mode not in ("SOLID", "SHELL"):
        raise ValueError(f"Unknown occupancy mode {occupancy_mode!r}")

    tris = np.asarray(triangles, dtype=np.float64)
    if tris.size == 0:
        raise ValueError("Import contains no evaluated triangles")

    occ_min = occupy_min if occupy_min is not None else extent_min
    occ_max = occupy_max if occupy_max is not None else extent_max_exclusive
    shell = occupancy_shell(tris, occ_min, occ_max)
    if not np.any(shell):
        raise ValueError("No voxels were occupied after fitting the mesh into the volume")

    mats: Sequence[SampledMaterial] = materials or [SampledMaterial(base_color=(0.8, 0.8, 0.8, 1.0))]
    mat_i = (
        np.asarray(material_indices, dtype=np.int32)
        if material_indices is not None
        else np.zeros(len(tris), dtype=np.int32)
    )

    occupied = shell
    if mode == "SOLID":
        if not mesh_closed:
            notes.append("Source mesh is open or non-manifold; solid fill fell back to shell occupancy")
        else:
            occupied = occupancy_solid(shell)

    colors = sample_occupied_nearest_surface(
        occupied,
        tris,
        uvs,
        mat_i,
        mats,
        occ_min,
    )

    if alpha_cutoff > 0.0:
        punched = occupied & (colors[..., 3] < float(alpha_cutoff))
        if np.any(punched):
            occupied = occupied.copy()
            occupied[punched] = False
            colors = colors.copy()
            colors[punched] = 0.0
            notes.append(
                f"Alpha cutoff {alpha_cutoff:g} cleared {int(np.count_nonzero(punched))} voxels"
            )
        if not np.any(occupied):
            raise ValueError("All occupied voxels were below the alpha cutoff")

    coords = occupied_coords(occupied, occ_min)
    if len(coords) == 0:
        raise ValueError("No voxels were occupied after conversion")

    samples = colors[occupied]
    max_colors = max(1, min(int(palette_size), 255))
    quantized = quantize_colors_median_cut(
        samples,
        max_colors=max_colors,
        alpha_threshold=float(alpha_cutoff),
    )
    remap = quantized.remap_indices
    if remap is None or len(remap) != len(coords):
        raise ValueError("Quantizer did not return a remap for every occupied voxel")

    grid = VoxelGrid(
        extent_min=extent_min,
        extent_max_exclusive=extent_max_exclusive,
        brick_size=brick_size,
    )
    for coord, pal_idx in zip(coords, remap):
        idx = int(pal_idx)
        if idx <= 0:
            continue
        if idx > 255:
            raise ValueError(f"Palette index {idx} outside 1..255")
        xyz = (int(coord[0]), int(coord[1]), int(coord[2]))
        if not grid.in_extent(xyz):
            raise ValueError(f"Occupied coordinate {xyz} lies outside the target extent")
        grid.set(xyz, idx)

    occupied_count = int(sum(1 for pal_idx in remap if int(pal_idx) > 0))
    if occupied_count == 0:
        raise ValueError("Quantization mapped every occupied voxel to empty")

    palette = [tuple(float(c) for c in col) for col in quantized.palette]
    return VoxelizeResult(
        grid=grid,
        palette=palette,
        occupied_count=occupied_count,
        palette_color_count=max(0, len(palette) - 1),
        warnings=notes,
    )
