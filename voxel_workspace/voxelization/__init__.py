"""Mesh-to-voxel conversion (fit, occupancy, color sampling)."""
from .fit import FitResult, contain_fit
from .occupancy import occupancy_solid, occupancy_shell
from .voxelize import VoxelizeResult, voxelize_fitted_mesh

__all__ = [
    "FitResult",
    "contain_fit",
    "occupancy_shell",
    "occupancy_solid",
    "VoxelizeResult",
    "voxelize_fitted_mesh",
]
