"""Uniform contain-fit of a source AABB into a padded voxel volume."""
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

VoxelCoord = Tuple[int, int, int]


@dataclass(frozen=True)
class FitResult:
    """Uniform contain-fit from source voxel-space AABB into a target volume."""

    scale: float
    translation: Tuple[float, float, float]
    source_min: Tuple[float, float, float]
    source_max: Tuple[float, float, float]
    usable_min: Tuple[float, float, float]
    usable_max: Tuple[float, float, float]
    fitted_min: Tuple[float, float, float]
    fitted_max: Tuple[float, float, float]
    utilization: Tuple[float, float, float]


def contain_fit(
    source_min: Sequence[float],
    source_max: Sequence[float],
    extent_min: VoxelCoord,
    extent_max_exclusive: VoxelCoord,
    padding: int = 1,
) -> FitResult:
    """Compute a uniform scale + translation that contains the source AABB.

    Coordinates are in voxel-index space (cell i occupies [i, i+1)).
    Aspect ratio is preserved. Default alignment centers X/Y and rests on
    minimum Z (volume floor) inside the padded usable region.

    Raises ValueError for degenerate source bounds or a padding that leaves
    no usable interior.
    """
    smin = np.asarray(source_min, dtype=np.float64).reshape(3)
    smax = np.asarray(source_max, dtype=np.float64).reshape(3)
    src_size = smax - smin
    if not np.all(np.isfinite(smin)) or not np.all(np.isfinite(smax)):
        raise ValueError("Source bounds are not finite")
    if np.any(src_size < 0.0):
        raise ValueError("Source max must be >= source min")
    if np.all(src_size <= 1e-12):
        raise ValueError("Source mesh has degenerate (zero) bounds")

    emin = np.asarray(extent_min, dtype=np.float64)
    emax = np.asarray(extent_max_exclusive, dtype=np.float64)
    pad = max(0, int(padding))
    usable_min = emin + pad
    usable_max = emax - pad
    usable = usable_max - usable_min
    if np.any(usable <= 1e-9):
        raise ValueError(
            f"Padding {pad} leaves no usable interior in extent "
            f"[{tuple(int(v) for v in emin)}, {tuple(int(v) for v in emax)})"
        )

    ratios = []
    for i in range(3):
        if src_size[i] > 1e-12:
            ratios.append(usable[i] / src_size[i])
    if not ratios:
        raise ValueError("Source mesh has degenerate (zero) bounds")
    scale = float(min(ratios))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Contain-fit produced a non-positive scale")

    scaled = src_size * scale
    # Center X/Y; rest on minimum Z.
    offset = np.empty(3, dtype=np.float64)
    offset[0] = usable_min[0] + 0.5 * (usable[0] - scaled[0])
    offset[1] = usable_min[1] + 0.5 * (usable[1] - scaled[1])
    offset[2] = usable_min[2]
    translation = offset - smin * scale

    fitted_min = smin * scale + translation
    fitted_max = smax * scale + translation
    utilization = tuple(
        float(scaled[i] / usable[i]) if usable[i] > 0 else 0.0 for i in range(3)
    )
    return FitResult(
        scale=scale,
        translation=(float(translation[0]), float(translation[1]), float(translation[2])),
        source_min=(float(smin[0]), float(smin[1]), float(smin[2])),
        source_max=(float(smax[0]), float(smax[1]), float(smax[2])),
        usable_min=(float(usable_min[0]), float(usable_min[1]), float(usable_min[2])),
        usable_max=(float(usable_max[0]), float(usable_max[1]), float(usable_max[2])),
        fitted_min=(float(fitted_min[0]), float(fitted_min[1]), float(fitted_min[2])),
        fitted_max=(float(fitted_max[0]), float(fitted_max[1]), float(fitted_max[2])),
        utilization=utilization,
    )


def apply_fit(points: np.ndarray, fit: FitResult) -> np.ndarray:
    """Apply uniform contain-fit to an (N, 3) or (..., 3) point array."""
    arr = np.asarray(points, dtype=np.float64)
    t = np.asarray(fit.translation, dtype=np.float64)
    return arr * fit.scale + t
