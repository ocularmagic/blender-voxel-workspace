"""Aspect-aware volume sizing and exterior-detail resolution estimates."""
from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ResolutionRecommendations:
    draft: int
    balanced: int
    fine: int
    feature_scale: float
    triangle_count: int


def volume_dimensions(
    source_size: Sequence[float],
    maximum_axis: int,
    padding: int = 0,
) -> Tuple[int, int, int]:
    """Return aspect-preserving complete volume dimensions, including padding."""
    size = np.asarray(source_size, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size < 0.0):
        raise ValueError("Source dimensions must be finite and non-negative")
    longest = float(np.max(size))
    if longest <= 1e-12:
        raise ValueError("Source dimensions are degenerate")
    maximum = max(1, int(maximum_axis))
    pad = max(0, int(padding))
    usable = maximum - 2 * pad
    if usable < 1:
        raise ValueError("Maximum axis is too small for the selected padding")
    longest_axis = int(np.argmax(size))
    result = [max(1 + 2 * pad, int(math.ceil(float(v) / longest * usable)) + 2 * pad) for v in size]
    result[longest_axis] = maximum
    return (int(result[0]), int(result[1]), int(result[2]))


def _round_to_step(value: float, step: int = 8) -> int:
    return max(step, int(step * round(float(value) / step)))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    vals = values[order]
    w = weights[order]
    total = float(np.sum(w))
    if total <= 1e-20:
        return float(np.percentile(vals, percentile))
    cumulative = np.cumsum(w)
    index = int(np.searchsorted(cumulative, total * percentile / 100.0, side="left"))
    return float(vals[min(index, len(vals) - 1)])


def recommend_resolutions(triangles: np.ndarray) -> ResolutionRecommendations:
    """Estimate Draft/Balanced/Fine maxima from exterior geometric detail.

    The estimate combines triangle complexity with an area-weighted local feature
    scale. Area weighting prevents tiny bevel/noise triangles from forcing an
    unnecessarily huge uniform grid, while the upper presets leave room for thin
    features that matter to the silhouette.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
        raise ValueError("triangles must have shape (T, 3, 3)")
    points = tris.reshape((-1, 3))
    span = points.max(axis=0) - points.min(axis=0)
    longest = float(np.max(span))
    if longest <= 1e-12:
        raise ValueError("Source geometry has degenerate bounds")

    e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    max_edge = np.maximum(np.maximum(e0, e1), e2)
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = (double_area > 1e-16) & (max_edge > 1e-12)
    if not np.any(valid):
        raise ValueError("Source geometry contains no non-degenerate triangles")

    # Minimum triangle altitude is a conservative local exterior feature scale.
    altitude = double_area[valid] / max_edge[valid]
    weights = double_area[valid]
    feature_scale = _weighted_percentile(altitude, weights, 70.0) / longest

    tri_count = int(np.count_nonzero(valid))
    complexity_target = 40.0 + 5.5 * math.log2(max(2, tri_count))
    feature_target = 0.9 / max(feature_scale, 1.0 / 512.0)
    balanced = _round_to_step(max(complexity_target, min(feature_target, 192.0)))
    balanced = max(48, min(256, balanced))
    draft = max(32, _round_to_step(balanced * 0.67))
    fine = min(512, max(balanced + 16, _round_to_step(balanced * 1.5)))
    return ResolutionRecommendations(
        draft=draft,
        balanced=balanced,
        fine=fine,
        feature_scale=float(feature_scale),
        triangle_count=tri_count,
    )
