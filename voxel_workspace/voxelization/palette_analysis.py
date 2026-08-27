"""Perceptual Draft/Balanced/Fine palette-size recommendations."""
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.quantize import quantize_colors_median_cut


@dataclass(frozen=True)
class PaletteRecommendations:
    draft: int
    balanced: int
    fine: int
    sample_count: int
    unique_color_count: int


def _linear_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    l = 0.4122214708 * values[:, 0] + 0.5363325363 * values[:, 1] + 0.0514459929 * values[:, 2]
    m = 0.2119034982 * values[:, 0] + 0.6806995451 * values[:, 1] + 0.1073969566 * values[:, 2]
    s = 0.0883024619 * values[:, 0] + 0.2817188376 * values[:, 1] + 0.6299787005 * values[:, 2]
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.column_stack((
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ))


def _perceptual_error(colors: np.ndarray, maximum: int, alpha_cutoff: float) -> float:
    result = quantize_colors_median_cut(colors, max_colors=int(maximum), alpha_threshold=float(alpha_cutoff))
    remap = result.remap_indices
    if remap is None or not len(remap):
        return 0.0
    valid = remap > 0
    if not np.any(valid):
        return 0.0
    palette = np.asarray(result.palette, dtype=np.float32)
    source_lab = _linear_rgb_to_oklab(colors[valid, :3])
    mapped_lab = _linear_rgb_to_oklab(palette[remap[valid], :3])
    distances = np.linalg.norm(source_lab - mapped_lab, axis=1)
    return float(np.percentile(distances, 95.0))


def recommend_palette_sizes(
    colors: Sequence[Sequence[float]] | np.ndarray,
    alpha_cutoff: float = 0.1,
) -> PaletteRecommendations:
    """Choose independent palette limits from 95th-percentile OKLab error."""
    samples = np.asarray(colors, dtype=np.float32).reshape((-1, 4))
    if len(samples) == 0:
        samples = np.asarray([(0.8, 0.8, 0.8, 1.0)], dtype=np.float32)
    valid = samples[:, 3] >= float(alpha_cutoff)
    visible = samples[valid]
    if len(visible) == 0:
        visible = samples
    # Work in the same 8-bit sRGB-scale distinct-color regime as the importer.
    clamped = np.clip(visible, 0.0, 1.0)
    low = clamped[:, :3] <= 0.0031308
    srgb = np.where(low, clamped[:, :3] * 12.92, 1.055 * clamped[:, :3] ** (1.0 / 2.4) - 0.055)
    packed = np.column_stack((np.round(srgb * 255.0), np.round(clamped[:, 3] * 255.0))).astype(np.uint8)
    unique_count = int(len(np.unique(packed, axis=0)))
    if unique_count <= 1:
        return PaletteRecommendations(1, 1, 1, len(visible), unique_count)

    candidates = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 255]
    candidates = [min(value, unique_count, 255) for value in candidates]
    candidates = sorted(set(max(1, value) for value in candidates))
    errors = {candidate: _perceptual_error(visible, candidate, alpha_cutoff) for candidate in candidates}

    def first_at_or_below(threshold: float) -> int:
        return next((candidate for candidate in candidates if errors[candidate] <= threshold), candidates[-1])

    draft = first_at_or_below(0.075)
    balanced = max(draft, first_at_or_below(0.035))
    fine = max(balanced, first_at_or_below(0.015))
    return PaletteRecommendations(draft, balanced, fine, len(visible), unique_count)
