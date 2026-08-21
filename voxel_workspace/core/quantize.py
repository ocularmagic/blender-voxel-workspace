"""Color quantization service using deterministic frequency-weighted median-cut.

Used by:
1. GLB import to conform imported colors into a target palette size (1..255).
2. Image-to-voxel / texture sampling tools.
3. .vox exporter for quantizing non-destructive copies to MagicaVoxel's 255-color palette limit.
4. Palette reduction and preset conformance.

Design principles:
- Transparent threshold support (alpha cutoff).
- Accent preservation (variance and range weighting).
- Non-destructive (returns mapping dictionary and new palette without altering source buffers).
- Deterministic and dependency-free (pure Python and NumPy).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .presets import linear_to_srgb_byte, srgb_byte_to_linear


@dataclass
class QuantizedPaletteResult:
    """Result of color quantization."""
    palette: List[Tuple[float, float, float, float]]  # Linear RGBA tuples (1-indexed colors, 0 reserved)
    color_map: Dict[Tuple[int, int, int, int], int]  # Maps source sRGB/RGBA bytes to 1-based palette index
    remap_indices: Optional[np.ndarray] = None       # If input was array of color samples, mapped palette index per sample


def quantize_colors_median_cut(
    colors_rgba_linear: Union[np.ndarray, Sequence[Tuple[float, float, float, float]]],
    max_colors: int = 255,
    weights: Optional[Union[np.ndarray, Sequence[float]]] = None,
    alpha_threshold: float = 0.1,
) -> QuantizedPaletteResult:
    """Quantize an array of linear RGBA color samples into a compact palette of size <= max_colors.
    
    Parameters:
        colors_rgba_linear: (N, 4) array-like of float linear RGBA [0.0..1.0].
        max_colors: Maximum number of colors in resulting palette (1..4096).
        weights: Optional frequency weights for each sample.
        alpha_threshold: Samples with alpha < alpha_threshold are treated as empty (index 0).
        
    Returns:
        QuantizedPaletteResult containing:
        - palette: list of (R, G, B, A) linear floats (indices 1..K, length K+1 with index 0 empty)
        - color_map: dict mapping (r_byte, g_byte, b_byte, a_byte) -> palette_index
        - remap_indices: (N,) int32 array mapping each input sample to its palette index (0 for transparent)
    """
    if max_colors < 1:
        max_colors = 1

    arr = np.ascontiguousarray(colors_rgba_linear, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 4))
    n_samples = len(arr)

    if n_samples == 0:
        return QuantizedPaletteResult(
            palette=[(0.0, 0.0, 0.0, 0.0)],
            color_map={},
            remap_indices=np.empty(0, dtype=np.int32),
        )

    # 1. Filter out transparent samples
    alpha = arr[:, 3]
    valid_mask = (alpha >= alpha_threshold)
    remap_indices = np.zeros(n_samples, dtype=np.int32)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return QuantizedPaletteResult(
            palette=[(0.0, 0.0, 0.0, 0.0)],
            color_map={},
            remap_indices=remap_indices,
        )

    valid_colors = arr[valid_indices]
    
    # Quantize to sRGB bytes to group identical colors and collect frequencies
    def to_bytes(linear_arr: np.ndarray) -> np.ndarray:
        # Vectorized sRGB byte conversion
        clamped = np.clip(linear_arr, 0.0, 1.0)
        low = clamped <= 0.0031308
        srgb = np.where(low, clamped * 12.92, 1.055 * (clamped ** (1.0 / 2.4)) - 0.055)
        return np.round(srgb * 255.0).astype(np.uint8)

    srgb_bytes = to_bytes(valid_colors[:, :3])
    alpha_bytes = np.round(np.clip(valid_colors[:, 3], 0.0, 1.0) * 255.0).astype(np.uint8)
    packed_bytes = np.column_stack((srgb_bytes, alpha_bytes))

    # Unique colors and frequency counts
    unq_bytes, inverse_indices, counts = np.unique(
        packed_bytes,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    if weights is not None:
        w_arr = np.array(weights, dtype=np.float32)[valid_indices]
        # Sum weights per unique color
        unq_weights = np.zeros(len(unq_bytes), dtype=np.float32)
        np.add.at(unq_weights, inverse_indices, w_arr)
    else:
        unq_weights = counts.astype(np.float32)

    # Convert unique colors back to float linear RGB for median cut partitioning
    unq_linear = np.zeros((len(unq_bytes), 4), dtype=np.float32)
    # Inverse sRGB conversion
    c_norm = unq_bytes[:, :3].astype(np.float32) / 255.0
    low = c_norm <= 0.04045
    unq_linear[:, :3] = np.where(low, c_norm / 12.92, ((c_norm + 0.055) / 1.055) ** 2.4)
    unq_linear[:, 3] = unq_bytes[:, 3].astype(np.float32) / 255.0

    # 2. If unique colors already fit within max_colors, direct 1:1 mapping
    if len(unq_bytes) <= max_colors:
        palette = [(0.0, 0.0, 0.0, 0.0)]
        color_map: Dict[Tuple[int, int, int, int], int] = {}
        for i, col in enumerate(unq_linear, start=1):
            palette.append(tuple(col))
            b_tuple = (int(unq_bytes[i - 1, 0]), int(unq_bytes[i - 1, 1]), int(unq_bytes[i - 1, 2]), int(unq_bytes[i - 1, 3]))
            color_map[b_tuple] = i

        remap_indices[valid_indices] = inverse_indices + 1
        return QuantizedPaletteResult(
            palette=palette,
            color_map=color_map,
            remap_indices=remap_indices,
        )

    # 3. Frequency-weighted Median-Cut partitioning
    @dataclass(eq=False)
    class ColorBox:
        indices: np.ndarray  # indices into unq_linear

        @property
        def total_weight(self) -> float:
            return float(np.sum(unq_weights[self.indices]))

        @property
        def range_spread(self) -> float:
            if len(self.indices) <= 1:
                return 0.0
            box_colors = unq_linear[self.indices, :3]
            min_c = np.min(box_colors, axis=0)
            max_c = np.max(box_colors, axis=0)
            return float(np.max(max_c - min_c))

        @property
        def priority(self) -> float:
            # Balance population count and color range variance to preserve small distinct accents
            return self.total_weight * (self.range_spread + 0.01)

        def split(self) -> Tuple["ColorBox", "ColorBox"]:
            box_colors = unq_linear[self.indices, :3]
            min_c = np.min(box_colors, axis=0)
            max_c = np.max(box_colors, axis=0)
            spread = max_c - min_c
            split_axis = int(np.argmax(spread))

            # Sort box indices along the widest axis
            vals = box_colors[:, split_axis]
            sorted_order = np.argsort(vals)
            sorted_indices = self.indices[sorted_order]
            sorted_weights = unq_weights[sorted_indices]

            # Find weighted median split point
            half_weight = np.sum(sorted_weights) / 2.0
            cum_weight = np.cumsum(sorted_weights)
            split_point = np.searchsorted(cum_weight, half_weight)
            split_point = max(1, min(len(sorted_indices) - 1, int(split_point)))

            box1 = ColorBox(indices=sorted_indices[:split_point])
            box2 = ColorBox(indices=sorted_indices[split_point:])
            return box1, box2

        def representative_color(self) -> Tuple[float, float, float, float]:
            w = unq_weights[self.indices]
            total_w = np.sum(w)
            if total_w > 0:
                avg_rgb = np.sum(unq_linear[self.indices, :3] * w[:, None], axis=0) / total_w
                avg_a = np.sum(unq_linear[self.indices, 3] * w) / total_w
            else:
                avg_rgb = np.mean(unq_linear[self.indices, :3], axis=0)
                avg_a = float(np.mean(unq_linear[self.indices, 3]))
            return (float(avg_rgb[0]), float(avg_rgb[1]), float(avg_rgb[2]), float(avg_a))

    boxes: List[ColorBox] = [ColorBox(indices=np.arange(len(unq_bytes)))]

    while len(boxes) < max_colors:
        # Find splittable box with highest priority
        splittable_idx = [i for i, b in enumerate(boxes) if len(b.indices) > 1]
        if not splittable_idx:
            break
        # Pick index of box with highest priority
        best_i = max(splittable_idx, key=lambda i: boxes[i].priority)
        box_to_split = boxes.pop(best_i)
        b1, b2 = box_to_split.split()
        boxes.append(b1)
        boxes.append(b2)

    # 4. Assemble palette and build color map
    palette = [(0.0, 0.0, 0.0, 0.0)] # 0 is empty
    unq_to_pal_idx = np.zeros(len(unq_bytes), dtype=np.int32)
    color_map = {}

    for pal_idx, box in enumerate(boxes, start=1):
        rep_col = box.representative_color()
        palette.append(rep_col)
        unq_to_pal_idx[box.indices] = pal_idx
        for u_idx in box.indices:
            b_tuple = (int(unq_bytes[u_idx, 0]), int(unq_bytes[u_idx, 1]), int(unq_bytes[u_idx, 2]), int(unq_bytes[u_idx, 3]))
            color_map[b_tuple] = pal_idx

    remap_indices[valid_indices] = unq_to_pal_idx[inverse_indices]

    return QuantizedPaletteResult(
        palette=palette,
        color_map=color_map,
        remap_indices=remap_indices,
    )


def quantize_palette_to_limit(
    palette: Sequence[Tuple[float, float, float, float]],
    max_colors: int = 255,
    counts: Optional[Dict[int, int]] = None,
) -> Tuple[List[Tuple[float, float, float, float]], Dict[int, int]]:
    """Quantize an existing indexed palette (e.g. >255 colors) down to max_colors.
    
    Used by .vox export to build a non-destructive remap table without mutating the source volume.
    
    Parameters:
        palette: Existing palette entries (index 0 empty, 1..N colors).
        max_colors: Target color limit (e.g. 255).
        counts: Optional voxel usage count per palette index to weight quantization.
        
    Returns:
        (quantized_palette, remap_table: {old_index: new_index})
    """
    valid_old_indices = [idx for idx in range(1, len(palette)) if idx < len(palette)]
    if len(valid_old_indices) <= max_colors:
        # Palette already fits within limit
        remap_table = {idx: idx for idx in valid_old_indices}
        return list(palette), remap_table

    colors = [palette[idx] for idx in valid_old_indices]
    weights = [counts.get(idx, 1) if counts else 1 for idx in valid_old_indices]

    res = quantize_colors_median_cut(
        colors_rgba_linear=colors,
        max_colors=max_colors,
        weights=weights,
        alpha_threshold=0.01,
    )

    remap_table: Dict[int, int] = {}
    for pos, old_idx in enumerate(valid_old_indices):
        remap_table[old_idx] = int(res.remap_indices[pos]) if res.remap_indices is not None and len(res.remap_indices) > pos else 1

    return res.palette, remap_table
