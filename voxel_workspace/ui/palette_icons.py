"""Custom dynamic preview icons for voxel palette swatches with active/used indicators."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import bpy
    import bpy.utils.previews
except ImportError:
    bpy = None

PALETTE_NAMES = {
    1: "Neutral Gray",
    2: "Red",
    3: "Green",
    4: "Blue",
    5: "Yellow",
    6: "Magenta",
    7: "Cyan",
    8: "Orange",
}

_preview_collection: Any = None
_DYNAMIC_SWATCH_CACHE: Dict[str, Any] = {}


def register_palette_icons() -> None:
    """Load packaged color swatches once for EnumProperty icon buttons."""
    global _preview_collection
    if bpy is None or _preview_collection is not None:
        return
    previews = bpy.utils.previews.new()
    assets = Path(__file__).resolve().parent.parent / "assets"
    for index in PALETTE_NAMES:
        icon_path = assets / f"palette_{index}.png"
        if icon_path.exists():
            previews.load(
                f"palette_{index}",
                str(icon_path),
                "IMAGE",
            )
    _preview_collection = previews


def unregister_palette_icons() -> None:
    global _preview_collection, _DYNAMIC_SWATCH_CACHE
    if bpy is not None and _preview_collection is not None:
        bpy.utils.previews.remove(_preview_collection)
    _preview_collection = None
    _DYNAMIC_SWATCH_CACHE.clear()


def palette_enum_items(_self=None, _context=None):
    """Return stable icon-backed enum items for indices 1–8."""
    items = []
    for index, name in PALETTE_NAMES.items():
        icon_id = 0
        if _preview_collection is not None:
            preview = _preview_collection.get(f"palette_{index}")
            if preview is not None:
                icon_id = preview.icon_id
        items.append(
            (
                str(index),
                name,
                f"Place voxels using {name} (palette index {index})",
                icon_id,
                index,
            )
        )
    return items


def generate_swatch_icon_id(
    color_rgba: Tuple[float, float, float, float],
    is_active: bool = False,
    is_used: bool = False,
    size: int = 32,
) -> int:
    """Generate or retrieve a cached square swatch preview icon.
    
    - Center filled with swatch color
    - High-contrast bright highlight border when is_active=True (e.g. bright cyan/white double border)
    - Small black dot in the center when is_used=True
    """
    if bpy is None or _preview_collection is None:
        return 0

    # Quantize color to avoid cache explosion
    r_q = int(round(color_rgba[0] * 255))
    g_q = int(round(color_rgba[1] * 255))
    b_q = int(round(color_rgba[2] * 255))
    a_q = int(round(color_rgba[3] * 255))

    cache_key = f"swatch_{r_q}_{g_q}_{b_q}_{a_q}_{int(is_active)}_{int(is_used)}"
    if cache_key in _preview_collection:
        return _preview_collection[cache_key].icon_id

    # Create pixel buffer (size x size x 4)
    # Convert linear float input to sRGB/display byte representation
    img_data = np.zeros((size, size, 4), dtype=np.uint8)

    # Fill base swatch color (convert linear approx to byte)
    def to_byte(val: float) -> int:
        clamped = max(0.0, min(1.0, float(val)))
        # Standard gamma conversion
        if clamped <= 0.0031308:
            srgb = clamped * 12.92
        else:
            srgb = 1.055 * (clamped ** (1.0 / 2.4)) - 0.055
        return int(round(max(0.0, min(1.0, srgb)) * 255))

    sr = to_byte(color_rgba[0])
    sg = to_byte(color_rgba[1])
    sb = to_byte(color_rgba[2])
    sa = int(round(max(0.0, min(1.0, float(color_rgba[3]))) * 255))

    # Base fill
    img_data[:, :, 0] = sr
    img_data[:, :, 1] = sg
    img_data[:, :, 2] = sb
    img_data[:, :, 3] = sa

    # Subtle inner dark border for contrast
    img_data[0, :, :3] = 40
    img_data[-1, :, :3] = 40
    img_data[:, 0, :3] = 40
    img_data[:, -1, :3] = 40

    # Active highlight border: high-contrast bright yellow/white/cyan 3px border
    if is_active:
        # Outer bright border (3 pixels wide)
        border_color = [255, 230, 40, 255] # High-contrast bright yellow
        border_inner = [255, 255, 255, 255] # Inner bright white
        
        # Ring 0 and 1: yellow
        img_data[0:2, :, :] = border_color
        img_data[-2:, :, :] = border_color
        img_data[:, 0:2, :] = border_color
        img_data[:, -2:, :] = border_color
        
        # Ring 2: white
        img_data[2, 2:-2, :] = border_inner
        img_data[-3, 2:-2, :] = border_inner
        img_data[2:-2, 2, :] = border_inner
        img_data[2:-2, -3, :] = border_inner

    # Used dot: small black dot in the center of the swatch
    if is_used:
        center = size // 2
        radius = max(2, size // 8) # 3-4 pixel radius dot
        y, x = np.ogrid[:size, :size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        # Black center dot with subtle white anti-aliased edge
        inner_mask = dist_sq <= (radius ** 2)
        outer_mask = (dist_sq <= ((radius + 1) ** 2)) & ~inner_mask
        
        # Solid black center
        img_data[inner_mask] = [10, 10, 10, 255]
        # Subtle light outline around dot for visibility against dark swatches
        if sr + sg + sb < 180: # Dark background swatch
            img_data[outer_mask] = [240, 240, 240, 220]

    # Create dynamic ImagePreview
    preview = _preview_collection.new(cache_key)
    preview.icon_size = (size, size)
    preview.is_icon_custom = True
    # icon_pixels_float expects float array of size*size*4 in 0..1 range
    float_pixels = (img_data.astype(np.float32) / 255.0).reshape(-1)
    preview.icon_pixels_float = float_pixels.tolist()

    # Note: in headless background mode without a window manager/OpenGL context,
    # icon_id is 0 until rendered by a UI pass, but the preview datablock is populated.
    return preview.icon_id if preview.icon_id != 0 else 0

