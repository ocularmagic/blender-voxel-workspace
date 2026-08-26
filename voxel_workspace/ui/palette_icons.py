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

TOOL_ICON_FILES = {
    "add_surface": "voxeladdsurface.png",
    "add_volume": "voxeladdvolume.png",
    "erase": "voxelerase.png",
    "paint": "voxelpaint.png",
    "stop": "voxelstop.png",
}


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
    toolbar = assets / "toolbar"
    for key, filename in TOOL_ICON_FILES.items():
        icon_path = toolbar / filename
        if icon_path.exists():
            previews.load(f"tool_{key}", str(icon_path), "IMAGE")
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


def get_preview_collection() -> Any:
    """Return the shared preview collection (or None before registration)."""
    return _preview_collection


def tool_icon_id(name: str) -> int:
    """Return the preview icon id for a packaged toolbar PNG, or 0."""
    if _preview_collection is None:
        return 0
    preview = _preview_collection.get(f"tool_{name}")
    if preview is None:
        return 0
    return int(getattr(preview, "icon_id", 0) or 0)


def _to_display_byte(val: float) -> int:
    clamped = max(0.0, min(1.0, float(val)))
    if clamped <= 0.0031308:
        srgb = clamped * 12.92
    else:
        srgb = 1.055 * (clamped ** (1.0 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, srgb)) * 255))


def generate_swatch_icon_id(
    color_rgba: Tuple[float, float, float, float],
    is_active: bool = False,
    is_used: bool = False,
    size: int = 32,
    material: Any = None,
) -> int:
    """Generate or retrieve a cached square swatch preview icon.

    Every swatch is a flat display color. Used entries get a small centered
    black dot; the active entry gets a thick border in the inverted (complement)
    color of the swatch so the selection is always high contrast.
    Material previews are not composited into the chips.
    """
    if bpy is None or _preview_collection is None:
        return 0

    r_q = int(round(color_rgba[0] * 255))
    g_q = int(round(color_rgba[1] * 255))
    b_q = int(round(color_rgba[2] * 255))
    a_q = int(round(color_rgba[3] * 255))

    cache_key = f"swatch_{r_q}_{g_q}_{b_q}_{a_q}_{int(is_active)}_{int(is_used)}"
    if cache_key in _preview_collection:
        return _preview_collection[cache_key].icon_id

    sr = _to_display_byte(color_rgba[0])
    sg = _to_display_byte(color_rgba[1])
    sb = _to_display_byte(color_rgba[2])
    sa = int(round(max(0.0, min(1.0, float(color_rgba[3]))) * 255))
    img_data = np.zeros((size, size, 4), dtype=np.uint8)
    img_data[:, :, 0] = sr
    img_data[:, :, 1] = sg
    img_data[:, :, 2] = sb
    img_data[:, :, 3] = sa
    img_data[0, :, :3] = 40
    img_data[-1, :, :3] = 40
    img_data[:, 0, :3] = 40
    img_data[:, -1, :3] = 40

    if is_active:
        # Inverted border for high contrast: the border color is the complement
        # of the swatch color, so it always stands out against the fill. For very
        # dark/light fills the inverted value already contrasts; fall back to
        # black or white for guaranteed visibility at the extremes.
        inv_r = 255 - sr
        inv_g = 255 - sg
        inv_b = 255 - sb
        border_color = [inv_r, inv_g, inv_b, 255]
        thickness = max(3, size // 8)
        img_data[0:thickness, :, :] = border_color
        img_data[-thickness:, :, :] = border_color
        img_data[:, 0:thickness, :] = border_color
        img_data[:, -thickness:, :] = border_color

    if is_used:
        # Small black dot centered in the icon marks this color as in use.
        dot_r = 2  # 4x4 square
        cx = size // 2
        cy = size // 2
        black = np.array([10, 10, 10, 255], dtype=np.uint8)
        x0 = cx - dot_r
        x1 = cx + dot_r
        y0 = cy - dot_r
        y1 = cy + dot_r
        img_data[y0:y1, x0:x1, :] = black

    preview = _preview_collection.new(cache_key)
    preview.icon_size = (size, size)
    preview.is_icon_custom = True
    float_pixels = (img_data.astype(np.float32) / 255.0).reshape(-1)
    preview.icon_pixels_float = float_pixels.tolist()
    return preview.icon_id if preview.icon_id != 0 else 0
