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
_pending_preview_ptrs: set = set()
_preview_timer_registered = False

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
    global _preview_collection, _DYNAMIC_SWATCH_CACHE, _preview_timer_registered
    if bpy is not None and _preview_collection is not None:
        bpy.utils.previews.remove(_preview_collection)
    _preview_collection = None
    _DYNAMIC_SWATCH_CACHE.clear()
    _pending_preview_ptrs.clear()
    _preview_timer_registered = False


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


def _ui_panel_rgba() -> Tuple[int, int, int, int]:
    """Approximate the N-panel button background so preview holes are not black."""
    try:
        theme = bpy.context.preferences.themes[0]
        inner = theme.user_interface.wcol_regular.inner
        return (
            _to_display_byte(inner[0]),
            _to_display_byte(inner[1]),
            _to_display_byte(inner[2]),
            255,
        )
    except Exception:
        return (72, 72, 72, 255)


def _composite_preview_on_ui(arr: Any, size: int) -> Any:
    """Sit the material sphere on the UI color; knock out the black preview backdrop."""
    bg = np.array(_ui_panel_rgba(), dtype=np.uint8)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[:, :] = bg
    src = arr.astype(np.float32)
    corners = np.stack(
        [src[0, 0, :3], src[0, -1, :3], src[-1, 0, :3], src[-1, -1, :3]]
    )
    key = np.median(corners, axis=0)
    dist = np.linalg.norm(src[:, :, :3] - key.astype(np.float32), axis=2)
    src_a = src[:, :, 3] / 255.0
    corner_alpha = float(src[[0, 0, -1, -1], [0, -1, 0, -1], 3].mean())
    if corner_alpha < 80.0:
        keep = src_a > 0.12
    else:
        keep = (dist > 24.0) & (src_a > 0.08)
    mask = keep.astype(np.float32)[..., None]
    blended = out.astype(np.float32) * (1.0 - mask) + src * mask
    blended[:, :, 3] = 255.0
    return blended.astype(np.uint8)


def _material_preview_rgba(material: Any, size: int) -> Optional[Any]:
    """Copy Blender's generated material preview into a size×size RGBA byte image."""
    if material is None:
        return None
    preview = None
    try:
        ensure = getattr(material, "preview_ensure", None)
        preview = ensure() if callable(ensure) else getattr(material, "preview", None)
    except Exception:
        preview = getattr(material, "preview", None)
    if preview is None:
        return None

    pixels = None
    width = height = 0
    for size_attr, pix_attr in (
        ("icon_size", "icon_pixels_float"),
        ("image_size", "image_pixels_float"),
    ):
        dims = getattr(preview, size_attr, None)
        pix = getattr(preview, pix_attr, None)
        if dims is None or not pix:
            continue
        try:
            width, height = int(dims[0]), int(dims[1])
        except Exception:
            continue
        if width < 2 or height < 2:
            continue
        pixels = pix
        break
    if pixels is None:
        return None

    arr = np.asarray(pixels, dtype=np.float32)
    if arr.size != width * height * 4:
        return None
    arr = arr.reshape((height, width, 4))
    if float(arr[:, :, :3].max()) < 0.02:
        return None
    arr = arr[::-1]
    if (height, width) != (size, size):
        rows = (np.arange(size) * height / size).astype(np.int32)
        cols = (np.arange(size) * width / size).astype(np.int32)
        arr = arr[rows][:, cols]
    rgba = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _composite_preview_on_ui(rgba, size)


def _flush_material_previews() -> None:
    """Generate missing material previews off the UI draw path, then redraw."""
    global _preview_timer_registered
    _preview_timer_registered = False
    if bpy is None:
        return None
    try:
        bpy.ops.wm.previews_ensure()
    except Exception:
        pass
    try:
        wm = bpy.context.window_manager
        if wm is not None:
            for window in wm.windows:
                screen = getattr(window, "screen", None)
                if screen is None:
                    continue
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
    except Exception:
        pass
    _pending_preview_ptrs.clear()
    return None


def request_material_preview(material: Any) -> None:
    """Queue Blender's native material-preview render (not from panel draw)."""
    global _preview_timer_registered
    if material is None or bpy is None:
        return
    try:
        _pending_preview_ptrs.add(int(material.as_pointer()))
    except Exception:
        return
    if _preview_timer_registered:
        return
    try:
        bpy.app.timers.register(_flush_material_previews, first_interval=0.05)
        _preview_timer_registered = True
    except Exception:
        _preview_timer_registered = False


def generate_swatch_icon_id(
    color_rgba: Tuple[float, float, float, float],
    is_active: bool = False,
    is_used: bool = False,
    size: int = 32,
    material: Any = None,
) -> int:
    """Generate or retrieve a cached square swatch preview icon.

    Unused entries stay a flat display color. After the index is placed,
    the bound material's generated preview is composited on the UI color
    and a lower-right black triangle marks that it is in use.
    """
    if bpy is None or _preview_collection is None:
        return 0

    r_q = int(round(color_rgba[0] * 255))
    g_q = int(round(color_rgba[1] * 255))
    b_q = int(round(color_rgba[2] * 255))
    a_q = int(round(color_rgba[3] * 255))

    mat_preview = None
    if is_used and material is not None:
        request_material_preview(material)
        mat_preview = _material_preview_rgba(material, size)

    mat_key = "flat"
    if mat_preview is not None:
        mat_key = f"mat_{int(material.as_pointer())}_{int(mat_preview.sum())}"
    cache_key = f"swatch_{r_q}_{g_q}_{b_q}_{a_q}_{int(is_active)}_{int(is_used)}_{mat_key}"
    if cache_key in _preview_collection:
        return _preview_collection[cache_key].icon_id
    legacy_key = f"swatch_{r_q}_{g_q}_{b_q}_{a_q}_{int(is_active)}_{int(is_used)}"
    if material is None and legacy_key in _preview_collection:
        return _preview_collection[legacy_key].icon_id

    if mat_preview is not None:
        img_data = mat_preview.copy()
    else:
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
        border_color = [255, 230, 40, 255]
        border_inner = [255, 255, 255, 255]
        img_data[0:2, :, :] = border_color
        img_data[-2:, :, :] = border_color
        img_data[:, 0:2, :] = border_color
        img_data[:, -2:, :] = border_color
        img_data[2, 2:-2, :] = border_inner
        img_data[-3, 2:-2, :] = border_inner
        img_data[2:-2, 2, :] = border_inner
        img_data[2:-2, -3, :] = border_inner

    if is_used:
        # icon_pixels row 0 is the displayed bottom. Bottom-right triangle.
        tri = max(5, size // 5)
        black = np.array([10, 10, 10, 255], dtype=np.uint8)
        outline = np.array([240, 240, 240, 220], dtype=np.uint8)
        for y in range(tri):
            width = y + 1
            img_data[y, size - width :] = black
            left = size - width - 1
            if left >= 0:
                img_data[y, left] = outline

    pending_preview = bool(is_used and material is not None and mat_preview is None)
    store_key = legacy_key if material is None else cache_key
    if pending_preview:
        store_key = f"{legacy_key}_pending"
        if store_key in _preview_collection:
            preview = _preview_collection[store_key]
            preview.icon_size = (size, size)
            preview.is_icon_custom = True
            preview.icon_pixels_float = (img_data.astype(np.float32) / 255.0).reshape(-1).tolist()
            return preview.icon_id if preview.icon_id != 0 else 0

    preview = _preview_collection.new(store_key)
    preview.icon_size = (size, size)
    preview.is_icon_custom = True
    float_pixels = (img_data.astype(np.float32) / 255.0).reshape(-1)
    preview.icon_pixels_float = float_pixels.tolist()
    return preview.icon_id if preview.icon_id != 0 else 0
