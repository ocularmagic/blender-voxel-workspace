"""Custom preview icons for the compact built-in voxel palette."""
from pathlib import Path
from typing import Any

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


def register_palette_icons() -> None:
    """Load packaged color swatches once for EnumProperty icon buttons."""
    global _preview_collection
    if bpy is None or _preview_collection is not None:
        return
    previews = bpy.utils.previews.new()
    assets = Path(__file__).resolve().parent.parent / "assets"
    for index in PALETTE_NAMES:
        previews.load(
            f"palette_{index}",
            str(assets / f"palette_{index}.png"),
            "IMAGE",
        )
    _preview_collection = previews


def unregister_palette_icons() -> None:
    global _preview_collection
    if bpy is not None and _preview_collection is not None:
        bpy.utils.previews.remove(_preview_collection)
    _preview_collection = None


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
