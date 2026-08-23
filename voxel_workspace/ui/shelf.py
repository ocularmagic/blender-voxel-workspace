"""Voxel tool Asset Shelf — Sculpting-style thumbnails on the bottom dock."""
from pathlib import Path
from typing import Any

try:
    import bpy
    from bpy.types import AssetShelf
except ImportError:
    bpy = None
    AssetShelf = object

from .panels import _in_voxel_workspace

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "toolbar"
_TOOL_PROP = "voxel_shelf_tool"
_SHELF_PREVIEW_SIZE = 48

SHELF_TOOLS = (
    ("ADD_SURFACE", "Voxel Add Surface", "voxeladdsurface.png"),
    ("ADD_VOLUME", "Voxel Add Volume", "voxeladdvolume.png"),
    ("REPAINT", "Voxel Repaint", "voxelpaint.png"),
    ("ERASE", "Voxel Erase", "voxelerase.png"),
    ("STOP", "Voxel Stop Editing", "voxelstop.png"),
)


def _apply_preview(id_data: Any, png_path: Path) -> None:
    if bpy is None or not png_path.is_file():
        return
    loader = bpy.data.images.load(str(png_path), check_existing=False)
    try:
        preview = id_data.preview_ensure()
        preview.image_size = [int(loader.size[0]), int(loader.size[1])]
        preview.image_pixels_float = list(loader.pixels)
    finally:
        bpy.data.images.remove(loader)


def ensure_tool_assets() -> None:
    """Create or refresh unlinked empty-object assets used as shelf tiles."""
    if bpy is None:
        return
    for tool_id, name, filename in SHELF_TOOLS:
        png_path = _ASSET_DIR / filename
        obj = bpy.data.objects.get(name)
        if obj is None:
            obj = bpy.data.objects.new(name, None)
        obj.use_fake_user = True
        obj.hide_viewport = True
        obj.hide_render = True
        if getattr(obj, "asset_data", None) is None:
            obj.asset_mark()
        obj[_TOOL_PROP] = tool_id
        if png_path.is_file():
            _apply_preview(obj, png_path)


class VOXEL_AST_workspace(AssetShelf):
    """Bottom Asset Shelf listing the four voxel edit tools."""

    bl_idname = "VIEW3D_AST_voxel_workspace"
    bl_space_type = "VIEW_3D"
    bl_options = {"DEFAULT_VISIBLE", "NO_ASSET_DRAG"}
    bl_activate_operator = "voxel.shelf_activate"
    bl_default_preview_size = _SHELF_PREVIEW_SIZE
    # All type filters default to False. Without this the shelf is empty.
    filter_object = True

    @classmethod
    def poll(cls, context: Any) -> bool:
        return _in_voxel_workspace(context)

    @classmethod
    def asset_poll(cls, asset: Any) -> bool:
        local = getattr(asset, "local_id", None)
        if local is not None:
            return bool(local.get(_TOOL_PROP))
        return getattr(asset, "id_type", "") == "OBJECT" and str(
            getattr(asset, "name", "")
        ).startswith("Voxel ")


SHELF_CLASSES = [VOXEL_AST_workspace]
