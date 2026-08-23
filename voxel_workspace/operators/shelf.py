"""Activate a voxel tool from the Asset Shelf thumbnail."""
from typing import Any

try:
    import bpy
    from bpy.props import EnumProperty, StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    EnumProperty = StringProperty = None

_TOOL_OPERATORS = {
    "ADD_SURFACE": "voxel.start_surface",
    "ADD_VOLUME": "voxel.start_volume",
    "REPAINT": "voxel.start_repaint",
    "ERASE": "voxel.start_erase",
    "STOP": "voxel.stop_editing",
}


_TOOL_NAMES = {
    "Voxel Add Surface": "ADD_SURFACE",
    "Voxel Add Volume": "ADD_VOLUME",
    "Voxel Repaint": "REPAINT",
    "Voxel Erase": "ERASE",
    "Voxel Stop Editing": "STOP",
}


def _asset_tool_id(context: Any, relative_asset_identifier: str = "") -> str:
    asset = getattr(context, "asset", None) if context is not None else None
    local = getattr(asset, "local_id", None) if asset is not None else None
    if local is not None:
        value = local.get("voxel_shelf_tool", "")
        if value:
            return str(value)
    # Asset Shelf activation is required to pass weak-reference properties.
    # In some UI contexts bpy.context.asset is unavailable, so resolve local
    # generated assets from the final identifier component instead.
    identifier = str(relative_asset_identifier or "").replace("\\", "/").rstrip("/")
    asset_name = identifier.rsplit("/", 1)[-1] if identifier else ""
    return _TOOL_NAMES.get(asset_name, "")


class VOXEL_OT_shelf_activate(Operator):
    """Run the voxel tool bound to the clicked Asset Shelf thumbnail."""

    bl_idname = "voxel.shelf_activate"
    bl_label = "Activate Voxel Shelf Tool"
    bl_description = "Activate the voxel tool for the selected Asset Shelf thumbnail"
    bl_options = {"INTERNAL"}

    if bpy is not None:
        asset_library_type: EnumProperty(
            name="Asset Library Type",
            items=[
                ("ALL", "All", "All asset libraries"),
                ("LOCAL", "Current File", "Assets in the current file"),
                ("ESSENTIALS", "Essentials", "Built-in essentials library"),
                ("ONLINE_ESSENTIALS", "Online Essentials", "Online essentials library"),
                ("CUSTOM", "Custom", "Custom asset library"),
            ],
            default="LOCAL",
            options={'HIDDEN'},
        )
        asset_library_identifier: StringProperty(options={'HIDDEN'})
        relative_asset_identifier: StringProperty(options={'HIDDEN'})

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        tool_id = _asset_tool_id(
            context,
            getattr(self, "relative_asset_identifier", ""),
        )
        op_id = _TOOL_OPERATORS.get(tool_id)
        if not op_id:
            self.report({"WARNING"}, "No voxel tool bound to this asset")
            return {"CANCELLED"}
        mod, name = op_id.split(".")
        getattr(getattr(bpy.ops, mod), name)()
        return {"FINISHED"}


SHELF_OPERATOR_CLASSES = [VOXEL_OT_shelf_activate]
