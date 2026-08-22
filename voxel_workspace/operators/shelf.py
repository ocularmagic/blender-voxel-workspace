"""Activate a voxel tool from the Asset Shelf thumbnail."""
from typing import Any

try:
    import bpy
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object

_TOOL_OPERATORS = {
    "ADD_SURFACE": "voxel.start_surface",
    "ADD_VOLUME": "voxel.start_volume",
    "ERASE": "voxel.start_erase",
    "STOP": "voxel.stop_editing",
}


def _asset_tool_id(context: Any) -> str:
    asset = getattr(context, "asset", None) if context is not None else None
    local = getattr(asset, "local_id", None) if asset is not None else None
    if local is None:
        return ""
    value = local.get("voxel_shelf_tool", "")
    return str(value or "")


class VOXEL_OT_shelf_activate(Operator):
    """Run the voxel tool bound to the clicked Asset Shelf thumbnail."""

    bl_idname = "voxel.shelf_activate"
    bl_label = "Activate Voxel Shelf Tool"
    bl_description = "Activate the voxel tool for the selected Asset Shelf thumbnail"
    bl_options = {"INTERNAL"}

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        op_id = _TOOL_OPERATORS.get(_asset_tool_id(context))
        if not op_id:
            self.report({"WARNING"}, "No voxel tool bound to this asset")
            return {"CANCELLED"}
        mod, name = op_id.split(".")
        getattr(getattr(bpy.ops, mod), name)()
        return {"FINISHED"}


SHELF_OPERATOR_CLASSES = [VOXEL_OT_shelf_activate]
