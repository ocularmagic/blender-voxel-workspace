"""Operators for managing voxel edit sessions."""
from typing import Any, Optional

try:
    import bpy
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object

from ..blender.gpu_preview import (
    start_editing,
    stop_editing,
    is_editing_active,
)
from ..blender.runtime import (
    get_active_volume_uuid,
    has_volume,
    get_volume,
)
from ..blender.object_graph import (
    resolve_volume_context,
    resolve_authoritative_mesh,
    resolve_voxel_root,
    resolve_surface_object,
)


def _request_brush_stop() -> None:
    """Invalidate any running voxel brush before changing edit sessions."""
    from .brush import request_brush_modal_stop
    request_brush_modal_stop()


def is_valid_voxel_object(obj_or_context: Any) -> bool:
    """Check if an object or context resolves to a valid voxel volume mesh datablock."""
    if obj_or_context is None:
        return False
    mesh = resolve_authoritative_mesh(obj_or_context)
    return bool(
        mesh is not None
        and hasattr(mesh, "voxel_workspace")
        and mesh.voxel_workspace.is_voxel_mesh
        and bool(mesh.voxel_workspace.uuid)
    )


class VOXEL_OT_start_place(Operator):
    """Start placing voxels on the active voxel volume."""
    bl_idname = "voxel.start_place"
    bl_label = "Start Place"
    bl_description = "Start interactive modal voxel place tool"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        if context is None:
            return False
        return is_valid_voxel_object(context)

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        v_ctx = resolve_volume_context(context)
        if v_ctx is None or not v_ctx.mesh_uuid:
            self.report({'ERROR'}, "Active object is not a valid voxel volume")
            return {'CANCELLED'}

        vol_uuid = v_ctx.mesh_uuid
        _request_brush_stop()
        stop_editing(context)
        start_editing(vol_uuid, context)
        context.scene.voxel_workspace.active_tool = 'PLACE'

        # Invoke modal brush with mode='PLACE'
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "brush"):
            try:
                bpy.ops.voxel.brush('INVOKE_DEFAULT', mode='PLACE')
            except Exception:
                pass

        return {'FINISHED'}


class VOXEL_OT_start_erase(Operator):
    """Start erasing voxels from the active voxel volume."""
    bl_idname = "voxel.start_erase"
    bl_label = "Start Erase"
    bl_description = "Start interactive modal voxel erase tool"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        if context is None:
            return False
        return is_valid_voxel_object(context)

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        v_ctx = resolve_volume_context(context)
        if v_ctx is None or not v_ctx.mesh_uuid:
            self.report({'ERROR'}, "Active object is not a valid voxel volume")
            return {'CANCELLED'}

        vol_uuid = v_ctx.mesh_uuid
        _request_brush_stop()
        stop_editing(context)
        start_editing(vol_uuid, context)
        context.scene.voxel_workspace.active_tool = 'ERASE'

        # Invoke modal brush with mode='ERASE'
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "brush"):
            try:
                bpy.ops.voxel.brush('INVOKE_DEFAULT', mode='ERASE')
            except Exception:
                pass

        return {'FINISHED'}


class VOXEL_OT_stop_editing(Operator):
    """Stop active voxel editing session and restore view overlays."""
    bl_idname = "voxel.stop_editing"
    bl_label = "Stop Editing"
    bl_description = "Stop active voxel editing session and restore view overlays"
    bl_options = {'REGISTER'}

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {'CANCELLED'}
        _request_brush_stop()
        stop_editing(context)
        return {'FINISHED'}


EDIT_SESSION_OPERATOR_CLASSES = [
    VOXEL_OT_start_place,
    VOXEL_OT_start_erase,
    VOXEL_OT_stop_editing,
]
