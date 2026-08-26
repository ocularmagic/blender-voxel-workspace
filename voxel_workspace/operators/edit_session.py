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
    if mesh is not None and hasattr(mesh, "voxel_workspace") \
            and mesh.voxel_workspace.is_voxel_mesh and bool(mesh.voxel_workspace.uuid):
        return True
    # Asset-shelf clicks can leave the view_layer without an active object
    # while a voxel volume is already loaded in the runtime registry. Trust
    # that registry entry so tool polls keep working mid-session.
    try:
        from ..blender.runtime import get_active_volume_uuid
        return bool(get_active_volume_uuid())
    except Exception:
        return False


def _resolve_v_ctx_with_fallback(context: Any):
    """Resolve volume context, falling back to the runtime registry.

    Asset-shelf activation can drop the active object from context while an
    editing session is still valid. Resolve through any object carrying the
    active volume UUID so tool startup does not dead-end.
    """
    v_ctx = resolve_volume_context(context)
    if v_ctx is not None and v_ctx.mesh_uuid:
        return v_ctx
    try:
        from ..blender.runtime import get_active_volume_uuid
        active_uuid = get_active_volume_uuid()
    except Exception:
        return None
    if not active_uuid or bpy is None:
        return None
    for obj in getattr(getattr(context, "scene", None), "objects", []):
        mesh = getattr(obj, "data", None)
        props = getattr(mesh, "voxel_workspace", None) if mesh is not None else None
        if props is not None and getattr(props, "uuid", "") == active_uuid:
            return resolve_volume_context(obj)
    return None


def _start_brush(context: Any, operator: Any, mode: str) -> set:
    if bpy is None or context is None:
        return {'CANCELLED'}
    if str(getattr(context.scene.voxel_workspace, "active_tool", "NONE")) in ("ADJUST", "SCALE"):
        operator.report({'WARNING'}, "Exit the active adjustment tool before editing voxels")
        return {'CANCELLED'}
    v_ctx = _resolve_v_ctx_with_fallback(context)
    if v_ctx is None or not v_ctx.mesh_uuid:
        operator.report({'ERROR'}, "Active object is not a valid voxel field")
        return {'CANCELLED'}
    _request_brush_stop()
    stop_editing(context)
    start_editing(v_ctx.mesh_uuid, context)
    context.scene.voxel_workspace.active_tool = mode
    if mode == "ADD_SURFACE":
        context.scene.voxel_workspace.active_palette_tab = "SURFACE"
    elif mode == "ADD_VOLUME":
        context.scene.voxel_workspace.active_palette_tab = "VOLUME"
    try:
        bpy.ops.voxel.brush('INVOKE_DEFAULT', mode=mode)
    except Exception:
        pass
    return {'FINISHED'}


class VOXEL_OT_start_surface(Operator):
    """Start adding Surface voxels."""
    bl_idname = "voxel.start_surface"
    bl_label = "Add Surface"
    bl_description = "Add voxels using the active Surface Palette entry"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        if context is None:
            return False
        return is_valid_voxel_object(context)

    def execute(self, context: Any) -> set:
        return _start_brush(context, self, 'ADD_SURFACE')


class VOXEL_OT_start_volume(Operator):
    """Start adding Volume voxels."""
    bl_idname = "voxel.start_volume"
    bl_label = "Add Volume"
    bl_description = "Add voxels using the active Volume Palette entry"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return context is not None and is_valid_voxel_object(context)

    def execute(self, context: Any) -> set:
        return _start_brush(context, self, 'ADD_VOLUME')


class VOXEL_OT_start_place(Operator):
    """Compatibility alias for the former Place tool."""
    bl_idname = "voxel.start_place"
    bl_label = "Add Surface"
    bl_description = "Compatibility alias for Add Surface"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return context is not None and is_valid_voxel_object(context)

    def execute(self, context: Any) -> set:
        return _start_brush(context, self, 'ADD_SURFACE')


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

        return _start_brush(context, self, 'ERASE')


class VOXEL_OT_start_repaint(Operator):
    """Start repainting existing voxels to the active palette material."""
    bl_idname = "voxel.start_repaint"
    bl_label = "Start Repaint"
    bl_description = "Start interactive modal voxel repaint tool"
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

        return _start_brush(context, self, 'REPAINT')


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
    VOXEL_OT_start_surface,
    VOXEL_OT_start_volume,
    VOXEL_OT_start_place,
    VOXEL_OT_start_erase,
    VOXEL_OT_start_repaint,
    VOXEL_OT_stop_editing,
]
