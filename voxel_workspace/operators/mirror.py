"""Instant mirror operator: copies one half of the volume onto the other.

Axis and paint-only mode are scene-level properties (radio row + checkbox in
the Voxel Palette panel); the two buttons only supply the direction.
"""
from typing import Any

try:
    import bpy
    from bpy.props import EnumProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    EnumProperty = None

from ..blender.gpu_preview import update_volume_gpu_preview, drop_palette_lut
from ..blender.object_graph import resolve_volume_context
from ..blender.persistence import serialize_volume
from ..blender.runtime import get_or_load, tag_redraw_all_viewports
from ..core.mirror import (
    mirror_half_to_half,
    mirror_half_paint_only,
    mirrored_cells_for_axes,
)


def live_mirror_axes(scene: Any) -> dict:
    """Read the active live-mirror axes from scene properties."""
    props = getattr(scene, "voxel_workspace", None)
    if props is None:
        return {}
    return {
        "X": bool(getattr(props, "mirror_live_x", False)),
        "Y": bool(getattr(props, "mirror_live_y", False)),
        "Z": bool(getattr(props, "mirror_live_z", False)),
    }


class VOXEL_OT_mirror(Operator):
    """Instantly copy one half of the volume onto the mirrored half."""

    bl_idname = "voxel.mirror"
    bl_label = "Mirror"
    bl_options = {'REGISTER', 'UNDO'}

    if bpy is not None:
        direction: EnumProperty(
            name="Direction",
            items=[
                ("NEG_TO_POS", "- → +",
                 "Copy the negative side onto the positive side"),
                ("POS_TO_NEG", "+ → -",
                 "Copy the positive side onto the negative side"),
            ],
            default="NEG_TO_POS",
        )

    @classmethod
    def description(cls, context, properties):
        direction = str(getattr(properties, "direction", "NEG_TO_POS"))
        axis = "XYZ"
        if context is not None:
            scene_props = getattr(context.scene, "voxel_workspace", None)
            if scene_props is not None:
                axis = str(getattr(scene_props, "mirror_axis", "X")).upper()
        paint_only = False
        if context is not None:
            scene_props = getattr(context.scene, "voxel_workspace", None)
            if scene_props is not None:
                paint_only = bool(getattr(scene_props, "mirror_paint_only", False))
        arrow = "-" if direction == "NEG_TO_POS" else "+"
        target = "+" if direction == "NEG_TO_POS" else "-"
        mode = "Recolor matching voxels" if paint_only else "Copy geometry and colors"
        return (
            f"{axis}{arrow} → {axis}{target}: {mode} across the "
            f"{axis} center plane"
        )

    def draw(self, context):
        # Axis/paint-only come from the panel; direction from the button.
        pass

    def execute(self, context: Any):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        entry = get_or_load(mesh)
        if entry is None or entry.grid is None:
            self.report({'WARNING'}, "Volume has no grid data")
            return {'CANCELLED'}
        grid = entry.grid

        scene_props = getattr(context.scene, "voxel_workspace", None)
        axis = str(getattr(scene_props, "mirror_axis", "X")).upper()
        paint_only = bool(getattr(scene_props, "mirror_paint_only", False))
        direction = str(getattr(self, "direction", "NEG_TO_POS")).upper()

        if paint_only:
            changed, changed_bricks = mirror_half_paint_only(grid, axis, direction)
        else:
            changed, changed_bricks = mirror_half_to_half(grid, axis, direction)

        if not changed_bricks:
            self.report({'INFO'}, f"Nothing to mirror along {axis}")
            return {'CANCELLED'}

        grid.dirty_bricks.update(changed_bricks)
        entry.dirty_bricks.update(changed_bricks)
        serialize_volume(mesh, grid, dirty_only=True)
        from ..blender.mesh_sync import sync_volume_mesh
        sync_volume_mesh(
            mesh,
            grid=grid,
            dirty_only=True,
            dirty_bricks=changed_bricks,
            voxel_size=entry.voxel_size,
        )
        drop_palette_lut(mesh.voxel_workspace.uuid)
        update_volume_gpu_preview(entry, dirty_only=False)
        tag_redraw_all_viewports()

        mode = "Paint-only mirror" if paint_only else "Mirror"
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message=f"{mode} {axis} ({direction.replace('_TO_', '→ ')})")
            except Exception:
                pass
        self.report({'INFO'}, f"{mode}: {changed} voxels along {axis}")
        return {'FINISHED'}


MIRROR_OPERATOR_CLASSES = [VOXEL_OT_mirror]
