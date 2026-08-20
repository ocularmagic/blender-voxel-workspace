"""Voxel Workspace N-panel UI."""
from typing import Any
import numpy as np

try:
    import bpy
    from bpy.types import Panel
except ImportError:
    bpy = None
    Panel = object

from ..blender.runtime import get_volume


class VOXEL_PT_main_panel(Panel):
    """Voxel Workspace main N-panel in the 3D Viewport."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Voxel'
    bl_label = 'Voxel Volume'

    def draw(self, context: Any) -> None:
        layout = self.layout
        if context is None or bpy is None:
            return

        # 1. Volume Creation Section
        create_box = layout.box()
        create_box.label(text="Creation", icon='PLUS')
        create_box.operator("voxel.create_volume", text="Create Volume", icon='ADD')

        # 2. Active Volume Context & Inspection
        obj = context.active_object
        is_voxel = (
            obj is not None
            and obj.type == 'MESH'
            and hasattr(obj, "data")
            and obj.data is not None
            and hasattr(obj.data, "voxel_workspace")
            and obj.data.voxel_workspace.is_voxel_mesh
        )

        vol_box = layout.box()
        if is_voxel:
            mesh = obj.data
            props = mesh.voxel_workspace
            vol_box.label(text=f"Volume: {obj.name}", icon='MESH_CUBE')

            ext_min = tuple(props.extent_min)
            ext_max = tuple(props.extent_max)
            dim_x = ext_max[0] - ext_min[0]
            dim_y = ext_max[1] - ext_min[1]
            dim_z = ext_max[2] - ext_min[2]

            info_col = vol_box.column(align=True)
            info_col.label(text=f"Dimensions: {dim_x} × {dim_y} × {dim_z}")
            info_col.label(text=f"Voxel Size: {props.voxel_size:.4g}")

            # Determine occupied brick count
            vol_uuid = props.uuid
            entry = get_volume(vol_uuid) if vol_uuid else None
            if entry is not None:
                occupied_bricks = len([b for b in entry.grid.bricks.values() if np.any(b)])
            else:
                occupied_bricks = len(
                    [k for k in mesh.keys() if k.startswith("vox_brick_") and not k.endswith("_len")]
                )
            info_col.label(text=f"Occupied Bricks: {occupied_bricks}")
        else:
            vol_box.label(text="No active voxel volume", icon='INFO')

        # 3. Palette Section
        scene = context.scene
        if scene is not None and hasattr(scene, "voxel_workspace"):
            pal_box = layout.box()
            pal_box.label(text="Palette", icon='COLOR')
            pal_box.prop(scene.voxel_workspace, "active_palette_index", text="Active Index")

        # 4. Tool & Stroke Actions (Task 11 safe references)
        tools_box = layout.box()
        tools_box.label(text="Tools", icon='TOOL_SETTINGS')
        tools_col = tools_box.column(align=True)

        # Start Place
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "start_place"):
            tools_col.operator("voxel.start_place", text="Start Place", icon='BRUSH_DATA')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Start Place", icon='BRUSH_DATA')

        # Start Erase
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "start_erase"):
            tools_col.operator("voxel.start_erase", text="Start Erase", icon='ERASER')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Start Erase", icon='ERASER')

        # Stop Editing
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "stop_editing"):
            tools_col.operator("voxel.stop_editing", text="Stop Editing", icon='CANCEL')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Stop Editing", icon='CANCEL')


PANEL_CLASSES = [
    VOXEL_PT_main_panel,
]
