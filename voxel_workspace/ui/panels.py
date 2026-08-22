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
from ..operators.palette import get_used_palette_counts
from .palette_icons import generate_swatch_icon_id


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
        scene = context.scene
        create_box = layout.box()
        create_box.label(text="Create Volume", icon='CUBE')
        if scene is not None and hasattr(scene, "voxel_workspace"):
            sc_props = scene.voxel_workspace
            col_dims = create_box.column(align=True)
            row_xyz = col_dims.row(align=True)
            row_xyz.prop(sc_props, "create_size_x", text="X")
            row_xyz.prop(sc_props, "create_size_y", text="Y")
            row_xyz.prop(sc_props, "create_size_z", text="Z")
            col_dims.prop(sc_props, "create_voxel_size", text="Voxel Size")

            op = create_box.operator("voxel.create_volume", text="Create Volume", icon='ADD')
            op.size_x = sc_props.create_size_x
            op.size_y = sc_props.create_size_y
            op.size_z = sc_props.create_size_z
            op.voxel_size = sc_props.create_voxel_size
        else:
            create_box.operator("voxel.create_volume", text="Create Volume", icon='ADD')

        # 2. Active Volume Context & Inspection
        from ..blender.object_graph import resolve_volume_context
        v_ctx = resolve_volume_context(context)
        is_voxel = v_ctx is not None and v_ctx.mesh is not None
        obj = v_ctx.root if (v_ctx and v_ctx.root) else (v_ctx.surface_object if v_ctx else context.active_object)

        vol_box = layout.box()
        mesh = v_ctx.mesh if is_voxel else None
        if is_voxel and mesh is not None:
            props = mesh.voxel_workspace
            display_name = v_ctx.root.name if v_ctx.root else v_ctx.surface_object.name
            vol_box.label(text=f"Volume: {display_name}", icon='MESH_CUBE')

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
            empty = occupied_bricks == 0
            info_col.label(text="Volume is empty" if empty else "Volume has voxels")
            import_row = vol_box.row()
            import_row.scale_y = 1.2
            import_row.operator("voxel.import_glb", text="Import GLB into Volume", icon="IMPORT")
        else:
            vol_box.label(text="No active voxel volume", icon="INFO")

        # 3. Dynamic Palette Section for Active Volume
        scene = context.scene
        if scene is not None and hasattr(scene, "voxel_workspace"):
            pal_box = layout.box()
            pal_header = pal_box.row()
            pal_header.label(text="Palette", icon='COLOR')
            if is_voxel and mesh is not None:
                pal_header.operator("voxel.eyedropper", text="Pick", icon='EYEDROPPER')
                pal_header.operator("voxel.add_palette_color", text="Add", icon='ADD')
                pal_header.operator("voxel.compact_palette", text="Compact", icon='ALIGN_JUSTIFY')

            palette_props = scene.voxel_workspace
            active_index = palette_props.active_palette_index

            if is_voxel and mesh is not None:
                props = mesh.voxel_workspace
                counts = get_used_palette_counts(mesh)

                # Filter and Preset Tools Row
                top_row = pal_box.row(align=True)
                top_row.prop(palette_props, "palette_filter", expand=True)
                preset_sub = top_row.row(align=True)
                preset_sub.operator("voxel.load_palette_preset", text="Load Preset", icon='IMPORT')
                preset_sub.operator("voxel.save_palette_preset", text="Save Preset", icon='EXPORT')

                # Collect non-zero entries sorted by index
                all_entries = sorted([e for e in props.palette if e.index > 0], key=lambda e: e.index)
                if palette_props.palette_filter == "USED":
                    entries = [e for e in all_entries if counts.get(e.index, 0) > 0]
                else:
                    entries = all_entries

                # Swatch Grid: Square buttons with swatch color fill, highlighted active border, and small center dot if used
                # 8 swatches per row, scale_y=1.0 for square button outline
                grid_flow = pal_box.grid_flow(row_major=True, columns=8, even_columns=True, even_rows=True)
                for entry_item in entries:
                    idx = entry_item.index
                    is_active = (idx == active_index)
                    is_used = (counts.get(idx, 0) > 0)
                    icon_id = generate_swatch_icon_id(
                        tuple(entry_item.color),
                        is_active=is_active,
                        is_used=is_used,
                        size=32,
                    )
                    
                    cell_box = grid_flow.column(align=True)
                    row = cell_box.row(align=True)
                    row.scale_y = 1.0
                    row.scale_x = 1.0
                    # Square button with custom swatch icon (color fill + active border + center used dot) and no number text
                    op = row.operator(
                        "voxel.edit_palette_material",
                        text="",
                        icon_value=icon_id if icon_id != 0 else 0,
                    )
                    op.index = idx

                # Active Swatch Editor Details
                active_entry = next((e for e in props.palette if e.index == active_index), None)
                if active_entry is not None:
                    edit_box = pal_box.box()
                    active_count = counts.get(active_index, 0)
                    edit_box.label(
                        text=f"Color [{active_index}]  •  {active_count} voxels",
                        icon='COLOR',
                    )
                    # Color picker for in-place editing
                    edit_box.prop(active_entry, "color", text="")
                    edit_box.label(text="Display color controls brush/GPU preview; Material controls render.", icon='INFO')
                    edit_box.prop(active_entry, "name", text="Name")

                    # Native Material Controls
                    mat_box = edit_box.box()
                    mat_box.label(text="Native Blender Material", icon='MATERIAL')
                    mat_box.label(text=f"Domain: {active_entry.material_domain.title()}")
                    mat_box.label(text=f"Material: {active_entry.material.name if active_entry.material else 'None'}")
                    edit_op = mat_box.operator("voxel.edit_palette_material", text="Edit Material Binding…", icon='PREFERENCES')
                    edit_op.index = active_index

                    sync_row = mat_box.row(align=True)
                    op_apply = sync_row.operator("voxel.sync_display_to_material", text="Apply Display", icon='FORWARD')
                    op_apply.index = active_index
                    op_read = sync_row.operator("voxel.sync_material_to_display", text="Read Base Color", icon='BACK')
                    op_read.index = active_index

                    if not active_entry.material_owned and active_entry.material is not None:
                        op_single = mat_box.operator("voxel.make_material_single_user", text="Make Single User", icon='UNLINKED')
                        op_single.index = active_index

                    # Duplicate & Remove Actions
                    btn_row = edit_box.row(align=True)
                    dup_op = btn_row.operator("voxel.duplicate_palette_color", text="Duplicate", icon='DUPLICATE')
                    dup_op.source_index = active_index

                    rem_op = btn_row.operator("voxel.remove_palette_color", text="Remove", icon='TRASH')
                    rem_op.index = active_index
                    rem_op.replacement_index = 1 if active_index != 1 else (all_entries[1].index if len(all_entries) > 1 else 0)
            else:
                # Fallback display when no volume selected
                pal_box.label(text=f"Active Brush Index: {active_index}", icon='RADIOBUT_ON')

        # 4. Tool & Stroke Actions (Task 11 safe references)
        tools_box = layout.box()
        tool_name = context.scene.voxel_workspace.active_tool.title()
        tools_box.label(text=f"Voxel Brush  •  {tool_name}", icon='TOOL_SETTINGS')
        tools_col = tools_box.column(align=True)
        tools_col.prop(
            context.scene.voxel_workspace,
            "show_voxel_edges",
            text="Show Voxel Edges",
        )
        # Active Color Indicator near brush controls
        active_idx = context.scene.voxel_workspace.active_palette_index
        tools_col.label(text=f"Active Brush Color: Index [{active_idx}]", icon='COLOR')
        tools_col.separator(factor=0.5)

        # Start Place
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "start_place"):
            button = tools_col.row(align=True)
            button.scale_y = 1.4
            button.operator("voxel.start_place", text="PLACE VOXELS", icon='BRUSH_DATA')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Start Place", icon='BRUSH_DATA')

        # Start Erase
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "start_erase"):
            button = tools_col.row(align=True)
            button.scale_y = 1.4
            button.operator("voxel.start_erase", text="ERASE VOXELS", icon='REMOVE')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Start Erase", icon='REMOVE')

        # Stop Editing
        if hasattr(bpy.ops, "voxel") and hasattr(bpy.ops.voxel, "stop_editing"):
            tools_col.separator(factor=0.5)
            tools_col.operator("voxel.stop_editing", text="Stop Editing  (Esc)", icon='CANCEL')
        else:
            row = tools_col.row()
            row.enabled = False
            row.label(text="Stop Editing", icon='CANCEL')


PANEL_CLASSES = [
    VOXEL_PT_main_panel,
]
