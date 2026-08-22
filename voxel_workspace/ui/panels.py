"""Voxel Workspace panels for supported Blender 5.1 UI regions.

Layout:
* Left palette -> VIEW_3D sidebar (UI), flipped left by the custom workspace.
* Bottom tools -> VIEW_3D tool header callback, flipped bottom by the workspace.
* Settings -> right Properties editor in the custom workspace, with a trimmed
  VIEW_3D N-panel fallback outside it.
"""
from typing import Any

try:
    import bpy
    from bpy.types import Panel
except ImportError:
    bpy = None
    Panel = object

from ..blender.runtime import get_volume
from ..blender.object_graph import resolve_volume_context
from ..operators.palette import get_used_palette_counts
from .palette_icons import generate_swatch_icon_id
from ..blender.material_domains import get_palette


# ---------------------------------------------------------------------------
# Shared typed-palette renderer
# ---------------------------------------------------------------------------
def draw_typed_palette(layout: Any, context: Any, *, compact: bool = False) -> None:
    """Render the active typed palette's tabs, swatches, and editor.

    Used by the left palette panel. ``compact`` trims spacing for the narrow
    left column.
    """
    if context is None or bpy is None:
        return

    scene = context.scene
    palette_props = scene.voxel_workspace if hasattr(scene, "voxel_workspace") else None
    if palette_props is None:
        layout.label(text="No voxel scene properties", icon='ERROR')
        return

    v_ctx = resolve_volume_context(context)
    if v_ctx is None:
        active_objects = getattr(getattr(context, "view_layer", None), "objects", None)
        active_object = getattr(active_objects, "active", None)
        if active_object is not None:
            v_ctx = resolve_volume_context(active_object)
    is_voxel = v_ctx is not None and v_ctx.mesh is not None
    mesh = v_ctx.mesh if is_voxel else None
    pal_tab = getattr(palette_props, "active_palette_tab", "SURFACE").upper()

    # Material Type / Palette Selector Tabs
    tab_row = layout.row(align=True)
    tab_row.prop(palette_props, "active_palette_tab", expand=True)

    header = layout.row()
    header.label(text=f"{pal_tab.title()} Palette", icon='COLOR')
    if is_voxel and mesh is not None:
        op_pick = header.operator("voxel.eyedropper", text="Pick", icon='EYEDROPPER')
        op_add = header.operator("voxel.add_palette_color", text="Add", icon='ADD')
        op_add.palette_type = pal_tab
        op_comp = header.operator("voxel.compact_palette", text="Compact", icon='ALIGN_JUSTIFY')
        op_comp.palette_type = pal_tab

    active_index = (
        palette_props.active_volume_palette_index
        if pal_tab == "VOLUME"
        else palette_props.active_surface_palette_index
    )

    if not (is_voxel and mesh is not None):
        layout.label(text=f"Active Brush Index: {active_index}", icon='RADIOBUT_ON')
        return

    props = mesh.voxel_workspace
    counts = get_used_palette_counts(mesh, palette_type=pal_tab)

    # Filter and Preset Tools Row
    top_row = layout.row(align=True)
    top_row.prop(palette_props, "palette_filter", expand=True)
    preset_sub = top_row.row(align=True)
    load_op = preset_sub.operator("voxel.load_palette_preset", text="Load", icon='IMPORT')
    load_op.palette_type = pal_tab
    save_op = preset_sub.operator("voxel.save_palette_preset", text="Save", icon='EXPORT')
    save_op.palette_type = pal_tab

    target_palette = get_palette(mesh, pal_tab)
    all_entries = sorted([e for e in target_palette if e.index > 0], key=lambda e: e.index)

    if palette_props.palette_filter == "USED":
        entries = [e for e in all_entries if counts.get(e.index, 0) > 0]
    else:
        entries = all_entries

    # Swatch Grid
    columns = 4 if compact else 8
    grid_flow = layout.grid_flow(row_major=True, columns=columns, even_columns=True, even_rows=True)
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
        op = row.operator(
            "voxel.select_palette_color",
            text="",
            icon_value=icon_id if icon_id != 0 else 0,
        )
        op.palette_type = pal_tab
        op.index = idx

    # Active Swatch Editor Details
    active_entry = next((e for e in target_palette if e.index == active_index), None)
    if active_entry is not None:
        edit_box = layout.box()
        active_count = counts.get(active_index, 0)
        edit_box.label(
            text=f"{pal_tab.title()} Color [{active_index}]  •  {active_count} voxels",
            icon='COLOR',
        )
        edit_box.prop(active_entry, "color", text="")
        edit_box.label(text="Display color controls brush/GPU preview; Material controls render.", icon='INFO')
        edit_box.prop(active_entry, "name", text="Name")

        mat_box = edit_box.box()
        mat_box.label(text="Native Blender Material", icon='MATERIAL')
        mat_box.label(text=f"Material: {active_entry.material.name if active_entry.material else 'None'}")
        edit_op = mat_box.operator("voxel.edit_palette_material", text="Edit Material Binding…", icon='PREFERENCES')
        edit_op.palette_type = pal_tab
        edit_op.index = active_index

        sync_row = mat_box.row(align=True)
        op_apply = sync_row.operator("voxel.sync_display_to_material", text="Apply Display", icon='FORWARD')
        op_apply.palette_type = pal_tab
        op_apply.index = active_index
        op_read = sync_row.operator("voxel.sync_material_to_display", text="Read Base Color", icon='BACK')
        op_read.palette_type = pal_tab
        op_read.index = active_index

        if not active_entry.material_owned and active_entry.material is not None:
            op_single = mat_box.operator("voxel.make_material_single_user", text="Make Single User", icon='UNLINKED')
            op_single.palette_type = pal_tab
            op_single.index = active_index

        # Duplicate & Remove Actions
        btn_row = edit_box.row(align=True)
        dup_op = btn_row.operator("voxel.duplicate_palette_color", text="Duplicate", icon='DUPLICATE')
        dup_op.palette_type = pal_tab
        dup_op.source_index = active_index
        rem_op = btn_row.operator("voxel.remove_palette_color", text="Remove", icon='TRASH')
        rem_op.palette_type = pal_tab
        rem_op.index = active_index
        rem_op.replacement_index = (
            1 if active_index != 1 else (all_entries[1].index if len(all_entries) > 1 else 0)
        )


# ---------------------------------------------------------------------------
# Workspace helpers and left palette panel
# ---------------------------------------------------------------------------
def _in_voxel_workspace(context: Any) -> bool:
    workspace = getattr(context, "workspace", None) if context is not None else None
    return workspace is not None and workspace.name == "Voxel Workspace"


def _sidebar_alignment(context: Any) -> str:
    region = getattr(context, "region", None) if context is not None else None
    return getattr(region, "alignment", "") if region is not None else ""


class VOXEL_PT_palette_panel(Panel):
    """Surface / Volume palette in the left panel of Voxel Workspace."""
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Voxel Palette'
    bl_label = 'Voxel Palette'

    @classmethod
    def poll(cls, context: Any) -> bool:
        return _in_voxel_workspace(context)

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.label(text="Voxel Workspace", icon='MESH_CUBE')
        draw_typed_palette(layout, context, compact=True)


# ---------------------------------------------------------------------------
# Bottom tool bar (ASSET_SHELF region)
# ---------------------------------------------------------------------------
_TOOL_SPECS = (
    ("ADD_SURFACE", "voxel.start_surface", "ADD SURFACE", 'BRUSH_DATA'),
    ("ADD_VOLUME", "voxel.start_volume", "ADD VOLUME", 'MOD_FLUIDSIM'),
    ("ERASE", "voxel.start_erase", "ERASE VOXEL", 'REMOVE'),
)


def draw_voxel_tool_header(self: Any, context: Any) -> None:
    """Draw square voxel brush buttons in the bottom-flipped tool header."""
    if context is None or bpy is None or not _in_voxel_workspace(context):
        return
    layout = self.layout
    sc_props = getattr(context.scene, "voxel_workspace", None)
    active_tool = getattr(sc_props, "active_tool", "NONE") if sc_props else "NONE"

    row = layout.row(align=True)
    row.scale_y = 1.35
    row.alignment = 'CENTER'
    for mode, op_id, _label, icon in _TOOL_SPECS:
        sub = row.row(align=True)
        sub.alert = active_tool == mode
        sub.scale_x = 1.35
        sub.operator(op_id, text="", icon=icon)

    if active_tool in {"ADD_SURFACE", "ADD_VOLUME", "ERASE", "PLACE"}:
        row.separator(factor=0.5)
        row.operator("voxel.stop_editing", text="", icon='CANCEL')

    row.separator(factor=1.0)
    if sc_props is not None:
        if active_tool == 'ADD_VOLUME':
            active_index = sc_props.active_volume_palette_index
            label = "Volume"
        elif active_tool in {"ADD_SURFACE", "PLACE", "ERASE"}:
            active_index = sc_props.active_surface_palette_index
            label = "Surface"
        else:
            active_index = "-"
            label = "None"
        row.label(text=f"{label} [{active_index}]", icon='COLOR')


# ---------------------------------------------------------------------------
# Shared creation / import / volume settings renderer
# ---------------------------------------------------------------------------
def _draw_volume_settings(layout: Any, context: Any) -> None:
    if context is None or bpy is None:
        return

    scene = context.scene

    # 1. Volume Creation Section
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
    v_ctx = resolve_volume_context(context)
    is_voxel = v_ctx is not None and v_ctx.mesh is not None

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

        vol_uuid = props.uuid
        entry = get_volume(vol_uuid) if vol_uuid else None
        if entry is not None:
            occupied_bricks = len([b for b in entry.grid.bricks.values() if _any_occupied(b)])
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


def _any_occupied(brick: Any) -> bool:
    """Return True if a brick buffer has any non-zero cell."""
    try:
        import numpy as np
        return bool(np.any(brick))
    except Exception:
        return bool(brick)


class VOXEL_PT_main_panel(Panel):
    """Voxel N-panel on the default right sidebar."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Voxel'
    bl_label = 'Voxel Volume'

    @classmethod
    def poll(cls, context: Any) -> bool:
        return not _in_voxel_workspace(context) or _sidebar_alignment(context) == 'RIGHT'

    def draw(self, context: Any) -> None:
        _draw_volume_settings(self.layout, context)


class VOXEL_PT_workspace_settings(Panel):
    """Creation/import/settings in the custom workspace's right Properties editor."""
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'scene'
    bl_label = 'Voxel Volume'

    @classmethod
    def poll(cls, context: Any) -> bool:
        return _in_voxel_workspace(context)

    def draw(self, context: Any) -> None:
        _draw_volume_settings(self.layout, context)


PANEL_CLASSES = [
    VOXEL_PT_palette_panel,
    VOXEL_PT_main_panel,
]
