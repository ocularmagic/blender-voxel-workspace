"""Voxel Workspace panels for supported Blender 5.1 UI regions.

Layout:
* Voxel Palette -> native VIEW_3D N-panel category (same tab strip as Item/Tool/Voxel).
* Edit tools -> VIEW_3D Asset Shelf thumbnails.
* Voxel settings stay on the right N-panel Voxel tab.
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
from ..blender.material_domains import get_palette, display_rgba_from_entry

# Operator chips show a scaled icon slot. Keep the source bitmap at 32px.
PALETTE_SWATCH_SIZE_PX = 32
# Button scale factor for the swatch chips. 1.0 renders the icon at the default
# ~16px slot; raising it makes the visible swatch larger. Grid stays at 8 columns.
PALETTE_SWATCH_SCALE = 1.5


def _principled_volume_node(tree: Any) -> Any:
    for node in getattr(tree, "nodes", []):
        if getattr(node, "bl_idname", "") == "ShaderNodeVolumePrincipled":
            return node
    return tree.nodes.get("Principled Volume") if tree is not None else None


def _draw_node_inputs(layout: Any, node: Any) -> None:
    if node is None:
        return
    column = layout.column(align=True)
    for socket in node.inputs:
        if getattr(socket, "is_linked", False):
            continue
        if hasattr(socket, "enabled") and not socket.enabled:
            continue
        if getattr(socket, "type", "") == "SHADER":
            continue
        try:
            column.prop(socket, "default_value", text=socket.name)
        except Exception:
            continue


def _draw_material_socket_widget(layout: Any, material: Any, pal_tab: str) -> None:
    """Draw the same Surface/Volume socket UI as the Material properties tab."""
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return
    layout.use_property_split = True
    if str(pal_tab).upper() == "VOLUME":
        _draw_node_inputs(layout, _principled_volume_node(tree))
        return
    try:
        from bl_ui.properties_material import panel_node_draw
        panel_node_draw(layout, tree, "OUTPUT_MATERIAL", "Surface")
        return
    except Exception:
        pass
    bsdf = tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next((node for node in tree.nodes if getattr(node, "bl_idname", "") == "ShaderNodeBsdfPrincipled"), None)
    _draw_node_inputs(layout, bsdf)


def _draw_palette_entry_editor(
    layout: Any,
    active_entry: Any,
    pal_tab: str,
    active_index: int,
    counts: Any,
    all_entries: Any,
) -> None:
    """Draw active-entry controls in a block separate from the preview."""
    material = getattr(active_entry, "material", None)
    if material is not None:
        _draw_material_socket_widget(layout, material, pal_tab)

    edit_box = layout.box()
    active_count = counts.get(active_index, 0)
    edit_box.label(
        text=f"{pal_tab.title()} [{active_index}]  •  {active_count} voxels",
        icon='COLOR',
    )

    mat_box = edit_box.box()
    mat_box.label(text="Native Blender Material", icon='MATERIAL')
    mat_box.label(text=f"Material: {active_entry.material.name if active_entry.material else 'None'}")
    edit_op = mat_box.operator("voxel.edit_palette_material", text="Edit Material Binding…", icon='PREFERENCES')
    edit_op.palette_type = pal_tab
    edit_op.index = active_index

    if not active_entry.material_owned and active_entry.material is not None:
        op_single = mat_box.operator("voxel.make_material_single_user", text="Make Single User", icon='UNLINKED')
        op_single.palette_type = pal_tab
        op_single.index = active_index

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
# Shared typed-palette renderer
# ---------------------------------------------------------------------------
def draw_typed_palette(
    layout: Any,
    context: Any,
    *,
    compact: bool = False,
) -> None:
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

    # Material Type / Palette Selector Tabs. Operators keep the typed palette
    # and the active Surface/Volume placement tool synchronized both ways.
    tab_row = layout.row(align=True)
    surface_tab = tab_row.operator(
        "voxel.select_palette_tab",
        text="Surface",
        depress=pal_tab == "SURFACE",
    )
    surface_tab.palette_type = "SURFACE"
    volume_tab = tab_row.operator(
        "voxel.select_palette_tab",
        text="Volume",
        depress=pal_tab == "VOLUME",
    )
    volume_tab.palette_type = "VOLUME"

    header = layout.row()
    if is_voxel and mesh is not None:
        op_pick = header.operator("voxel.eyedropper", text="Pick", icon='EYEDROPPER')
        op_add = header.operator("voxel.add_palette_color", text="Add", icon='ADD')
        op_add.palette_type = pal_tab
        op_comp = header.operator("voxel.compact_palette", text="Compact", icon='ALIGN_JUSTIFY')
        op_comp.palette_type = pal_tab
        op_sort = header.operator("voxel.sort_palette_color", text="Sort", icon='SORT_ASC')
        op_sort.palette_type = pal_tab

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

    # Square 16px operator icons, 8 per row. Do not scale the button or
    # stack template_icon — both distort or hide the chip.
    columns = 8
    grid_flow = layout.grid_flow(
        row_major=True,
        columns=columns,
        even_columns=False,
        even_rows=False,
    )
    for entry_item in entries:
        idx = entry_item.index
        is_active = (idx == active_index)
        is_used = (counts.get(idx, 0) > 0)
        icon_id = generate_swatch_icon_id(
            display_rgba_from_entry(entry_item, pal_tab),
            is_active=is_active,
            is_used=is_used,
            size=PALETTE_SWATCH_SIZE_PX,
            material=getattr(entry_item, "material", None),
        )
        cell = grid_flow.column(align=True)
        cell.scale_y = PALETTE_SWATCH_SCALE
        cell.scale_x = PALETTE_SWATCH_SCALE
        op = cell.operator(
            "voxel.select_palette_color",
            text="",
            icon_value=icon_id if icon_id else 0,
            depress=is_active,
        )
        op.palette_type = pal_tab
        op.index = idx

    # Active material preview sits flush under the last chip row.
    active_entry = next((e for e in target_palette if e.index == active_index), None)
    if active_entry is not None:
        material = getattr(active_entry, "material", None)
        if material is not None:
            try:
                # Match Blender's Material Properties panel exactly. Do not
                # force a preview render here; Blender owns this widget's
                # refresh and interaction state.
                layout.template_preview(material)
            except Exception:
                pass

        _draw_palette_entry_editor(
            layout,
            active_entry,
            pal_tab,
            active_index,
            counts,
            all_entries,
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
    """Surface / Volume palette as a native N-panel category tab."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Voxel Palette'
    bl_label = 'Voxel Palette'
    bl_order = 1

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.label(text="Voxel Workspace", icon='MESH_CUBE')
        draw_typed_palette(layout, context, compact=True)


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
        export_row = vol_box.row()
        export_row.scale_y = 1.2
        export_row.operator("voxel.export_slices", text="Export Voxel Slices", icon="EXPORT")
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
    bl_order = 0

    @classmethod
    def poll(cls, context: Any) -> bool:
        return True

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
    VOXEL_PT_main_panel,
    VOXEL_PT_palette_panel,
]
