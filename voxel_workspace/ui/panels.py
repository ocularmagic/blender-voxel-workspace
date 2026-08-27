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
from ..operators.palette import (
    get_used_palette_counts,
    get_palette_selection,
)
from ..operators.adjust_voxel_root import is_adjustment_active
from ..operators.scale_voxels import is_scaling_active
from ..operators.import_glb import get_glb_analysis
from ..voxelization.resolution_analysis import volume_dimensions
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
    """Draw the Surface/Volume socket UI, whitelisted sockets expanded.

    Base Color (resolving through the render-only edge-mix node when linked),
    Metallic, Roughness, IOR, and Alpha are drawn as always-expanded rows.
    Everything else in the shader stays collapsed/hidden so the panel reads
    at a glance.
    """
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return
    layout.use_property_split = True
    if str(pal_tab).upper() == "VOLUME":
        # Volume palette keeps its flat full socket list; Color is the main
        # adjustment users make there.
        _draw_node_inputs(layout, _principled_volume_node(tree))
        return
    bsdf = tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = next((node for node in tree.nodes if getattr(node, "bl_idname", "") == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return

    def _draw_flat(socket_name: str, text: str = None) -> None:
        try:
            socket = bsdf.inputs[socket_name]
            if getattr(socket, "is_linked", False):
                row = layout.row(align=True)
                row.enabled = False
                row.prop(socket, "default_value", text=text or socket_name)
                return
            row = layout.row(align=True)
            row.prop(socket, "default_value", text=text or socket.name)
        except Exception:
            pass

    # Owned Surface materials route the visible color through the
    # 'VoxelSurfaceEdgeMix' render-only overlay; the true palette color
    # lives in that node's Color1 (input index 1).
    bc_socket = None
    try:
        bc_input = bsdf.inputs["Base Color"]
        if bc_input.is_linked:
            link = next(iter(bc_input.links), None)
            src_node = getattr(getattr(link, "from_socket", None), "node", None)
            if src_node is not None and getattr(src_node, "name", "") == "VoxelSurfaceEdgeMix":
                bc_socket = src_node.inputs[1]
        if bc_socket is None:
            bc_socket = bc_input
        bc_row = layout.row(align=True)
        bc_row.prop(bc_socket, "default_value", text="Base Color")
    except Exception:
        pass

    _draw_flat("Metallic")
    _draw_flat("Roughness")
    _draw_flat("IOR")
    _draw_flat("Alpha")


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

    # Material socket edits (Base Color etc.) bypass the palette RNA update
    # callbacks, so the cached GPU display LUTs can go stale. Diff-and-refresh
    # here; it rebuilds batches only when a bound material actually changed.
    if mesh is not None:
        try:
            from ..blender.gpu_preview import refresh_material_display_colors
            from ..blender.runtime import get_or_load
            refresh_material_display_colors(get_or_load(mesh))
        except Exception:
            pass

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

    if is_voxel and mesh is not None:
        row1 = layout.row(align=True)
        op_pick = row1.operator("voxel.eyedropper", text="Pick", icon='EYEDROPPER')
        op_add = row1.operator("voxel.add_palette_color", text="Add", icon='ADD')
        op_add.palette_type = pal_tab

        row2 = layout.row(align=True)
        op_comp = row2.operator("voxel.compact_palette", text="Compact", icon='ALIGN_JUSTIFY')
        op_comp.palette_type = pal_tab
        op_sort = row2.operator("voxel.sort_palette_color", text="Sort", icon='SORT_ASC')
        op_sort.palette_type = pal_tab

        fill_row = layout.row(align=True)
        op_fill = fill_row.operator("voxel.fill_interior", text="Fill Interior", icon='ADD')
        op_fill.palette_type = pal_tab

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

    selected = get_palette_selection(palette_props, pal_tab)
    selected_count = len(selected)
    if selected_count > 0:
        sel_row = layout.row(align=True)
        sel_row.label(text=f"Selected: {selected_count}", icon='RESTRICT_SELECT_OFF')
        clear_op = sel_row.operator("voxel.clear_palette_selection", text="", icon='X')
        clear_op.palette_type = pal_tab
        if selected_count >= 2:
            merge_op = sel_row.operator("voxel.merge_palette_colors", text="Merge to…", icon='MODIFIER')
            merge_op.palette_type = pal_tab

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
        is_paint = (idx == active_index)
        is_selected = idx in selected
        # Empty selection: invert still marks the paint color (existing look).
        # Once chips are multi-selected, invert marks the merge set.
        is_active = is_selected or (is_paint and selected_count == 0)
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
            depress=is_paint,
        )
        op.palette_type = pal_tab
        op.index = idx

    # Active material preview sits flush under the last chip row. Keep the
    # native widget used by Blender's Material Properties tab so the preview
    # matches its rendered material ball exactly. A custom template_icon may
    # be large, but it is still only an approximation of the node graph.
    active_entry = next((e for e in target_palette if e.index == active_index), None)
    if active_entry is not None:
        material = getattr(active_entry, "material", None)
        if material is not None:
            try:
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

    # Mirror tools in one separated space at the bottom of the panel:
    # Active Mirror (live brush symmetry) + Instant Mirror (one-shot copy).
    scene_props = getattr(context.scene, "voxel_workspace", None)
    if scene_props is not None:
        mirror_box = layout.box()
        mirror_box.label(text="Mirror", icon='MOD_MIRROR')

        active_col = mirror_box.column(align=True)
        active_col.label(text="Active Mirror:")
        axis_row = active_col.row(align=True)
        axis_row.label(text="Axes:")
        axis_row.prop(scene_props, "mirror_live_x", toggle=True)
        axis_row.prop(scene_props, "mirror_live_y", toggle=True)
        axis_row.prop(scene_props, "mirror_live_z", toggle=True)

        instant_col = mirror_box.column(align=True)
        instant_col.separator()
        instant_col.label(text="Instant Mirror:")
        instant_col.prop(scene_props, "mirror_axis", expand=True)
        instant_col.prop(scene_props, "mirror_paint_only")
        btn_row = instant_col.row(align=True)
        op = btn_row.operator("voxel.mirror", text="+ -> -")
        op.direction = "POS_TO_NEG"
        op = btn_row.operator("voxel.mirror", text="- -> +")
        op.direction = "NEG_TO_POS"


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

    # 2. GLB analysis and create/import workflow. This is independent of an
    # active volume so Maximum Axis remains the default first-class path.
    glb_box = layout.box()
    glb_box.label(text="GLB Import", icon="IMPORT")
    analysis = get_glb_analysis(scene)
    analyze_row = glb_box.row(align=True)
    analyze_row.operator("voxel.analyze_glb", text="Analyze GLB...", icon="FILEBROWSER")
    if analysis is not None:
        analyze_row.label(text=analysis.filepath.rsplit("\\", 1)[-1].rsplit("/", 1)[-1])
        if len(analysis.objects) == 1:
            item = analysis.objects[0]
            glb_box.label(
                text=f"Mesh: {item.name} — {item.triangle_count:,} triangles",
                icon="OUTLINER_OB_MESH",
            )
        else:
            objects_box = glb_box.box()
            objects_box.label(text="Detected Meshes", icon="OUTLINER_OB_MESH")
            for object_index, item in enumerate(analysis.objects):
                row = objects_box.row(align=True)
                toggle = row.operator(
                    "voxel.toggle_glb_object",
                    text="",
                    icon="CHECKBOX_HLT" if item.included else "CHECKBOX_DEHLT",
                    depress=item.included,
                )
                toggle.object_index = object_index
                primary = " (Primary)" if item.primary else ""
                row.label(text=f"{item.name}{primary}")
                row.label(text=f"{item.triangle_count:,} tris")

        glb_box.prop(sc_props, "glb_target_mode")
        if sc_props.glb_target_mode == "MAX_AXIS":
            glb_box.prop(sc_props, "glb_quality", expand=True)
            rec = analysis.recommendations
            quality_max = {
                "DRAFT": rec.draft,
                "BALANCED": rec.balanced,
                "FINE": rec.fine,
                "CUSTOM": int(sc_props.glb_custom_max_axis),
            }[sc_props.glb_quality]
            if sc_props.glb_quality == "CUSTOM":
                glb_box.prop(sc_props, "glb_custom_max_axis")
            try:
                suggested_dims = volume_dimensions(
                    analysis.dimensions, quality_max, int(sc_props.glb_padding)
                )
                cells = suggested_dims[0] * suggested_dims[1] * suggested_dims[2]
                glb_box.label(
                    text=f"Target: {suggested_dims[0]} × {suggested_dims[1]} × {suggested_dims[2]} ({cells:,} cells)",
                    icon="INFO",
                )
            except ValueError as exc:
                glb_box.label(text=str(exc), icon="ERROR")
            preset_row = glb_box.row(align=True)
            preset_row.label(text=f"Draft {rec.draft}")
            preset_row.label(text=f"Balanced {rec.balanced}")
            preset_row.label(text=f"Fine {rec.fine}")
        elif sc_props.glb_target_mode == "PANEL_DIMENSIONS":
            glb_box.label(
                text=f"Target: {sc_props.create_size_x} × {sc_props.create_size_y} × {sc_props.create_size_z}",
                icon="INFO",
            )
        else:
            existing_ctx = resolve_volume_context(context)
            if existing_ctx is None or existing_ctx.mesh is None:
                glb_box.label(text="Select a voxel volume first", icon="ERROR")
            else:
                glb_box.prop(sc_props, "glb_clear_and_replace")

        glb_box.prop(sc_props, "glb_padding")
        if sc_props.glb_target_mode != "EXISTING":
            glb_box.prop(sc_props, "glb_override_voxel_size")
            if sc_props.glb_override_voxel_size:
                glb_box.prop(sc_props, "glb_voxel_size")
            else:
                glb_box.label(text="Physical dimensions preserved", icon="CHECKMARK")
        options = glb_box.column(align=True)
        options.prop(sc_props, "glb_occupancy")
        palette_box = glb_box.box()
        palette_box.label(text="Palette Detail", icon="COLOR")
        palette_box.prop(sc_props, "glb_palette_quality", expand=True)
        pal_rec = analysis.palette_recommendations
        if sc_props.glb_palette_quality == "CUSTOM":
            palette_box.prop(sc_props, "glb_palette_size")
        palette_presets = palette_box.row(align=True)
        palette_presets.label(text=f"Draft {pal_rec.draft}")
        palette_presets.label(text=f"Balanced {pal_rec.balanced}")
        palette_presets.label(text=f"Fine {pal_rec.fine}")
        options.prop(sc_props, "glb_alpha_cutoff")
        options.prop(sc_props, "glb_keep_source")

        import_row = glb_box.row()
        import_row.scale_y = 1.25
        import_row.enabled = any(item.included for item in analysis.objects)
        op = import_row.operator(
            "voxel.import_glb",
            text="Import into Selected Volume" if sc_props.glb_target_mode == "EXISTING" else "Create and Import GLB",
            icon="IMPORT",
        )
        op.filepath = analysis.filepath
        op.target_mode = sc_props.glb_target_mode
        op.quality = sc_props.glb_quality
        op.maximum_axis = sc_props.glb_custom_max_axis
        op.padding = sc_props.glb_padding
        op.override_voxel_size = sc_props.glb_override_voxel_size
        op.voxel_size = sc_props.glb_voxel_size
        op.occupancy = sc_props.glb_occupancy
        op.palette_quality = sc_props.glb_palette_quality
        op.palette_size = sc_props.glb_palette_size
        op.alpha_cutoff = sc_props.glb_alpha_cutoff
        op.keep_source = sc_props.glb_keep_source
        op.clear_and_replace = sc_props.glb_clear_and_replace
    else:
        glb_box.label(text="Analyze a source file to choose geometry and detail", icon="INFO")

    # 3. Active Volume Context & Inspection
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

        root_props = getattr(v_ctx.root, "voxel_workspace", None) if v_ctx.root is not None else None
        if root_props is not None:
            edge_box = vol_box.box()
            edge_box.label(text="Rendered Surface Edges", icon="MOD_BEVEL")
            edge_box.prop(root_props, "show_rendered_surface_edges", text="Show in Final Render")
            width_row = edge_box.row()
            width_row.enabled = bool(root_props.show_rendered_surface_edges)
            width_row.prop(root_props, "rendered_surface_edge_width", text="Width")
            color_row = edge_box.row()
            color_row.enabled = bool(root_props.show_rendered_surface_edges)
            color_row.prop(root_props, "rendered_surface_edge_color", text="Line Color")

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
        # Resize section: only meaningful when a volume exists
        resize_box = vol_box.box()
        resize_box.label(text="Resize Volume", icon="FULLSCREEN_ENTER")
        resize_box.label(text="Drag an axis arrow at any corner. Existing voxels cannot be removed.", icon="INFO")
        adjusting = is_adjustment_active()
        resize_box.operator("voxel.adjust_voxel_root", text="Adjust voxel root size", icon="ARROW_LEFTRIGHT")
        if adjusting:
            action_row = resize_box.row(align=True)
            action_row.scale_y = 1.25
            action_row.operator(
                "voxel.accept_adjust_voxel_root",
                text="Accept",
                icon="CHECKMARK",
                depress=True,
            )
            cancel_row = action_row.row(align=True)
            cancel_row.alert = True
            cancel_row.operator("voxel.cancel_adjust_voxel_root", text="Cancel", icon="CANCEL")

        # Stretch/Squash section: rescales the voxels inside the root.
        # Sits below the Resize Volume box, as its own section (not nested).
        scale_box = vol_box.box()
        scale_box.label(text="Stretch / Squash Interior", icon="FULLSCREEN_ENTER")
        scale_box.label(text="Drag an arrow to scale one axis. Voxels stretch (filled) or squash with it.", icon="INFO")
        scaling = is_scaling_active()
        scale_box.operator("voxel.scale_voxels", text="Stretch / squash interior voxels", icon="MOD_SMOOTH")
        if scaling:
            s_action_row = scale_box.row(align=True)
            s_action_row.scale_y = 1.25
            s_action_row.operator(
                "voxel.accept_scale_voxels",
                text="Accept",
                icon="CHECKMARK",
                depress=True,
            )
            s_cancel_row = s_action_row.row(align=True)
            s_cancel_row.alert = True
            s_cancel_row.operator("voxel.cancel_scale_voxels", text="Cancel", icon="CANCEL")

        export_row = vol_box.row()
        export_row.scale_y = 1.2
        export_row.operator("voxel.export_slices", text="Export Voxel Slices", icon="EXPORT")
        obj_export_row = vol_box.row()
        obj_export_row.scale_y = 1.2
        obj_export_row.enabled = not empty
        obj_export_row.operator("voxel.export_obj", text="Export Exact Voxel-Lined OBJ", icon="EXPORT")

    else:
        vol_box.label(text="No active voxel volume", icon="INFO")

    # Editing display: edge visibility and edge color mode. These are scene
    # settings, so they render even without an active volume selected.
    if scene is not None and hasattr(scene, "voxel_workspace"):
        sc_props_edge = scene.voxel_workspace
        edge_box = layout.box()
        edge_box.label(text="Editing Display", icon="MOD_EDGESPLIT")
        edge_col = edge_box.column(align=True)
        edge_col.prop(sc_props_edge, "show_voxel_edges")
        edge_col.prop(sc_props_edge, "voxel_edge_auto_contrast")
        color_row = edge_col.row()
        color_row.enabled = not bool(sc_props_edge.voxel_edge_auto_contrast)
        color_row.prop(sc_props_edge, "voxel_edge_color")


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


def _draw_brush_shapes_settings(layout: Any, context: Any) -> None:
    """Brushes/Shapes tab: brush footprint shape and size for add/erase."""
    if context is None or bpy is None:
        return
    scene = getattr(context, "scene", None)
    props = getattr(scene, "voxel_workspace", None)
    if props is None:
        layout.label(text="No voxel scene properties", icon='ERROR')
        return

    box = layout.box()
    box.label(text="Brush Footprint", icon='BRUSHES_ALL')
    row = box.row(align=True)
    row.prop(props, "brush_shape", expand=True)
    col = box.column(align=True)
    col.prop(props, "brush_diameter", text="Size (voxels)")
    active_tool = str(getattr(props, "active_tool", "NONE")).upper()
    if active_tool in {"ADD_SURFACE", "ADD_VOLUME", "ERASE", "PLACE"}:
        from ..core.brush_shapes import radius_from_diameter
        d = int(props.brush_diameter)
        if d <= 1:
            hint = "Single voxel"
        else:
            width = 2 * radius_from_diameter(d) - 1
            kind = "cube" if str(props.brush_shape).upper() == "CUBE" else "sphere"
            hint = f"{width}×{width}×{width} {kind} footprint"
        col.label(text=hint, icon='INFO')


class VOXEL_PT_brush_shapes_panel(Panel):
    """Brushes/Shapes N-panel tab."""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Brushes/Shapes'
    bl_label = 'Brushes / Shapes'
    bl_order = 5

    def draw(self, context: Any) -> None:
        _draw_brush_shapes_settings(self.layout, context)


PANEL_CLASSES = [
    VOXEL_PT_main_panel,
    VOXEL_PT_palette_panel,
    VOXEL_PT_brush_shapes_panel,
]
