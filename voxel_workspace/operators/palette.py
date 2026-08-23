"""Operators for volume palette management: select, edit, add, duplicate, and remove/remap."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import bpy
    from bpy.props import EnumProperty, FloatVectorProperty, IntProperty, StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    EnumProperty = FloatVectorProperty = IntProperty = StringProperty = None

from ..constants import BrickCoord, DEFAULT_PALETTE
from ..core.coords import split_coord
from ..core.presets import (
    PalettePreset,
    PalettePresetEntry,
    PRESET_SCHEMA_VERSION,
    rgba_linear_to_srgb_bytes,
    rgba_srgb_bytes_to_linear,
    find_nearest_palette_index,
    BUILTIN_PRESETS,
)
from ..blender.runtime import get_volume, get_or_load, tag_redraw_all_viewports
from ..blender.object_graph import resolve_volume_context, resolve_authoritative_mesh, resolve_surface_object
from ..blender.persistence import serialize_volume, commit_volume_state
from ..blender.gpu_preview import drop_palette_lut, update_volume_gpu_preview
from ..geometry.visible_faces import mesh_visible_faces


def remap_volume_palette_indices(
    mesh: Any,
    remap_table: Dict[int, int],
    palette_type: str = "SURFACE",
    push_undo: bool = True,
    undo_message: str = "Remap Palette",
) -> int:
    """Remap voxel indices in a volume simultaneously using remap_table: {src_index: dst_index}.
    
    If grid is TaggedVoxelGrid, only remaps cells matching the specified palette_type (domain).
    Scans all occupied bricks, performs simultaneous vectorized lookup replacement,
    serializes to mesh IDProperties, synchronizes mesh and GPU preview, and optionally pushes exactly ONE undo step.
    Returns total count of remapped voxels.
    """
    if mesh is None or not remap_table:
        return 0

    entry = get_or_load(mesh)
    if entry is None or entry.grid is None:
        return 0

    grid = entry.grid
    is_tagged = hasattr(grid, "remap_indices")

    if is_tagged:
        from ..core.tagged_grid import VoxelDomain
        domain_enum = VoxelDomain.VOLUME if palette_type.upper() == "VOLUME" else VoxelDomain.SURFACE
        counts_before = grid.count_indices(domain_enum)
        total_remapped = 0
        for src, dst in remap_table.items():
            if src in counts_before and src != dst:
                total_remapped += counts_before[src]
        grid.remap_indices(domain_enum, remap_table)
        changed_coords = set(grid.dirty_bricks)
        for c in changed_coords:
            entry.dirty_bricks.add(c)
    else:
        # Build 256-element simultaneous lookup array
        lookup = np.arange(256, dtype=np.uint8)
        has_change = False
        for src_idx, dst_idx in remap_table.items():
            if 0 <= src_idx <= 255 and 0 <= dst_idx <= 255 and src_idx != dst_idx:
                lookup[src_idx] = dst_idx
                has_change = True

        if not has_change:
            return 0

        total_remapped = 0
        changed_coords: Set[BrickCoord] = set()

        for coord, brick in list(grid.bricks.items()):
            if not np.any(brick):
                continue
            new_brick = lookup[brick]
            diff_mask = (new_brick != brick)
            count = int(np.count_nonzero(diff_mask))
            if count > 0:
                brick[:] = new_brick
                total_remapped += count
                changed_coords.add(coord)
                grid.dirty_bricks.add(coord)
                entry.dirty_bricks.add(coord)
                if not np.any(brick):
                    del grid.bricks[coord]

    if total_remapped > 0:
        serialize_volume(mesh, grid, dirty_only=True)
        from ..blender.mesh_sync import sync_volume_mesh
        sync_volume_mesh(
            mesh,
            grid=grid,
            dirty_only=True,
            dirty_bricks=changed_coords,
            voxel_size=entry.voxel_size,
            mesher=mesh_visible_faces,
        )
        update_volume_gpu_preview(entry, dirty_only=True, dirty_bricks=changed_coords)
        tag_redraw_all_viewports()

    if push_undo and bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass

    return total_remapped


def get_used_palette_counts(mesh: Any, palette_type: str = "SURFACE") -> Dict[int, int]:
    """Calculate the count of voxels in the volume for each palette index in the given domain."""
    counts: Dict[int, int] = {}
    if mesh is None:
        return counts

    entry = get_or_load(mesh)
    if entry is None or entry.grid is None:
        return counts

    grid = entry.grid
    if hasattr(grid, "count_indices"):
        from ..core.tagged_grid import VoxelDomain
        domain_enum = VoxelDomain.VOLUME if palette_type.upper() == "VOLUME" else VoxelDomain.SURFACE
        return grid.count_indices(domain_enum)

    for brick in grid.bricks.values():
        if not np.any(brick):
            continue
        unq, unq_counts = np.unique(brick, return_counts=True)
        for val, count in zip(unq, unq_counts):
            idx = int(val)
            if idx != 0:
                counts[idx] = counts.get(idx, 0) + int(count)

    return counts


def reconcile_native_render(mesh: Any) -> None:
    """Fully reconcile native surface slots, proxy domains, and GPU preview."""
    entry = get_or_load(mesh)
    if entry is None or entry.grid is None:
        return
    entry.cpu_buffers.clear()
    entry.volume_proxy_buffers.clear()
    entry.gpu_batches.clear()
    entry.gpu_edge_batches.clear()
    entry.palette_lut = None
    from ..blender.mesh_sync import sync_volume_mesh
    sync_volume_mesh(mesh, grid=entry.grid, entry=entry, dirty_only=False, ensure_material=False)
    from ..blender.volume_proxy import reconcile_all_root_instances
    reconcile_all_root_instances(mesh, entry.grid, volume_entry=entry, dirty_bricks=None)
    drop_palette_lut(mesh.voxel_workspace.uuid)
    update_volume_gpu_preview(entry, dirty_only=False)


_SWATCH_SYNC_GUARD = False


def apply_active_palette_index(context: Any, pal_type: str, index: int) -> None:
    """Set the active typed palette index and keep the swatch list highlight in sync."""
    global _SWATCH_SYNC_GUARD
    pal_type = (pal_type or "SURFACE").upper()
    scene = getattr(context, "scene", None) if context is not None else None
    if scene is None or not hasattr(scene, "voxel_workspace"):
        return
    props = scene.voxel_workspace
    if pal_type == "VOLUME":
        props.active_volume_palette_index = index
    else:
        props.active_surface_palette_index = index
        props.active_palette_index = index
        if 1 <= index <= 8:
            try:
                props.active_palette_choice = str(index)
            except Exception:
                pass

    v_ctx = resolve_volume_context(context)
    if v_ctx and v_ctx.mesh and pal_type == "SURFACE":
        from ..blender.material_domains import used_surface_indices
        mesh = v_ctx.mesh
        entry = get_or_load(mesh)
        if entry and entry.grid:
            surf_indices = used_surface_indices(mesh, entry.grid)
            if index in surf_indices and v_ctx.surface_object:
                v_ctx.surface_object.active_material_index = surf_indices.index(index)

    if v_ctx is not None and v_ctx.mesh is not None and not _SWATCH_SYNC_GUARD:
        from ..blender.material_domains import get_palette
        palette = get_palette(v_ctx.mesh, pal_type)
        for list_index, entry_item in enumerate(palette):
            if int(entry_item.index) != int(index):
                continue
            current = (
                props.volume_swatch_list_index
                if pal_type == "VOLUME"
                else props.surface_swatch_list_index
            )
            if current != list_index:
                _SWATCH_SYNC_GUARD = True
                try:
                    if pal_type == "VOLUME":
                        props.volume_swatch_list_index = list_index
                    else:
                        props.surface_swatch_list_index = list_index
                finally:
                    _SWATCH_SYNC_GUARD = False
            break

    tag_redraw_all_viewports()


def apply_swatch_list_selection(scene_props: Any, context: Any, pal_type: str) -> None:
    """Map a UIList collection index to the active palette index."""
    if _SWATCH_SYNC_GUARD or scene_props is None or context is None:
        return
    pal_type = (pal_type or "SURFACE").upper()
    v_ctx = resolve_volume_context(context)
    if v_ctx is None or v_ctx.mesh is None:
        return
    from ..blender.material_domains import get_palette
    palette = get_palette(v_ctx.mesh, pal_type)
    list_index = (
        scene_props.volume_swatch_list_index
        if pal_type == "VOLUME"
        else scene_props.surface_swatch_list_index
    )
    if list_index < 0 or list_index >= len(palette):
        return
    index = int(palette[list_index].index)
    if index <= 0:
        return
    apply_active_palette_index(context, pal_type, index)


class VOXEL_OT_select_palette_tab(Operator):
    """Switch typed palette and activate its matching placement tool."""
    bl_idname = "voxel.select_palette_tab"
    bl_label = "Select Voxel Palette"
    bl_description = "Switch palette and start placing voxels of that type"
    bl_options = {'INTERNAL'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Use the Surface Palette and Add Surface tool"),
                ("VOLUME", "Volume", "Use the Volume Palette and Add Volume tool"),
            ],
            default="SURFACE",
        )

    def execute(self, context):
        if bpy is None or context is None or context.scene is None:
            return {'CANCELLED'}
        props = getattr(context.scene, "voxel_workspace", None)
        if props is None:
            return {'CANCELLED'}
        pal_type = str(getattr(self, "palette_type", "SURFACE")).upper()
        props.active_palette_tab = pal_type
        desired_tool = "ADD_VOLUME" if pal_type == "VOLUME" else "ADD_SURFACE"
        if props.active_tool != desired_tool:
            op = bpy.ops.voxel.start_volume if pal_type == "VOLUME" else bpy.ops.voxel.start_surface
            try:
                op()
            except Exception:
                # Palette switching remains valid when no voxel field is active.
                pass
        tag_redraw_all_viewports()
        return {'FINISHED'}


class VOXEL_OT_select_palette_color(Operator):
    """Set the active placement color index."""
    bl_idname = "voxel.select_palette_color"
    bl_label = "Select Color"
    bl_description = "Set active color for placing voxels"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        index: IntProperty(
            name="Index",
            description="Palette index to select",
            default=1,
            min=1,
            max=255,
        )

    def execute(self, context):
        apply_active_palette_index(
            context,
            getattr(self, "palette_type", "SURFACE"),
            int(self.index),
        )
        return {'FINISHED'}


class VOXEL_OT_edit_palette_material(Operator):
    """Select a palette index and edit its native Material binding."""
    bl_idname = "voxel.edit_palette_material"
    bl_label = "Edit Palette Material"
    bl_options = {'REGISTER'}

    def _material_items(_self, _context):
        items = [("__NEW__", "New Default Material", "Create a new owned default for this domain")]
        if bpy is not None:
            for idx, material in enumerate(bpy.data.materials, start=1):
                items.append((material.name, material.name, "Use existing Blender Material", idx))
        return items

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        index: IntProperty(name="Index", default=1, min=1, max=255)
        material_choice: EnumProperty(name="Material", items=_material_items)

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            return {'CANCELLED'}
        mesh = v_ctx.mesh
        from ..blender.material_domains import find_entry
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        entry = find_entry(mesh, pal_type, self.index)
        if entry is None and hasattr(mesh.voxel_workspace, "palette"):
            entry = next((item for item in mesh.voxel_workspace.palette if item.index == self.index), None)
        if entry is None:
            return {'CANCELLED'}
        if pal_type == "VOLUME":
            context.scene.voxel_workspace.active_volume_palette_index = self.index
        else:
            context.scene.voxel_workspace.active_surface_palette_index = self.index
            context.scene.voxel_workspace.active_palette_index = self.index
        self.material_choice = entry.material.name if entry.material is not None else "__NEW__"
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        pal_type = getattr(self, "palette_type", "SURFACE").title()
        layout.label(text=f"{pal_type} Palette Index {self.index}", icon='MATERIAL')
        layout.prop(self, "material_choice")
        layout.label(text="Use Material Properties or Shader Editor for the full node graph.", icon='INFO')

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            return {'CANCELLED'}
        mesh = v_ctx.mesh
        from ..blender.material_domains import find_entry
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        entry = find_entry(mesh, pal_type, self.index)
        if entry is None and hasattr(mesh.voxel_workspace, "palette"):
            entry = next((item for item in mesh.voxel_workspace.palette if item.index == self.index), None)
        if entry is None:
            return {'CANCELLED'}

        old_material = entry.material
        old_owned = bool(entry.material_owned)

        try:
            if self.material_choice == "__NEW__":
                from ..blender.material_domains import create_default_surface_material, create_default_volume_material
                if pal_type == "VOLUME":
                    entry.material = create_default_volume_material(mesh, entry, volume_color=tuple(entry.color))
                else:
                    entry.material = create_default_surface_material(mesh, entry, base_color=tuple(entry.color))
                entry.material_owned = True
            else:
                from ..blender.material_domains import assign_external_material
                chosen = bpy.data.materials.get(self.material_choice)
                if chosen is None:
                    self.report({'ERROR'}, f"Material '{self.material_choice}' not found")
                    return {'CANCELLED'}
                assign_external_material(entry, chosen)

            reconcile_native_render(mesh)
        except Exception as exc:
            entry.material = old_material
            entry.material_owned = old_owned
            try:
                reconcile_native_render(mesh)
            except Exception:
                pass
            self.report({'ERROR'}, f"Failed to assign material: {exc}")
            return {'CANCELLED'}

        if old_owned and old_material is not None and old_material != entry.material:
            from ..blender.material_domains import cleanup_owned_materials
            cleanup_owned_materials([old_material])

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Edit Palette Material")
            except Exception:
                pass
        return {'FINISHED'}


class VOXEL_OT_sync_display_to_material_color(Operator):
    """Apply the palette display color to the native material's Principled BSDF Base Color."""
    bl_idname = "voxel.sync_display_to_material"
    bl_label = "Apply Display Color to Material"
    bl_description = "Set the Principled BSDF Base Color from this palette entry's display color"
    bl_options = {'REGISTER', 'UNDO'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        index: IntProperty(name="Index", default=1, min=1, max=255)

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        from ..blender.material_domains import find_entry
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        entry = find_entry(v_ctx.mesh, pal_type, self.index)
        if entry is None or entry.material is None:
            self.report({'WARNING'}, "No material bound to this palette entry")
            return {'CANCELLED'}

        from ..blender.material_domains import set_generated_surface_base_color, set_generated_volume_color
        success = set_generated_volume_color(entry) if pal_type == "VOLUME" else set_generated_surface_base_color(entry)
        if not success:
            self.report({'INFO'}, f"Material does not have a recognized Principled {pal_type.title()} color input")
        else:
            self.report({'INFO'}, "Updated Material Base Color from palette display color")
        return {'FINISHED'}


class VOXEL_OT_sync_material_to_display_color(Operator):
    """Read the native material's Principled BSDF Base Color into the palette display color."""
    bl_idname = "voxel.sync_material_to_display"
    bl_label = "Set Display Color from Material"
    bl_description = "Read the Principled BSDF Base Color from the material into this palette entry's display color"
    bl_options = {'REGISTER', 'UNDO'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        index: IntProperty(name="Index", default=1, min=1, max=255)

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        from ..blender.material_domains import find_entry
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        entry = find_entry(v_ctx.mesh, pal_type, self.index)
        if entry is None or entry.material is None or not entry.material.use_nodes:
            self.report({'WARNING'}, "No node material bound to this palette entry")
            return {'CANCELLED'}

        if pal_type == "VOLUME":
            node = next((n for n in entry.material.node_tree.nodes if n.bl_idname == "ShaderNodeVolumePrincipled"), None)
            socket_name = "Color"
        else:
            node = entry.material.node_tree.nodes.get("Principled BSDF")
            socket_name = "Base Color"
        if node and socket_name in node.inputs:
            col = node.inputs[socket_name].default_value
            alpha = node.inputs["Alpha"].default_value if "Alpha" in node.inputs else 1.0
            entry.color = (float(col[0]), float(col[1]), float(col[2]), float(alpha))
            from ..blender.gpu_preview import drop_palette_lut
            drop_palette_lut(v_ctx.mesh_uuid, pal_type)
            tag_redraw_all_viewports()
            self.report({'INFO'}, "Updated palette display color from Material Base Color")
        else:
            self.report({'INFO'}, "Material does not have a recognizable Principled BSDF Base Color")

        return {'FINISHED'}


class VOXEL_OT_make_material_single_user(Operator):
    """Make the active palette entry's material single-user (owned by this volume)."""
    bl_idname = "voxel.make_material_single_user"
    bl_label = "Make Material Single User"
    bl_description = "Create an independent owned copy of this material for this volume"
    bl_options = {'REGISTER', 'UNDO'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[("SURFACE", "Surface", "Surface Palette"), ("VOLUME", "Volume", "Volume Palette")],
            default="SURFACE",
        )
        index: IntProperty(name="Index", default=1, min=1, max=255)

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        from ..blender.material_domains import find_entry
        pal_type = self.palette_type.upper()
        entry = find_entry(mesh, pal_type, self.index)
        if entry is None:
            return {'CANCELLED'}

        from ..blender.material_domains import make_entry_material_single_user
        make_entry_material_single_user(mesh, entry)
        reconcile_native_render(mesh)

        tag_redraw_all_viewports()
        self.report({'INFO'}, f"Material for {pal_type.title()} [{self.index}] is now single-user")
        return {'FINISHED'}


class VOXEL_OT_add_palette_color(Operator):
    """Add a new custom color to the active volume's palette."""
    bl_idname = "voxel.add_palette_color"
    bl_label = "Add Color"
    bl_description = "Allocate the lowest unused palette index and make it active"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        color: FloatVectorProperty(
            name="Initial Color",
            description="Initial linear RGBA color",
            size=4,
            subtype='COLOR',
            default=(0.8, 0.8, 0.8, 1.0),
            min=0.0,
            max=1.0,
        )
        name: StringProperty(
            name="Color Name",
            description="Optional name for the color entry",
            default="",
        )

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        from ..blender.properties import ensure_palette
        from ..blender.material_domains import get_palette, initialize_surface_entry, initialize_volume_entry
        ensure_palette(mesh)

        target_palette = get_palette(mesh, pal_type)
        existing_indices = {entry.index for entry in target_palette}
        new_index = None
        for i in range(1, 256):
            if i not in existing_indices:
                new_index = i
                break

        if new_index is None:
            self.report({'ERROR'}, f"{pal_type.title()} Palette is full (maximum 255 colors)")
            return {'CANCELLED'}

        item = target_palette.add()
        if pal_type == "VOLUME":
            initialize_volume_entry(mesh, item, index=new_index, name=self.name or f"Volume {new_index}", color=tuple(self.color))
        else:
            initialize_surface_entry(mesh, item, index=new_index, name=self.name or f"Color {new_index}", color=tuple(self.color))

        drop_palette_lut(props.uuid)

        # Set as active color in scene
        if pal_type == "VOLUME":
            context.scene.voxel_workspace.active_volume_palette_index = new_index
        else:
            context.scene.voxel_workspace.active_surface_palette_index = new_index
            context.scene.voxel_workspace.active_palette_index = new_index
            if 1 <= new_index <= 8:
                context.scene.voxel_workspace.active_palette_choice = str(new_index)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message=f"Add {pal_type.title()} Color")
            except Exception:
                pass
        return {'FINISHED'}


class VOXEL_OT_duplicate_palette_color(Operator):
    """Duplicate a palette color to a new index."""
    bl_idname = "voxel.duplicate_palette_color"
    bl_label = "Duplicate Color"
    bl_description = "Duplicate this color to a new index so it can be edited independently"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        source_index: IntProperty(
            name="Source Index",
            description="Palette index to duplicate",
            default=1,
            min=1,
            max=255,
        )

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        from ..blender.properties import ensure_palette
        from ..blender.material_domains import get_palette, copy_palette_entry_binding
        ensure_palette(mesh)

        target_palette = get_palette(mesh, pal_type)
        src_entry = None
        for entry in target_palette:
            if entry.index == self.source_index:
                src_entry = entry
                break


        if src_entry is None:
            self.report({'ERROR'}, f"Color index {self.source_index} not found in {pal_type.lower()} palette")
            return {'CANCELLED'}

        existing_indices = {entry.index for entry in target_palette}
        new_index = None
        for i in range(1, 256):
            if i not in existing_indices:
                new_index = i
                break

        if new_index is None:
            self.report({'ERROR'}, f"{pal_type.title()} Palette is full (maximum 255 colors)")
            return {'CANCELLED'}

        item = target_palette.add()
        copy_palette_entry_binding(src_entry, item, mesh.voxel_workspace.uuid)
        item.index = new_index
        item.name = f"{src_entry.name} (Copy)" if src_entry.name else f"Color {new_index}"

        drop_palette_lut(props.uuid)

        if pal_type == "VOLUME":
            context.scene.voxel_workspace.active_volume_palette_index = new_index
        else:
            context.scene.voxel_workspace.active_surface_palette_index = new_index
            context.scene.voxel_workspace.active_palette_index = new_index
            if 1 <= new_index <= 8:
                context.scene.voxel_workspace.active_palette_choice = str(new_index)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message=f"Duplicate {pal_type.title()} Color")
            except Exception:
                pass
        return {'FINISHED'}


def _replacement_items(self, context):
    """Dynamic EnumProperty items for selecting a replacement palette color."""
    items = []
    # Option 0: Erase / Empty
    items.append(("0", "0: Erase (Empty Space)", "Erase affected voxels to empty space", 0))
    mesh = resolve_authoritative_mesh(context)
    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        from ..blender.material_domains import get_palette
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        target_palette = get_palette(mesh, pal_type)
        for entry in target_palette:
            if entry.index > 0 and entry.index != self.index:
                name_str = f"{entry.index}: {entry.name}" if entry.name else f"{entry.index}: Color {entry.index}"
                items.append((str(entry.index), name_str, f"Remap voxels to index {entry.index}", entry.index))
    return items


def _replacement_choice_updated(self, _context):
    """Keep the execution target synchronized with the visible replacement choice."""
    try:
        self.replacement_index = int(self.replacement_choice)
    except (TypeError, ValueError):
        self.replacement_index = -1


class VOXEL_OT_remove_palette_color(Operator):
    """Remove a palette entry, remapping used voxels if necessary."""
    bl_idname = "voxel.remove_palette_color"
    bl_label = "Remove Color"
    bl_description = "Remove a palette color (remaps voxels if in use)"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        index: IntProperty(
            name="Index to Remove",
            description="Palette index to remove",
            default=1,
            min=1,
            max=255,
        )
        replacement_index: IntProperty(
            name="Replacement Index",
            description="Palette index to remap existing voxels to (0 = Erase, or another valid color)",
            default=-1,
            min=-1,
            max=255,
        )
        replacement_choice: EnumProperty(
            name="Replacement Color",
            description="Choose replacement color for affected voxels",
            items=_replacement_items,
            update=_replacement_choice_updated,
        )

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            return {'CANCELLED'}
        mesh = v_ctx.mesh
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        counts = get_used_palette_counts(mesh, palette_type=pal_type)
        used_count = counts.get(self.index, 0)
        if used_count > 0:
            from ..blender.material_domains import get_palette
            target_palette = get_palette(mesh, pal_type)
            valid_targets = [e.index for e in target_palette if e.index > 0 and e.index != self.index]
            default_target = valid_targets[0] if valid_targets else 0
            self.replacement_choice = str(default_target)
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        mesh = v_ctx.mesh if v_ctx else None
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        counts = get_used_palette_counts(mesh, palette_type=pal_type) if mesh else {}
        used_count = counts.get(self.index, 0)

        layout.label(text=f"{pal_type.title()} Index {self.index} is used by {used_count} voxels.", icon='INFO')
        layout.prop(self, "replacement_choice", text="Replace with")
        if self.replacement_choice == "0":
            layout.label(text="WARNING: Voxels will be permanently erased!", icon='ERROR')

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        if self.index == 0:
            self.report({'ERROR'}, "Index 0 is reserved empty and cannot be removed")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        from ..blender.material_domains import get_palette, cleanup_owned_materials
        target_palette = get_palette(mesh, pal_type)

        if self.replacement_index == -1:
            self.replacement_index = 0

        target_item_pos = None
        for pos, entry in enumerate(target_palette):
            if entry.index == self.index:
                target_item_pos = pos
                break


        if target_item_pos is None:
            self.report({'WARNING'}, f"Color index {self.index} not in {pal_type.lower()} palette")
            return {'CANCELLED'}

        removed_material = target_palette[target_item_pos].material if target_item_pos < len(target_palette) else None

        counts = get_used_palette_counts(mesh, palette_type=pal_type)
        used_count = counts.get(self.index, 0)

        allocated_indices = {entry.index for entry in target_palette}
        if used_count > 0:
            if self.replacement_index == self.index:
                self.report({'ERROR'}, "Replacement index cannot be the same as the index being removed")
                return {'CANCELLED'}
            if self.replacement_index != 0 and self.replacement_index not in allocated_indices:
                self.report({'ERROR'}, f"Replacement index {self.replacement_index} is not allocated in palette")
                return {'CANCELLED'}

            remap_volume_palette_indices(
                mesh,
                {self.index: self.replacement_index},
                palette_type=pal_type,
                push_undo=False,
            )

        # Remove from typed palette
        if target_item_pos < len(target_palette):
            target_palette.remove(target_item_pos)

        from ..blender.persistence import serialize_volume
        entry = get_or_load(mesh)
        if entry is not None and entry.grid is not None:
            serialize_volume(mesh, entry.grid, dirty_only=False)

        reconcile_native_render(mesh)
        from ..blender.material_domains import cleanup_owned_materials
        if removed_material is not None:
            cleanup_owned_materials([removed_material])

        scene = context.scene
        if pal_type == "VOLUME":
            if scene.voxel_workspace.active_volume_palette_index == self.index:
                rem_choices = [e.index for e in target_palette if e.index > 0]
                scene.voxel_workspace.active_volume_palette_index = rem_choices[0] if rem_choices else 1
        else:
            if scene.voxel_workspace.active_surface_palette_index == self.index or scene.voxel_workspace.active_palette_index == self.index:
                rem_choices = [e.index for e in target_palette if e.index > 0]
                new_act = rem_choices[0] if rem_choices else 1
                scene.voxel_workspace.active_surface_palette_index = new_act
                scene.voxel_workspace.active_palette_index = new_act
                if 1 <= new_act <= 8:
                    scene.voxel_workspace.active_palette_choice = str(new_act)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message=f"Remove {pal_type.title()} Color")
            except Exception:
                pass
        return {'FINISHED'}


class VOXEL_OT_eyedropper(Operator):
    """Pick color and domain from an occupied voxel in the 3D viewport without leaving edit mode."""
    bl_idname = "voxel.eyedropper"
    bl_label = "Pick Voxel Color"
    bl_description = "Pick palette color from clicked voxel (or press 'I' / Alt+LMB while editing)"
    bl_options = {'REGISTER'}

    def _finish(self, context, result):
        """Restore the normal cursor before terminating the sampling session."""
        context.window.cursor_set('DEFAULT')
        return result

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        entry = get_or_load(v_ctx.mesh)
        if entry is None or entry.grid is None:
            return {'CANCELLED'}

        # Modal sample on click
        if hasattr(context, "window_manager") and context.window_manager is not None:
            try:
                context.window_manager.modal_handler_add(self)
            except Exception:
                pass
        if hasattr(context, "window") and context.window is not None:
            try:
                context.window.cursor_set('EYEDROPPER')
            except Exception:
                pass
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self._finish(context, {'CANCELLED'})

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            context.window.cursor_set('EYEDROPPER')
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            v_ctx = resolve_volume_context(context)
            if v_ctx is not None and v_ctx.mesh is not None:
                entry = get_or_load(v_ctx.mesh)
                if entry is not None and entry.grid is not None:
                    transform_obj = v_ctx.root if v_ctx.root is not None else v_ctx.surface_object
                    # Raycast into viewport
                    try:
                        import bpy_extras.view3d_utils as view3d_utils
                        from ..core.dda import trace_grid
                        from ..core.line import world_ray_to_grid_ray
                        area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
                        if area is not None:
                            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                            rv3d = getattr(area.spaces.active, "region_3d", None)
                            if region is not None and rv3d is not None:
                                rx = event.mouse_x - region.x
                                ry = event.mouse_y - region.y
                                origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, (rx, ry))
                                dir_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, (rx, ry))
                                if origin_world is not None and dir_world is not None and transform_obj is not None:
                                    origin_grid, dir_grid = world_ray_to_grid_ray(
                                        origin_world, dir_world, transform_obj.matrix_world, voxel_size=entry.voxel_size
                                    )
                                    hit = trace_grid(entry.grid, origin_grid, dir_grid, max_distance=1000.0)
                                    if hit is not None:
                                        if hasattr(entry.grid, "get_cell"):
                                            from ..core.tagged_grid import VoxelDomain
                                            cell = entry.grid.get_cell(hit.cell)
                                            if cell.domain == VoxelDomain.VOLUME and cell.index > 0:
                                                context.scene.voxel_workspace.active_volume_palette_index = cell.index
                                                if hasattr(context.scene.voxel_workspace, "active_palette_tab"):
                                                    context.scene.voxel_workspace.active_palette_tab = "VOLUME"
                                            elif cell.index > 0:
                                                context.scene.voxel_workspace.active_surface_palette_index = cell.index
                                                context.scene.voxel_workspace.active_palette_index = cell.index
                                                if hasattr(context.scene.voxel_workspace, "active_palette_tab"):
                                                    context.scene.voxel_workspace.active_palette_tab = "SURFACE"
                                                if 1 <= cell.index <= 8:
                                                    context.scene.voxel_workspace.active_palette_choice = str(cell.index)
                                        else:
                                            picked_index = entry.grid.get(hit.cell)
                                            if picked_index > 0:
                                                context.scene.voxel_workspace.active_surface_palette_index = picked_index
                                                context.scene.voxel_workspace.active_palette_index = picked_index
                                                if 1 <= picked_index <= 8:
                                                    context.scene.voxel_workspace.active_palette_choice = str(picked_index)
                                        tag_redraw_all_viewports()
                                        return self._finish(context, {'FINISHED'})
                    except Exception:
                        pass
            return self._finish(context, {'CANCELLED'})

        return {'PASS_THROUGH'}


class VOXEL_OT_compact_palette(Operator):
    """Remap palette indices to be contiguous 1..N and remove unused entries."""
    bl_idname = "voxel.compact_palette"
    bl_label = "Compact Palette"
    bl_description = "Remap volume palette indices to be contiguous 1..N and purge unused colors"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        mesh = v_ctx.mesh if v_ctx else None
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        counts = get_used_palette_counts(mesh, palette_type=pal_type) if mesh else {}
        used_indices = sorted(counts.keys())
        total_voxels = sum(counts.values())

        layout.label(text=f"{pal_type.title()} Palette: {len(used_indices)} used colors, {total_voxels} voxels.", icon='INFO')
        layout.label(text=f"Used colors will be remapped contiguously 1..{len(used_indices)}.")
        layout.label(text=f"Unused {pal_type.lower()} palette entries will be purged.")

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        from ..blender.material_domains import get_palette, palette_materials, cleanup_owned_materials, initialize_surface_entry, initialize_volume_entry
        target_palette = get_palette(mesh, pal_type)
        counts = get_used_palette_counts(mesh, palette_type=pal_type)

        # Build old -> new remap table for used indices
        used_indices = sorted(counts.keys())
        remap_table: Dict[int, int] = {}
        for new_idx, old_idx in enumerate(used_indices, start=1):
            remap_table[old_idx] = new_idx

        old_materials = palette_materials(mesh, domain=pal_type)

        # Snapshot complete authoritative bindings under their new indices.
        saved_entries = {}
        for entry in target_palette:
            if entry.index in remap_table:
                saved_entries[remap_table[entry.index]] = {
                    "name": str(entry.name),
                    "color": tuple(entry.color),
                    "material": entry.material,
                    "owned": bool(entry.material_owned),
                }

        # Remap occupied bricks for this domain only
        remap_volume_palette_indices(
            mesh,
            remap_table,
            palette_type=pal_type,
            push_undo=False,
        )

        # Rebuild typed palette collection
        target_palette.clear()

        # Reserved index 0
        from ..constants import DEFAULT_PALETTE
        empty_item = target_palette.add()
        empty_item.index = 0
        empty_item.name = "Empty"
        empty_item.color = DEFAULT_PALETTE[0]
        empty_item.material_owned = True

        # Add remapped entries 1..N
        for new_idx in range(1, len(used_indices) + 1):
            item = target_palette.add()
            saved = saved_entries.get(new_idx)
            if saved is None:
                if pal_type == "VOLUME":
                    initialize_volume_entry(mesh, item, index=new_idx, name=f"Volume {new_idx}", color=(0.5, 0.5, 0.5, 1.0))
                else:
                    initialize_surface_entry(mesh, item, index=new_idx, name=f"Color {new_idx}", color=(0.5, 0.5, 0.5, 1.0))
            else:
                item.index = new_idx
                item.name = saved["name"]
                item.color = saved["color"]
                item.material = saved["material"]
                item.material_owned = saved["owned"]

        # If no colors used at all in this domain, ensure defaults
        if len(used_indices) == 0:
            if pal_type == "VOLUME":
                item1 = target_palette.add()
                initialize_volume_entry(mesh, item1, index=1, name="Mist", color=(0.8, 0.8, 0.8, 1.0))
            else:
                from ..blender.properties import ensure_palette
                ensure_palette(mesh)

        # Force flush to Mesh IDProperties so Undo memfile captures the removal
        from ..blender.persistence import serialize_volume
        entry = get_or_load(mesh)
        if entry is not None and entry.grid is not None:
            serialize_volume(mesh, entry.grid, dirty_only=False)

        reconcile_native_render(mesh)
        cleanup_owned_materials(old_materials)

        # Update active color in scene
        scene = context.scene
        if pal_type == "VOLUME":
            old_active = scene.voxel_workspace.active_volume_palette_index
            new_active = remap_table.get(old_active, 1)
            scene.voxel_workspace.active_volume_palette_index = new_active
        else:
            old_active = scene.voxel_workspace.active_surface_palette_index
            new_active = remap_table.get(old_active, 1)
            scene.voxel_workspace.active_surface_palette_index = new_active
            scene.voxel_workspace.active_palette_index = new_active
            if 1 <= new_active <= 8:
                scene.voxel_workspace.active_palette_choice = str(new_active)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message=f"Compact {pal_type.title()} Palette")
            except Exception:
                pass
        return {'FINISHED'}


class VOXEL_OT_save_palette_preset(Operator):
    """Save the active volume's palette to a JSON preset file."""
    bl_idname = "voxel.save_palette_preset"
    bl_label = "Save Palette Preset"
    bl_description = "Save the active volume's palette as a JSON preset file"
    bl_options = {'REGISTER'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        filepath: StringProperty(
            name="File Path",
            description="Destination JSON file path",
            subtype='FILE_PATH',
            default="",
        )
        preset_name: StringProperty(
            name="Preset Name",
            description="Name for the palette preset",
            default="My Palette",
        )

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}
        name_hint = v_ctx.root.name if v_ctx.root else (v_ctx.surface_object.name if v_ctx.surface_object else "Volume")
        pal_type = getattr(self, "palette_type", "SURFACE").title()
        if not self.preset_name or self.preset_name == "My Palette":
            self.preset_name = f"{name_hint} {pal_type} Preset"
        if not self.filepath:
            import tempfile
            self.filepath = str(Path(tempfile.gettempdir()) / f"{self.preset_name.replace(' ', '_').lower()}.json")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        if not self.filepath:
            self.report({'ERROR'}, "No destination file path specified")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        from ..blender.material_domains import get_palette
        target_palette = get_palette(mesh, pal_type)
        color_entries = []

        for entry in sorted([e for e in target_palette if e.index > 0], key=lambda e: e.index):
            color_entries.append(
                PalettePresetEntry(
                    name=entry.name,
                    color_srgb=rgba_linear_to_srgb_bytes(tuple(entry.color)),
                    domain=pal_type,
                )
            )


        preset = PalettePreset(
            name=self.preset_name or "Custom Preset",
            schema_version=PRESET_SCHEMA_VERSION,
            color_space="sRGB",
            colors=color_entries,
            palette_type=pal_type,
        )

        try:
            preset.save_to_file(self.filepath)
            self.report({'INFO'}, f"Saved palette preset ({len(color_entries)} colors) to {self.filepath}")
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to save preset: {exc}")
            return {'CANCELLED'}

        return {'FINISHED'}


def _builtin_preset_enum_items(_self, _context):
    items = []
    for idx, name in enumerate(BUILTIN_PRESETS):
        items.append((name, name, f"Load built-in {name} preset", idx))
    items.append(("FILE", "From JSON File...", "Load a preset from an external JSON file", len(BUILTIN_PRESETS)))
    return items


class VOXEL_OT_load_palette_preset(Operator):
    """Load a palette preset onto the active volume with append or remap mode."""
    bl_idname = "voxel.load_palette_preset"
    bl_label = "Load Palette Preset"
    bl_description = "Load a palette preset (built-in or from JSON file) onto the active volume"
    bl_options = {'REGISTER', 'UNDO'}

    if bpy is not None:
        palette_type: EnumProperty(
            name="Palette Type",
            items=[
                ("SURFACE", "Surface", "Surface Palette"),
                ("VOLUME", "Volume", "Volume Palette"),
            ],
            default="SURFACE",
        )
        preset_source: EnumProperty(
            name="Preset",
            description="Choose a built-in preset or load from file",
            items=_builtin_preset_enum_items,
        )
        filepath: StringProperty(
            name="File Path",
            description="JSON preset file path (when source is FILE)",
            subtype='FILE_PATH',
            default="",
        )
        import_mode: EnumProperty(
            name="Import Mode",
            description="How to handle existing volume colors",
            items=[
                ("REPLACE", "Replace", "Replace palette values starting at index 1 (best for empty volumes)"),
                ("APPEND", "Append Unused", "Keep existing voxel indices; add only preset colors not already present"),
                ("REMAP", "Remap to Nearest", "Recolor existing voxels to the nearest matching preset color"),
            ],
            default="APPEND",
        )

    def invoke(self, context, event):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        counts = get_used_palette_counts(v_ctx.mesh, palette_type=pal_type)
        has_used_voxels = bool(counts)
        self.import_mode = "APPEND" if has_used_voxels else "REPLACE"

        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        counts = get_used_palette_counts(v_ctx.mesh, palette_type=pal_type) if (v_ctx and v_ctx.mesh) else {}
        has_used_voxels = bool(counts)

        layout.label(text=f"Target: {pal_type.title()} Palette", icon='COLOR')
        layout.prop(self, "preset_source")
        if self.preset_source == "FILE":
            layout.prop(self, "filepath")

        if has_used_voxels:
            layout.label(text=f"Volume has {sum(counts.values())} {pal_type.lower()} voxels in use.", icon='INFO')
            layout.prop(self, "import_mode", text="Mode")
            if self.import_mode == "REPLACE":
                layout.label(text="Warning: Indices will be overwritten in-place!", icon='ERROR')
        else:
            layout.label(text=f"Volume is empty. {pal_type.title()} Palette will be loaded starting at index 1.")

    def execute(self, context):
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        # 1. Resolve preset
        if self.preset_source == "FILE":
            if not self.filepath or not Path(self.filepath).exists():
                self.report({'ERROR'}, f"Preset file not found: {self.filepath}")
                return {'CANCELLED'}
            try:
                preset = PalettePreset.from_file(self.filepath)
            except Exception as exc:
                self.report({'ERROR'}, f"Failed to parse preset file: {exc}")
                return {'CANCELLED'}
        else:
            if self.preset_source not in BUILTIN_PRESETS:
                self.report({'ERROR'}, f"Unknown built-in preset: {self.preset_source}")
                return {'CANCELLED'}
            preset = BUILTIN_PRESETS[self.preset_source]

        if not preset.colors:
            self.report({'WARNING'}, "Preset contains no colors")
            return {'CANCELLED'}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        pal_type = getattr(self, "palette_type", "SURFACE").upper()
        if preset.palette_type.upper() != pal_type:
            self.report({'ERROR'}, f"{preset.palette_type.title()} preset cannot be loaded into {pal_type.title()} Palette")
            return {'CANCELLED'}
        from ..blender.material_domains import get_palette, palette_materials, cleanup_owned_materials, initialize_surface_entry, initialize_volume_entry
        from ..blender.properties import ensure_palette

        target_palette = get_palette(mesh, pal_type)
        if len(target_palette) == 0:
            ensure_palette(mesh)

        old_materials = palette_materials(mesh, domain=pal_type)
        counts = get_used_palette_counts(mesh, palette_type=pal_type)
        has_used_voxels = bool(counts)
        mode = self.import_mode if has_used_voxels else "REPLACE"

        # Convert preset colors to linear RGBA
        preset_linear_entries = []
        for c in preset.colors:
            lin_rgba = rgba_srgb_bytes_to_linear(c.color_srgb)
            preset_linear_entries.append((c.name, lin_rgba, getattr(c, "domain", pal_type)))

        if mode == "REPLACE":
            # Clear all non-zero entries in target_palette and assign preset colors starting at index 1
            target_palette.clear()
            # Index 0 empty
            from ..constants import DEFAULT_PALETTE
            empty_item = target_palette.add()
            empty_item.index = 0
            empty_item.name = "Empty"
            empty_item.color = DEFAULT_PALETTE[0]
            empty_item.material_owned = True

            for idx, (c_name, c_rgba, c_dom) in enumerate(preset_linear_entries, start=1):
                if idx > 255:
                    break
                item = target_palette.add()
                if pal_type == "VOLUME":
                    initialize_volume_entry(mesh, item, index=idx, name=c_name or f"Volume {idx}", color=c_rgba)
                else:
                    initialize_surface_entry(mesh, item, index=idx, name=c_name or f"Color {idx}", color=c_rgba)

        elif mode == "APPEND":
            existing_colors = [tuple(e.color) for e in target_palette if e.index > 0]
            existing_indices = {e.index for e in target_palette}

            def is_already_present(target_col):
                tr, tg, tb, _ = target_col
                for er, eg, eb, _ in existing_colors:
                    if (tr - er) ** 2 + (tg - eg) ** 2 + (tb - eb) ** 2 < 1e-4:
                        return True
                return False

            next_idx = 1
            for c_name, c_rgba, c_dom in preset_linear_entries:
                if is_already_present(c_rgba):
                    continue
                # Find lowest unused index
                while next_idx in existing_indices and next_idx <= 255:
                    next_idx += 1
                if next_idx > 255:
                    break
                item = target_palette.add()
                if pal_type == "VOLUME":
                    initialize_volume_entry(mesh, item, index=next_idx, name=c_name or f"Volume {next_idx}", color=c_rgba)
                else:
                    initialize_surface_entry(mesh, item, index=next_idx, name=c_name or f"Color {next_idx}", color=c_rgba)

                existing_indices.add(next_idx)
                existing_colors.append(c_rgba)

        elif mode == "REMAP":
            # Rebuild palette to contain the preset colors, and remap all existing voxels to nearest preset colors
            # 1. Build candidates list from preset (indices 1..N)
            candidates = []
            for idx, (c_name, c_rgba, c_dom) in enumerate(preset_linear_entries, start=1):
                if idx > 255:
                    break
                candidates.append((idx, c_name, c_rgba, c_dom))

            candidate_colors_for_matching = [(idx, c_rgba) for idx, c_name, c_rgba, c_dom in candidates]

            # 2. Build remap table for all used old indices
            remap_table = {}
            for old_idx in counts.keys():
                old_entry = next((e for e in target_palette if e.index == old_idx), None)
                if old_entry is not None:
                    old_rgba = tuple(old_entry.color)
                    nearest_new_idx = find_nearest_palette_index(old_rgba, candidate_colors_for_matching)
                    remap_table[old_idx] = nearest_new_idx

            # 3. Apply remap on voxel grid
            remap_volume_palette_indices(
                mesh,
                remap_table,
                palette_type=pal_type,
                push_undo=False,
            )

            # 4. Replace palette collection with preset colors
            target_palette.clear()
            from ..constants import DEFAULT_PALETTE
            empty_item = target_palette.add()
            empty_item.index = 0
            empty_item.name = "Empty"
            empty_item.color = DEFAULT_PALETTE[0]
            empty_item.material_owned = True

            for idx, c_name, c_rgba, c_dom in candidates:
                item = target_palette.add()
                if pal_type == "VOLUME":
                    initialize_volume_entry(mesh, item, index=idx, name=c_name or f"Volume {idx}", color=c_rgba)
                else:
                    initialize_surface_entry(mesh, item, index=idx, name=c_name or f"Color {idx}", color=c_rgba)

        # Sync caches and IDProperties
        from ..blender.persistence import serialize_volume
        entry = get_or_load(mesh)
        if entry is not None and entry.grid is not None:
            serialize_volume(mesh, entry.grid, dirty_only=False)

        reconcile_native_render(mesh)
        cleanup_owned_materials(old_materials)

        # Set active color to index 1 or first available
        scene = context.scene
        if scene and hasattr(scene, "voxel_workspace"):
            if pal_type == "VOLUME":
                active_available = [e.index for e in target_palette if e.index > 0]
                new_active = active_available[0] if active_available else 1
                scene.voxel_workspace.active_volume_palette_index = new_active
            else:
                active_available = [e.index for e in target_palette if e.index > 0]
                new_active = active_available[0] if active_available else 1
                scene.voxel_workspace.active_surface_palette_index = new_active
                scene.voxel_workspace.active_palette_index = new_active
                if 1 <= new_active <= 8:
                    scene.voxel_workspace.active_palette_choice = str(new_active)

        tag_redraw_all_viewports()
        self.report({'INFO'}, f"Loaded preset '{preset.name}' ({len(preset.colors)} colors) via {mode}")
        return {'FINISHED'}


PALETTE_OPERATOR_CLASSES = [
    VOXEL_OT_select_palette_tab,
    VOXEL_OT_select_palette_color,
    VOXEL_OT_edit_palette_material,
    VOXEL_OT_sync_display_to_material_color,
    VOXEL_OT_sync_material_to_display_color,
    VOXEL_OT_make_material_single_user,
    VOXEL_OT_add_palette_color,
    VOXEL_OT_duplicate_palette_color,
    VOXEL_OT_remove_palette_color,
    VOXEL_OT_eyedropper,
    VOXEL_OT_compact_palette,
    VOXEL_OT_save_palette_preset,
    VOXEL_OT_load_palette_preset,
]
