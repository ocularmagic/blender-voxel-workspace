"""Operators for volume palette management: select, edit, add, duplicate, and remove/remap."""
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

from ..constants import BrickCoord
from ..core.coords import split_coord
from ..blender.runtime import get_volume, get_or_load, tag_redraw_all_viewports
from ..blender.persistence import serialize_volume, commit_volume_state
from ..blender.materials import refresh_palette_image, ensure_voxel_material
from ..blender.gpu_preview import drop_palette_lut, update_volume_gpu_preview
from ..geometry.visible_faces import mesh_visible_faces


def remap_volume_palette_indices(
    mesh: Any,
    remap_table: Dict[int, int],
    push_undo: bool = True,
    undo_message: str = "Remap Palette",
) -> int:
    """Remap voxel indices in a volume simultaneously using remap_table: {src_index: dst_index}.
    
    Scans all occupied bricks, performs simultaneous vectorized lookup replacement (supporting swaps like {1:2, 2:1}),
    serializes to mesh IDProperties, synchronizes mesh and GPU preview, and optionally pushes exactly ONE undo step.
    Returns total count of remapped voxels.
    """
    if mesh is None or not remap_table:
        return 0

    entry = get_or_load(mesh)
    if entry is None or entry.grid is None:
        return 0

    # Build 256-element simultaneous lookup array
    lookup = np.arange(256, dtype=np.uint8)
    has_change = False
    for src_idx, dst_idx in remap_table.items():
        if 0 <= src_idx <= 255 and 0 <= dst_idx <= 255 and src_idx != dst_idx:
            lookup[src_idx] = dst_idx
            has_change = True

    if not has_change:
        return 0

    grid = entry.grid
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


def get_used_palette_counts(mesh: Any) -> Dict[int, int]:
    """Calculate the count of voxels in the volume for each palette index."""
    counts: Dict[int, int] = {}
    if mesh is None:
        return counts

    entry = get_or_load(mesh)
    if entry is None or entry.grid is None:
        return counts

    for brick in entry.grid.bricks.values():
        if not np.any(brick):
            continue
        unq, unq_counts = np.unique(brick, return_counts=True)
        for val, count in zip(unq, unq_counts):
            idx = int(val)
            if idx != 0:
                counts[idx] = counts.get(idx, 0) + int(count)

    return counts


class VOXEL_OT_select_palette_color(Operator):
    """Set the active placement color index."""
    bl_idname = "voxel.select_palette_color"
    bl_label = "Select Color"
    bl_description = "Set active color for placing voxels"
    bl_options = {'REGISTER'}

    if bpy is not None:
        index: IntProperty(
            name="Index",
            description="Palette index to select",
            default=1,
            min=1,
            max=255,
        )

    def execute(self, context):
        scene = context.scene
        if scene is not None and hasattr(scene, "voxel_workspace"):
            scene.voxel_workspace.active_palette_index = self.index
            # Keep active_palette_choice in sync if within 1..8
            if 1 <= self.index <= 8:
                scene.voxel_workspace.active_palette_choice = str(self.index)
        tag_redraw_all_viewports()
        return {'FINISHED'}


class VOXEL_OT_add_palette_color(Operator):
    """Add a new custom color to the active volume's palette."""
    bl_idname = "voxel.add_palette_color"
    bl_label = "Add Color"
    bl_description = "Allocate the lowest unused palette index and make it active"
    bl_options = {'REGISTER'}

    if bpy is not None:
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
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = obj.data
        props = mesh.voxel_workspace
        from ..blender.properties import ensure_palette
        ensure_palette(mesh)

        existing_indices = {entry.index for entry in props.palette}
        # Find lowest unused nonzero index (1..255)
        new_index = None
        for i in range(1, 256):
            if i not in existing_indices:
                new_index = i
                break

        if new_index is None:
            self.report({'ERROR'}, "Palette is full (maximum 255 colors)")
            return {'CANCELLED'}

        item = props.palette.add()
        item.index = new_index
        item.name = self.name or f"Color {new_index}"
        item.color = self.color

        refresh_palette_image(mesh)
        drop_palette_lut(props.uuid)

        # Set as active color in scene
        context.scene.voxel_workspace.active_palette_index = new_index
        if 1 <= new_index <= 8:
            context.scene.voxel_workspace.active_palette_choice = str(new_index)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Add Palette Color")
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
        source_index: IntProperty(
            name="Source Index",
            description="Palette index to duplicate",
            default=1,
            min=1,
            max=255,
        )

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = obj.data
        props = mesh.voxel_workspace
        from ..blender.properties import ensure_palette
        ensure_palette(mesh)

        src_entry = None
        for entry in props.palette:
            if entry.index == self.source_index:
                src_entry = entry
                break

        if src_entry is None:
            self.report({'ERROR'}, f"Color index {self.source_index} not found in palette")
            return {'CANCELLED'}

        existing_indices = {entry.index for entry in props.palette}
        new_index = None
        for i in range(1, 256):
            if i not in existing_indices:
                new_index = i
                break

        if new_index is None:
            self.report({'ERROR'}, "Palette is full (maximum 255 colors)")
            return {'CANCELLED'}

        item = props.palette.add()
        item.index = new_index
        item.name = f"{src_entry.name} (Copy)" if src_entry.name else f"Color {new_index}"
        item.color = src_entry.color

        refresh_palette_image(mesh)
        drop_palette_lut(props.uuid)

        context.scene.voxel_workspace.active_palette_index = new_index
        if 1 <= new_index <= 8:
            context.scene.voxel_workspace.active_palette_choice = str(new_index)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Duplicate Palette Color")
            except Exception:
                pass
        return {'FINISHED'}


def _replacement_items(self, context):
    """Dynamic EnumProperty items for selecting a replacement palette color."""
    items = []
    # Option 0: Erase / Empty
    items.append(("0", "0: Erase (Empty Space)", "Erase affected voxels to empty space", 0))
    obj = context.active_object if context else None
    if obj is not None and hasattr(obj.data, "voxel_workspace"):
        props = obj.data.voxel_workspace
        for entry in props.palette:
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
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            return {'CANCELLED'}
        mesh = obj.data
        counts = get_used_palette_counts(mesh)
        used_count = counts.get(self.index, 0)
        if used_count > 0:
            # Set default replacement choice
            props = mesh.voxel_workspace
            valid_targets = [e.index for e in props.palette if e.index > 0 and e.index != self.index]
            default_target = valid_targets[0] if valid_targets else 0
            self.replacement_choice = str(default_target)
            # Show popup dialog for replacement selection
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        mesh = obj.data if obj else None
        counts = get_used_palette_counts(mesh) if mesh else {}
        used_count = counts.get(self.index, 0)

        layout.label(text=f"Index {self.index} is used by {used_count} voxels.", icon='INFO')
        layout.prop(self, "replacement_choice", text="Replace with")
        if self.replacement_choice == "0":
            layout.label(text="WARNING: Voxels will be permanently erased!", icon='ERROR')

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        if self.index == 0:
            self.report({'ERROR'}, "Index 0 is reserved empty and cannot be removed")
            return {'CANCELLED'}

        mesh = obj.data
        props = mesh.voxel_workspace

        if self.replacement_index == -1:
            self.replacement_index = 0

        # Find entry index in CollectionProperty
        target_item_pos = None
        for pos, entry in enumerate(props.palette):
            if entry.index == self.index:
                target_item_pos = pos
                break

        if target_item_pos is None:
            self.report({'WARNING'}, f"Color index {self.index} not in palette")
            return {'CANCELLED'}

        counts = get_used_palette_counts(mesh)
        used_count = counts.get(self.index, 0)

        # Validate replacement target against allocated entries or 0 (Erase)
        allocated_indices = {entry.index for entry in props.palette}
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
                push_undo=False,
            )

        # Remove entry from CollectionProperty
        props.palette.remove(target_item_pos)

        # Force flush to Mesh IDProperties so Undo memfile captures the removal
        from ..blender.persistence import serialize_volume
        entry = get_or_load(mesh)
        if entry is not None and entry.grid is not None:
            serialize_volume(mesh, entry.grid, dirty_only=False)

        refresh_palette_image(mesh)
        drop_palette_lut(props.uuid)

        # If the removed index was active, pick the replacement or another valid index
        scene = context.scene
        if scene.voxel_workspace.active_palette_index == self.index:
            new_active = self.replacement_index if self.replacement_index > 0 else 1
            # Fallback to first available nonzero entry if needed
            if new_active not in {e.index for e in props.palette}:
                for e in props.palette:
                    if e.index > 0:
                        new_active = e.index
                        break
            scene.voxel_workspace.active_palette_index = new_active
            if 1 <= new_active <= 8:
                scene.voxel_workspace.active_palette_choice = str(new_active)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Remove Palette Color")
            except Exception:
                pass
        return {'FINISHED'}


class VOXEL_OT_eyedropper(Operator):
    """Pick color from an occupied voxel in the 3D viewport without leaving edit mode."""
    bl_idname = "voxel.eyedropper"
    bl_label = "Pick Voxel Color"
    bl_description = "Pick palette color from clicked voxel (or press 'I' / Alt+LMB while editing)"
    bl_options = {'REGISTER'}

    def _finish(self, context, result):
        """Restore the normal cursor before terminating the sampling session."""
        context.window.cursor_set('DEFAULT')
        return result

    def invoke(self, context, event):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        entry = get_or_load(obj.data)
        if entry is None or entry.grid is None:
            return {'CANCELLED'}

        # Modal sample on click
        context.window_manager.modal_handler_add(self)
        context.window.cursor_set('EYEDROPPER')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self._finish(context, {'CANCELLED'})

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            context.window.cursor_set('EYEDROPPER')
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            obj = context.active_object
            if obj is not None and hasattr(obj.data, "voxel_workspace"):
                entry = get_or_load(obj.data)
                if entry is not None and entry.grid is not None:
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
                                if origin_world is not None and dir_world is not None:
                                    origin_grid, dir_grid = world_ray_to_grid_ray(
                                        origin_world, dir_world, obj.matrix_world, voxel_size=entry.voxel_size
                                    )
                                    hit = trace_grid(entry.grid, origin_grid, dir_grid, max_distance=1000.0)
                                    if hit is not None:
                                        picked_index = entry.grid.get(hit.cell)
                                        if picked_index > 0:
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

    def invoke(self, context, event):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        mesh = obj.data if obj else None
        counts = get_used_palette_counts(mesh) if mesh else {}
        used_indices = sorted(counts.keys())
        total_voxels = sum(counts.values())

        entry = get_or_load(mesh) if mesh else None
        occupied_bricks = len([b for b in entry.grid.bricks.values() if np.any(b)]) if entry else 0

        layout.label(text=f"Volume: {occupied_bricks} occupied bricks, {total_voxels} voxels.", icon='INFO')
        layout.label(text=f"Used colors ({len(used_indices)}) will be remapped contiguously 1..{len(used_indices)}.")
        layout.label(text="Unused palette entries will be purged.")

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not hasattr(obj.data, "voxel_workspace"):
            self.report({'WARNING'}, "No active voxel volume")
            return {'CANCELLED'}

        mesh = obj.data
        props = mesh.voxel_workspace
        counts = get_used_palette_counts(mesh)

        # Build old -> new remap table for used indices
        used_indices = sorted(counts.keys())
        remap_table: Dict[int, int] = {}
        for new_idx, old_idx in enumerate(used_indices, start=1):
            remap_table[old_idx] = new_idx

        # Snapshot used palette colors & names
        saved_entries = {}
        for entry in props.palette:
            if entry.index in remap_table:
                saved_entries[remap_table[entry.index]] = (entry.name, tuple(entry.color))

        # Remap occupied bricks
        remap_volume_palette_indices(
            mesh,
            remap_table,
            push_undo=False,
        )

        # Rebuild palette collection on mesh
        props.palette.clear()

        # Reserved index 0
        from ..constants import DEFAULT_PALETTE
        empty_item = props.palette.add()
        empty_item.index = 0
        empty_item.name = "Empty"
        empty_item.color = DEFAULT_PALETTE[0]

        # Add remapped entries 1..N
        for new_idx in range(1, len(used_indices) + 1):
            item = props.palette.add()
            item.index = new_idx
            name, col = saved_entries.get(new_idx, (f"Color {new_idx}", (0.5, 0.5, 0.5, 1.0)))
            item.name = name
            item.color = col

        # If no colors used at all, ensure defaults
        if len(used_indices) == 0:
            from ..blender.properties import ensure_palette
            ensure_palette(mesh)

        # Force flush to Mesh IDProperties so Undo memfile captures the removal
        from ..blender.persistence import serialize_volume
        entry = get_or_load(mesh)
        if entry is not None and entry.grid is not None:
            serialize_volume(mesh, entry.grid, dirty_only=False)

        refresh_palette_image(mesh)
        drop_palette_lut(props.uuid)

        # Update active color in scene
        old_active = context.scene.voxel_workspace.active_palette_index
        new_active = remap_table.get(old_active, 1)
        context.scene.voxel_workspace.active_palette_index = new_active
        if 1 <= new_active <= 8:
            context.scene.voxel_workspace.active_palette_choice = str(new_active)

        tag_redraw_all_viewports()
        if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Compact Palette")
            except Exception:
                pass
        return {'FINISHED'}


PALETTE_OPERATOR_CLASSES = [
    VOXEL_OT_select_palette_color,
    VOXEL_OT_add_palette_color,
    VOXEL_OT_duplicate_palette_color,
    VOXEL_OT_remove_palette_color,
    VOXEL_OT_eyedropper,
    VOXEL_OT_compact_palette,
]
