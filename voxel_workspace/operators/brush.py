"""Modal Place and Erase Voxel Brush operator and stroke session management."""
from typing import Any, Optional, Tuple

try:
    import bpy
    from bpy.props import EnumProperty
    from bpy.types import Operator
    from bpy_extras import view3d_utils
except ImportError:
    bpy = None
    Operator = object
    EnumProperty = None
    view3d_utils = None

from ..constants import VoxelCoord
from ..core.coords import split_coord
from ..core.commands import VoxelStroke, apply_brush_value
from ..core.tagged_grid import TaggedVoxelGrid, VoxelCell, VoxelDomain, CELL_EMPTY
from ..core.line import (
    line_3d,
    compute_brush_target,
    world_ray_to_grid_ray,
)
from ..blender.runtime import (
    get_active_volume_uuid,
    get_volume,
    get_or_load,
    tag_redraw_all_viewports,
)
from ..blender.gpu_preview import (
    set_hover_state,
    clear_hover_state,
    refresh_material_display_colors,
    update_volume_gpu_preview,
    stop_editing,
)
from ..blender.persistence import serialize_volume
from ..blender.mesh_sync import sync_volume_mesh
from ..geometry.visible_faces import mesh_visible_faces


_BRUSH_MODAL_GENERATION = 0


def begin_brush_modal_session() -> int:
    """Invalidate older brush handlers and return a token for the new modal."""
    global _BRUSH_MODAL_GENERATION
    _BRUSH_MODAL_GENERATION += 1
    return _BRUSH_MODAL_GENERATION


def request_brush_modal_stop() -> None:
    """Invalidate the currently registered brush modal on its next event."""
    global _BRUSH_MODAL_GENERATION
    _BRUSH_MODAL_GENERATION += 1


def is_event_over_ui_region(context: Any, event: Any) -> bool:
    """Return True when a modal event belongs to UI rather than a 3D WINDOW."""
    context_region = getattr(context, "region", None)
    if context_region is not None and getattr(context_region, "type", None) != 'WINDOW':
        return True
    # Background projection/event coordinates are intentionally synthetic and
    # may fall outside the viewport; its harness supplies the WINDOW context.
    if bpy is not None and bpy.app.background:
        return False

    screen = getattr(context, "screen", None)
    mouse_x = getattr(event, "mouse_x", -1)
    mouse_y = getattr(event, "mouse_y", -1)
    if screen is None or mouse_x < 0 or mouse_y < 0:
        return False

    for area in screen.areas:
        if not (area.x <= mouse_x < area.x + area.width and area.y <= mouse_y < area.y + area.height):
            continue
        if area.type != 'VIEW_3D':
            return True
        # Sidebars/toolbars can overlap the WINDOW region in screen space, so
        # test every visible non-WINDOW region before accepting viewport input.
        for region in area.regions:
            if region.type == 'WINDOW' or region.width <= 1 or region.height <= 1:
                continue
            if (
                region.x <= mouse_x < region.x + region.width
                and region.y <= mouse_y < region.y + region.height
            ):
                return True
        return False
    return True


def snapshot_grid(grid):
    """Copy cells used for stable Place picking during one stroke."""
    if hasattr(grid, "get_cell"):
        from ..core.tagged_grid import TaggedVoxelGrid
        snapshot = TaggedVoxelGrid(
            extent_min=grid.extent_min,
            extent_max_exclusive=grid.extent_max_exclusive,
            brick_size=grid.brick_size,
        )
        snapshot.bricks = {coord: brick.copy() for coord, brick in grid.bricks.items()}
        return snapshot
    from ..core.grid import VoxelGrid
    snapshot = VoxelGrid(
        extent_min=grid.extent_min,
        extent_max_exclusive=grid.extent_max_exclusive,
        brick_size=grid.brick_size,
    )
    snapshot.bricks = {coord: brick.copy() for coord, brick in grid.bricks.items()}
    return snapshot


def brush_cell_for_scene(scene: Any, mode: str) -> VoxelCell:
    """Resolve the canonical tagged value for an explicit brush mode."""
    normalized = str(mode).upper()
    props = scene.voxel_workspace
    if normalized in {'ADD_SURFACE', 'PLACE'}:
        return VoxelCell(VoxelDomain.SURFACE, int(props.active_surface_palette_index))
    if normalized == 'ADD_VOLUME':
        return VoxelCell(VoxelDomain.VOLUME, int(props.active_volume_palette_index))
    if normalized == 'REPAINT':
        # Repaint converts the target voxel to the material family of the
        # active palette tab (SURFACE or VOLUME).
        if str(getattr(props, 'active_palette_tab', 'SURFACE')).upper() == 'VOLUME':
            return VoxelCell(VoxelDomain.VOLUME, int(props.active_volume_palette_index))
        return VoxelCell(VoxelDomain.SURFACE, int(props.active_surface_palette_index))
    if normalized == 'ERASE':
        return CELL_EMPTY
    raise ValueError(f"Unknown brush mode: {mode}")


def brush_display_color_for_scene(scene: Any, mesh: Any, mode: str) -> Tuple[float, float, float, float]:
    """Resolve the live material-derived hover color for the active brush."""
    normalized = str(mode).upper()
    if normalized == "ERASE":
        return (1.0, 0.2, 0.2, 0.6)
    if normalized == "REPAINT":
        palette_type = "VOLUME" if brush_cell_for_scene(scene, normalized).domain == VoxelDomain.VOLUME else "SURFACE"
    else:
        palette_type = "VOLUME" if normalized == "ADD_VOLUME" else "SURFACE"
    fallback = (0.25, 0.65, 1.0, 0.45) if palette_type == "VOLUME" else (1.0, 0.9, 0.2, 0.6)
    if scene is None or mesh is None:
        return fallback
    try:
        from ..blender.material_domains import find_entry, display_rgba_from_entry
        cell = brush_cell_for_scene(scene, normalized)
        entry = find_entry(mesh, palette_type, cell.index)
        rgba = display_rgba_from_entry(entry, palette_type)
        alpha = 0.45 if palette_type == "VOLUME" else 0.6
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha)
    except Exception:
        return fallback


from ..blender.object_graph import (
    resolve_volume_context,
    resolve_authoritative_mesh,
    resolve_voxel_root,
    resolve_surface_object,
)


def is_valid_voxel_object(obj_or_context: Any) -> bool:
    """Check if an object is a valid voxel volume mesh datablock."""
    if obj_or_context is None:
        return False
    mesh = resolve_authoritative_mesh(obj_or_context)
    return bool(
        mesh is not None
        and hasattr(mesh, "voxel_workspace")
        and mesh.voxel_workspace.is_voxel_mesh
        and bool(mesh.voxel_workspace.uuid)
    )


class BrushSession:
    """Core state machine and raycast/stroke engine for modal voxel painting/erasing."""

    def __init__(
        self,
        mode: str = "ADD_SURFACE",
        volume_uuid: Optional[str] = None,
        root_instance_uuid: Optional[str] = None,
        modal_token: Optional[int] = None,
    ) -> None:
        self.mode: str = mode
        self.volume_uuid: Optional[str] = volume_uuid or get_active_volume_uuid()
        self.root_instance_uuid: Optional[str] = root_instance_uuid
        self.is_dragging: bool = False
        self.stroke: Optional[VoxelStroke] = None
        self.last_target: Optional[VoxelCoord] = None
        self.pick_grid = None
        self.modal_token = modal_token

    def resolve_view3d_region(
        self, context: Any, event: Any
    ) -> Tuple[Optional[Any], Optional[Any], int, int]:
        """Find the VIEW_3D WINDOW region and rv3d corresponding to the mouse event."""
        # VIEW_3D sidebars/toolbars overlap the WINDOW region in screen
        # coordinates. Reject those visible UI regions first so their clicks
        # can pass through to Blender controls such as palette swatches.
        if context.screen is not None:
            mouse_x = getattr(event, "mouse_x", -1)
            mouse_y = getattr(event, "mouse_y", -1)
            for area in context.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for ui_region in area.regions:
                    if ui_region.type == 'WINDOW' or ui_region.width <= 1 or ui_region.height <= 1:
                        continue
                    if (
                        ui_region.x <= mouse_x < ui_region.x + ui_region.width
                        and ui_region.y <= mouse_y < ui_region.y + ui_region.height
                    ):
                        return None, None, 0, 0

        # 1. Direct match on context
        if (
            context.area is not None
            and context.area.type == 'VIEW_3D'
            and context.region is not None
            and context.region.type == 'WINDOW'
            and context.region_data is not None
        ):
            rx, ry = event.mouse_region_x, event.mouse_region_y
            if (
                0 <= rx < context.region.width and 0 <= ry < context.region.height
            ) or (bpy is not None and bpy.app.background):
                return context.region, context.region_data, rx, ry

        # 2. Window coordinate search across VIEW_3D areas
        if context.screen is not None:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for r in area.regions:
                        if r.type == 'WINDOW':
                            rx = event.mouse_x - r.x
                            ry = event.mouse_y - r.y
                            if 0 <= rx < r.width and 0 <= ry < r.height:
                                rv3d = getattr(area.spaces.active, "region_3d", context.region_data)
                                return r, rv3d, rx, ry

        # Background integration harnesses do not refresh RegionView3D
        # projection matrices like the foreground event loop. Keep their
        # historical clamped fallback without exposing it in the real UI.
        if bpy is not None and bpy.app.background and context.screen is not None:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for r in area.regions:
                        if r.type == 'WINDOW':
                            rx = max(0, min(r.width - 1, event.mouse_x - r.x))
                            ry = max(0, min(r.height - 1, event.mouse_y - r.y))
                            rv3d = getattr(area.spaces.active, "region_3d", context.region_data)
                            return r, rv3d, rx, ry

        return None, None, 0, 0

    def get_brush_target(
        self, context: Any, event: Any, entry: Any, transform_obj: Any
    ) -> Tuple[Optional[VoxelCoord], Optional[VoxelCoord], Optional[VoxelCoord]]:
        """Calculate world ray and pick target from mouse event."""
        if view3d_utils is None or transform_obj is None:
            return None, None, None

        region, rv3d, rx, ry = self.resolve_view3d_region(context, event)
        if region is None or rv3d is None:
            return None, None, None

        try:
            origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, (rx, ry))
            dir_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, (rx, ry))
            if origin_world is None or dir_world is None:
                return None, None, None

            origin_grid, dir_grid = world_ray_to_grid_ray(
                origin_world, dir_world, transform_obj.matrix_world, voxel_size=entry.voxel_size
            )
            picking_grid = self.pick_grid if self.mode in {'ADD_SURFACE', 'ADD_VOLUME', 'PLACE'} and self.pick_grid is not None else entry.grid
            return compute_brush_target(picking_grid, origin_grid, dir_grid, mode=self.mode)
        except Exception:
            return None, None, None

    def cleanup(self, entry: Optional[Any] = None) -> None:
        """Reset dragging state and revert uncommitted stroke if any."""
        if self.is_dragging and self.stroke is not None and entry is not None:
            self.stroke.revert(entry.grid)
            update_volume_gpu_preview(entry, dirty_only=False)
        self.is_dragging = False
        self.stroke = None
        self.last_target = None
        self.pick_grid = None
        clear_hover_state()

    def handle_event(self, context: Any, event: Any) -> set:
        """Process a window/viewport event according to modal stroke rules."""
        if bpy is None:
            return {'CANCELLED'}

        if self.modal_token is not None and self.modal_token != _BRUSH_MODAL_GENERATION:
            self.cleanup(get_volume(self.volume_uuid) if self.volume_uuid else None)
            return {'FINISHED'}

        # 1. Check volume validity and session state
        active_uuid = get_active_volume_uuid()
        if not active_uuid or active_uuid != self.volume_uuid:
            self.cleanup(get_volume(self.volume_uuid) if self.volume_uuid else None)
            return {'FINISHED'}

        # Never consume events delivered to sidebars, toolbars, headers, or
        # other editors. Their operators must remain usable during voxel edit.
        if is_event_over_ui_region(context, event):
            return {'PASS_THROUGH'}

        v_ctx = resolve_volume_context(context)
        if (
            v_ctx is None
            or v_ctx.mesh_uuid != self.volume_uuid
        ):
            self.cleanup()
            stop_editing(context)
            return {'CANCELLED'}

        entry = get_or_load(v_ctx.mesh)
        if entry is None:
            self.cleanup()
            stop_editing(context)
            return {'CANCELLED'}
        refresh_material_display_colors(entry)
        if self.root_instance_uuid and v_ctx.root_instance_uuid != self.root_instance_uuid:
            self.cleanup(entry)
            stop_editing(context)
            return {'CANCELLED'}

        # Ray conversion uses the root's matrix_world if available, else surface object
        transform_obj = v_ctx.root if v_ctx.root is not None else v_ctx.surface_object

        # 2. Pass-through viewport navigation and undo keys
        if event.type in {
            'MIDDLEMOUSE',
            'WHEELUPMOUSE',
            'WHEELDOWNMOUSE',
            'NUMPAD_1',
            'NUMPAD_2',
            'NUMPAD_3',
            'NUMPAD_4',
            'NUMPAD_5',
            'NUMPAD_6',
            'NUMPAD_7',
            'NUMPAD_8',
            'NUMPAD_9',
            'NUMPAD_0',
            'NUMPAD_PERIOD',
            'NDOF_MOTION',
        }:
            return {'PASS_THROUGH'}

        if (event.ctrl or getattr(event, "oskey", False)) and event.type in {'Z', 'Y'}:
            clear_hover_state()
            if self.is_dragging and self.stroke is not None:
                self.stroke = None
            self.is_dragging = False
            self.last_target = None
            self.pick_grid = None
            return {'PASS_THROUGH'}

        # 3. Handle Escape Key (Cancel stroke or Exit session)
        if event.type == 'ESC' and event.value == 'PRESS':
            if self.is_dragging and self.stroke is not None:
                # Revert stroke without undo push
                self.stroke.revert(entry.grid)
                update_volume_gpu_preview(entry, dirty_only=False)
                self.stroke = None
                self.is_dragging = False
                self.last_target = None
                self.pick_grid = None
                tag_redraw_all_viewports()
                return {'RUNNING_MODAL'}
            else:
                # Idle: exit editing session
                clear_hover_state()
                stop_editing(context)
                return {'FINISHED'}

        # 3b. Handle Eyedropper Hotkey (Alt+LMB or 'I' key press)
        is_eyedropper_event = (
            (event.type == 'I' and event.value == 'PRESS')
            or (getattr(event, "alt", False) and event.type == 'LEFTMOUSE' and event.value == 'PRESS')
        )
        if is_eyedropper_event and not self.is_dragging:
            # Trace ray against occupied voxels using DDA trace_grid
            if view3d_utils is not None:
                region, rv3d, rx, ry = self.resolve_view3d_region(context, event)
                if region is not None and rv3d is not None:
                    try:
                        from ..core.dda import trace_grid
                        from ..core.line import world_ray_to_grid_ray
                        origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, (rx, ry))
                        dir_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, (rx, ry))
                        if origin_world is not None and dir_world is not None:
                            origin_grid, dir_grid = world_ray_to_grid_ray(
                                origin_world, dir_world, transform_obj.matrix_world, voxel_size=entry.voxel_size
                            )
                            hit = trace_grid(entry.grid, origin_grid, dir_grid, max_distance=1000.0)
                            if hit is not None:
                                picked_index = entry.grid.get(hit.cell)
                                if picked_index > 0:
                                    cell = entry.grid.get_cell(hit.cell) if isinstance(entry.grid, TaggedVoxelGrid) else None
                                    if cell is not None and cell.domain == VoxelDomain.VOLUME:
                                        context.scene.voxel_workspace.active_volume_palette_index = picked_index
                                        context.scene.voxel_workspace.active_palette_tab = 'VOLUME'
                                    else:
                                        context.scene.voxel_workspace.active_surface_palette_index = picked_index
                                        context.scene.voxel_workspace.active_palette_index = picked_index
                                        context.scene.voxel_workspace.active_palette_tab = 'SURFACE'
                                        if 1 <= picked_index <= 8:
                                            context.scene.voxel_workspace.active_palette_choice = str(picked_index)
                                    tag_redraw_all_viewports()
                                    return {'RUNNING_MODAL'}
                    except Exception:
                        pass
            return {'RUNNING_MODAL'}

        # Sidebar and other UI events must remain available while editing.
        # But if mouse is not in a sidebar region, don't drop plain mousemove in background tests.
        if event.type == 'LEFTMOUSE' and not self.is_dragging:
            region, _rv3d, _rx, _ry = self.resolve_view3d_region(context, event)
            if region is None:
                return {'PASS_THROUGH'}

        # Calculate target for current mouse position
        target_cell, hover_cell, hover_normal = self.get_brush_target(context, event, entry, transform_obj)

        # 4. Handle LMB Press (Start Stroke)
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Ensure target region is VIEW_3D WINDOW
            region, _rv3d, _rx, _ry = self.resolve_view3d_region(context, event)
            if region is None:
                return {'PASS_THROUGH'}
            self.is_dragging = True
            self.stroke = VoxelStroke(brick_size=entry.grid.brick_size)
            self.pick_grid = snapshot_grid(entry.grid) if self.mode in {'ADD_SURFACE', 'ADD_VOLUME', 'PLACE'} else None
            cell = brush_cell_for_scene(context.scene, self.mode)
            if target_cell is not None:
                self.stroke.record(entry.grid, target_cell, cell)
                if isinstance(entry.grid, TaggedVoxelGrid):
                    apply_brush_value(entry.grid, target_cell, 'ADD_SURFACE' if self.mode == 'PLACE' else self.mode, cell.index, domain=cell.domain)
                else:
                    entry.grid.set(target_cell, cell.index)
                entry.dirty_bricks.add(split_coord(target_cell, entry.grid.brick_size)[0])
                update_volume_gpu_preview(entry, dirty_only=True)
                self.last_target = target_cell
            else:
                self.last_target = None
            tag_redraw_all_viewports()
            return {'RUNNING_MODAL'}

        # 5. Handle Mouse Move while Dragging (Stroke drag & fill missed cells)
        if event.type == 'MOUSEMOVE' and self.is_dragging:
            if target_cell is not None:
                cell = brush_cell_for_scene(context.scene, self.mode)
                if self.last_target is not None and self.last_target != target_cell:
                    cells = line_3d(self.last_target, target_cell)
                    for c in cells:
                        if entry.grid.in_extent(c):
                            self.stroke.record(entry.grid, c, cell)
                            if isinstance(entry.grid, TaggedVoxelGrid):
                                apply_brush_value(entry.grid, c, 'ADD_SURFACE' if self.mode == 'PLACE' else self.mode, cell.index, domain=cell.domain)
                            else:
                                entry.grid.set(c, cell.index)
                            entry.dirty_bricks.add(split_coord(c, entry.grid.brick_size)[0])
                    update_volume_gpu_preview(entry, dirty_only=True)
                    self.last_target = target_cell
                    tag_redraw_all_viewports()
                elif self.last_target is None:
                    self.stroke.record(entry.grid, target_cell, cell)
                    if isinstance(entry.grid, TaggedVoxelGrid):
                        apply_brush_value(entry.grid, target_cell, 'ADD_SURFACE' if self.mode == 'PLACE' else self.mode, cell.index, domain=cell.domain)
                    else:
                        entry.grid.set(target_cell, cell.index)
                    entry.dirty_bricks.add(split_coord(target_cell, entry.grid.brick_size)[0])
                    update_volume_gpu_preview(entry, dirty_only=True)
                    self.last_target = target_cell
                    tag_redraw_all_viewports()
            return {'RUNNING_MODAL'}

        # 6. Handle LMB Release (Commit Stroke)
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.is_dragging:
            self.is_dragging = False
            self.last_target = None
            if self.stroke is not None and len(self.stroke.deltas) > 0:
                mesh = v_ctx.mesh
                changed_bricks = self.stroke.changed_bricks()
                entry.grid.dirty_bricks.update(changed_bricks)
                entry.dirty_bricks.update(changed_bricks)
                serialize_volume(mesh, entry.grid, dirty_only=True)
                sync_volume_mesh(
                    mesh,
                    grid=entry.grid,
                    dirty_only=True,
                    dirty_bricks=changed_bricks,
                    voxel_size=entry.voxel_size,
                    mesher=mesh_visible_faces,
                )
                if hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
                    try:
                        bpy.ops.ed.undo_push(message="Voxel Stroke")
                    except Exception:
                        pass
            self.stroke = None
            self.pick_grid = None
            update_volume_gpu_preview(entry, dirty_only=False)
            tag_redraw_all_viewports()
            return {'RUNNING_MODAL'}

        # 7. Handle Mouse Move while Idle (Hover highlight)
        if event.type == 'MOUSEMOVE' and not self.is_dragging:
            if hover_cell is not None and hover_normal is not None:
                color = brush_display_color_for_scene(context.scene, v_ctx.mesh, self.mode)
                set_hover_state(hover_cell, hover_normal, color=color)
            else:
                clear_hover_state()
            tag_redraw_all_viewports()
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}


class VOXEL_OT_brush(Operator):
    """Interactive modal voxel brush tool for placing and erasing voxels."""
    bl_idname = "voxel.brush"
    bl_label = "Voxel Brush"
    bl_description = "Interactive modal voxel brush tool"
    # Non-blocking so palette swatches and other sidebar controls remain live.
    bl_options = set()

    if bpy is not None:
        mode: EnumProperty(
            name="Mode",
            description="Brush tool mode",
            items=[
                ('ADD_SURFACE', 'Add Surface', 'Add Surface voxels'),
                ('ADD_VOLUME', 'Add Volume', 'Add Volume voxels'),
                ('REPAINT', 'Repaint', 'Repaint existing voxels to the active palette material'),
                ('ERASE', 'Erase', 'Erase voxels'),
            ],
            default='ADD_SURFACE',
        )

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        v_ctx = resolve_volume_context(context)
        active_uuid = get_active_volume_uuid()
        if not active_uuid:
            if v_ctx is not None and v_ctx.mesh_uuid:
                from ..blender.gpu_preview import start_editing
                active_uuid = v_ctx.mesh_uuid
                start_editing(active_uuid, context)
            else:
                self.report({'WARNING'}, "No active voxel volume for brush")
                return {'CANCELLED'}

        modal_token = begin_brush_modal_session()
        self.session = BrushSession(
            mode=self.mode,
            volume_uuid=active_uuid,
            root_instance_uuid=v_ctx.root_instance_uuid if v_ctx is not None else None,
            modal_token=modal_token,
        )
        context.scene.voxel_workspace.active_tool = self.mode
        context.window_manager.modal_handler_add(self)
        tag_redraw_all_viewports()
        return {'RUNNING_MODAL'}

    def modal(self, context: Any, event: Any) -> set:
        if getattr(self, "session", None) is None:
            return {'CANCELLED'}
        return self.session.handle_event(context, event)


BRUSH_OPERATOR_CLASSES = [
    VOXEL_OT_brush,
]
