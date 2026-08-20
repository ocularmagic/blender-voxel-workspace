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
from ..core.commands import VoxelStroke
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
    update_volume_gpu_preview,
    stop_editing,
)
from ..blender.persistence import serialize_volume
from ..blender.mesh_sync import sync_volume_mesh
from ..geometry.visible_faces import mesh_visible_faces


def snapshot_grid(grid):
    """Copy cells used for stable Place picking during one stroke."""
    from ..core.grid import VoxelGrid
    snapshot = VoxelGrid(
        extent_min=grid.extent_min,
        extent_max_exclusive=grid.extent_max_exclusive,
        brick_size=grid.brick_size,
    )
    snapshot.bricks = {coord: brick.copy() for coord, brick in grid.bricks.items()}
    return snapshot


def is_valid_voxel_object(obj: Any) -> bool:
    """Check if an object is a valid voxel volume mesh datablock."""
    return bool(
        obj is not None
        and obj.type == 'MESH'
        and hasattr(obj, "data")
        and obj.data is not None
        and hasattr(obj.data, "voxel_workspace")
        and obj.data.voxel_workspace.is_voxel_mesh
        and bool(obj.data.voxel_workspace.uuid)
    )


class BrushSession:
    """Core state machine and raycast/stroke engine for modal voxel painting/erasing."""

    def __init__(self, mode: str = "PLACE", volume_uuid: Optional[str] = None) -> None:
        self.mode: str = mode
        self.volume_uuid: Optional[str] = volume_uuid or get_active_volume_uuid()
        self.is_dragging: bool = False
        self.stroke: Optional[VoxelStroke] = None
        self.last_target: Optional[VoxelCoord] = None
        self.pick_grid = None

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
        self, context: Any, event: Any, entry: Any, obj: Any
    ) -> Tuple[Optional[VoxelCoord], Optional[VoxelCoord], Optional[VoxelCoord]]:
        """Calculate world ray and pick target from mouse event."""
        if view3d_utils is None:
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
                origin_world, dir_world, obj.matrix_world, voxel_size=entry.voxel_size
            )
            picking_grid = self.pick_grid if self.mode == 'PLACE' and self.pick_grid is not None else entry.grid
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

        # 1. Check volume validity and session state
        active_uuid = get_active_volume_uuid()
        if not active_uuid or active_uuid != self.volume_uuid:
            self.cleanup()
            return {'FINISHED'}

        obj = context.active_object
        if (
            obj is None
            or not is_valid_voxel_object(obj)
            or obj.data.voxel_workspace.uuid != self.volume_uuid
        ):
            self.cleanup()
            stop_editing(context)
            return {'CANCELLED'}

        entry = get_or_load(obj.data)
        if entry is None:
            self.cleanup()
            stop_editing(context)
            return {'CANCELLED'}

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

        # Sidebar and other UI events must remain available while editing.
        if event.type == 'LEFTMOUSE' and (
            event.value == 'PRESS' or not self.is_dragging
        ):
            region, _rv3d, _rx, _ry = self.resolve_view3d_region(context, event)
            if region is None:
                return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            region, _rv3d, _rx, _ry = self.resolve_view3d_region(context, event)
            if region is None:
                return {'PASS_THROUGH'}

        # Calculate target for current mouse position
        target_cell, hover_cell, hover_normal = self.get_brush_target(context, event, entry, obj)

        # 4. Handle LMB Press (Start Stroke)
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self.is_dragging = True
            self.stroke = VoxelStroke(brick_size=entry.grid.brick_size)
            self.pick_grid = snapshot_grid(entry.grid) if self.mode == 'PLACE' else None
            val = getattr(context.scene.voxel_workspace, "active_palette_index", 1) if self.mode == 'PLACE' else 0
            if target_cell is not None:
                self.stroke.record(entry.grid, target_cell, val)
                entry.grid.set(target_cell, val)
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
                val = getattr(context.scene.voxel_workspace, "active_palette_index", 1) if self.mode == 'PLACE' else 0
                if self.last_target is not None and self.last_target != target_cell:
                    cells = line_3d(self.last_target, target_cell)
                    for c in cells:
                        if entry.grid.in_extent(c):
                            self.stroke.record(entry.grid, c, val)
                            entry.grid.set(c, val)
                            entry.dirty_bricks.add(split_coord(c, entry.grid.brick_size)[0])
                    update_volume_gpu_preview(entry, dirty_only=True)
                    self.last_target = target_cell
                    tag_redraw_all_viewports()
                elif self.last_target is None:
                    self.stroke.record(entry.grid, target_cell, val)
                    entry.grid.set(target_cell, val)
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
                mesh = obj.data
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
                color = (1.0, 0.9, 0.2, 0.6) if self.mode == 'PLACE' else (1.0, 0.2, 0.2, 0.6)
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
                ('PLACE', 'Place', 'Place voxels'),
                ('ERASE', 'Erase', 'Erase voxels'),
            ],
            default='PLACE',
        )

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        active_uuid = get_active_volume_uuid()
        if not active_uuid:
            obj = context.active_object
            if is_valid_voxel_object(obj):
                from ..blender.gpu_preview import start_editing
                active_uuid = obj.data.voxel_workspace.uuid
                start_editing(active_uuid, context)
            else:
                self.report({'WARNING'}, "No active voxel volume for brush")
                return {'CANCELLED'}

        self.session = BrushSession(mode=self.mode, volume_uuid=active_uuid)
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
