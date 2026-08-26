"""Viewport modal arrow tool that stretches/squashes a root's interior voxels.

Interaction mirrors :mod:`adjust_voxel_root`: persistent modal session,
RGB corner arrows, one commit per released drag, panel Accept/Cancel, Escape
restores the session-start state.  The difference is semantic: dragging an
arrow rescales that axis of the volume AND remaps every occupied voxel, so
the shape itself stretches (with solid gap-filling) or squashes.
"""
from typing import Any

try:
    import bpy
    import gpu
    from bpy.types import Operator
    from bpy_extras import view3d_utils
    from gpu_extras.batch import batch_for_shader
    from mathutils import Vector
except ImportError:
    bpy = gpu = None
    Operator = object
    view3d_utils = batch_for_shader = Vector = None

import numpy as np

from ..blender.object_graph import resolve_volume_context
from ..blender.runtime import get_or_load
from .brush import request_brush_modal_stop, is_event_over_ui_region
from .adjust_voxel_root import (
    _corners,
    _handle_length,
    _tag_all_areas,
)
from ..core.scale_volume import (
    apply_scaled_axis,
    compute_scale_writes,
    validate_scale_extent,
)

_STATE = None
_HANDLER = None


def is_scaling_active() -> bool:
    """Return whether the persistent interior-scale modal is active."""
    return _STATE is not None


def _world(matrix, point, voxel_size):
    return matrix @ Vector((point[0] * voxel_size, point[1] * voxel_size,
                            point[2] * voxel_size))


def _remove_handler():
    global _HANDLER, _STATE
    if _HANDLER is not None and bpy is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLER, 'WINDOW')
        except Exception:
            pass
    _HANDLER = None
    _STATE = None


def _finish_mode(context):
    if context is not None and getattr(context, 'scene', None) is not None:
        props = getattr(context.scene, 'voxel_workspace', None)
        if props is not None and str(getattr(props, 'active_tool', 'NONE')) == 'SCALE':
            props.active_tool = 'NONE'
    _remove_handler()
    _tag_all_areas()


def _snapshot_grid(grid):
    """Capture every occupied cell plus extents so Cancel can restore them."""
    cells = []
    bs = int(grid.brick_size)
    for bcoord, brick in grid.bricks.items():
        nz = np.argwhere(brick.indices > 0)
        has_domains = hasattr(brick, 'domains')
        for c in nz:
            lx, ly, lz = int(c[0]), int(c[1]), int(c[2])
            gc = bcoord[0] * bs + lx, bcoord[1] * bs + ly, bcoord[2] * bs + lz
            dom = int(brick.domains[lx, ly, lz]) if has_domains else 1
            cells.append((gc, dom, int(brick.indices[lx, ly, lz])))
    return cells


def _restore_snapshot(mesh, entry, snapshot):
    """Rebuild the grid from a snapshot and re-sync Blender data."""
    grid = entry.grid
    cells, extents = snapshot
    grid.extent_min = tuple(extents[0])
    grid.extent_max_exclusive = tuple(extents[1])
    grid.bricks.clear()
    grid.dirty_bricks.clear()
    tagged = hasattr(grid, 'set_cell')
    for coord, dom, index in cells:
        if tagged:
            grid.set_cell(coord, dom, index)
        else:
            grid.set(coord, index)
    from ..blender.persistence import serialize_volume
    serialize_volume(mesh, grid, dirty_only=False)
    from ..blender.mesh_sync import sync_volume_mesh
    entry.cpu_buffers.clear()
    sync_volume_mesh(mesh, grid=grid, entry=entry, dirty_only=False,
                     ensure_material=False, voxel_size=float(entry.voxel_size))
    _refresh_edit_preview(entry)


def _refresh_edit_preview(entry):
    """Rebuild the live editing-overlay batches so the viewport reflects the
    scaled/restored grid immediately (the committed mesh alone is not what
    the session draws)."""
    try:
        from ..blender.gpu_preview import update_volume_gpu_preview
        update_volume_gpu_preview(entry, dirty_only=False)
    except Exception:
        pass


def _draw_callback():
    """Bounds wireframe + RGB arrows; the live target box flashes amber."""
    if _STATE is None or gpu is None or batch_for_shader is None:
        return
    try:
        emin, emax = _STATE['extent']
        matrix, vs = _STATE['matrix'], _STATE['voxel_size']
        corners = _corners(emin, emax)
        edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
        segments = []
        for a, b in edges:
            segments.extend((_world(matrix, corners[a], vs),
                             _world(matrix, corners[b], vs)))
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        try:
            gpu.state.line_width_set(3.0)
        except Exception:
            pass
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES',
                                 {'pos': np.asarray(segments, dtype=np.float32)})
        color = ((0.95, 0.65, 0.10, 0.95) if _STATE['handle'] is not None
                 else (0.55, 0.55, 0.55, 0.8))
        shader.bind(); shader.uniform_float('color', color); batch.draw(shader)
        # Blender's standard transform colors: X red, Y green, Z blue.
        for axis, acolor in enumerate(((0.8, 0.08, 0.08, 0.98),
                                       (0.08, 0.75, 0.12, 0.98),
                                       (0.12, 0.35, 0.95, 0.98))):
            axis_segments = []
            for corner in corners:
                side = 1 if corner[axis] == emax[axis] else -1
                length = _handle_length(emin, emax, axis)
                end = list(corner); end[axis] += side * length
                axis_segments.extend((_world(matrix, corner, vs),
                                      _world(matrix, tuple(end), vs)))
                for other in ((axis + 1) % 3, (axis + 2) % 3):
                    wing = list(end); wing[other] -= 0.38 * side
                    wing[axis] -= 0.38 * side
                    axis_segments.extend((_world(matrix, tuple(end), vs),
                                          _world(matrix, tuple(wing), vs)))
            abatch = batch_for_shader(shader, 'LINES',
                                      {'pos': np.asarray(axis_segments, dtype=np.float32)})
            shader.bind(); shader.uniform_float('color', acolor); abatch.draw(shader)
        try:
            gpu.state.line_width_set(1.0)
        except Exception:
            pass
    except Exception:
        pass


class VOXEL_OT_accept_scale_voxels(Operator):
    """Accept all stretch/squash changes in the current session."""
    bl_idname = 'voxel.accept_scale_voxels'
    bl_label = 'Accept'
    bl_options = {'REGISTER'}

    def execute(self, context: Any) -> set:
        if _STATE is None:
            return {'CANCELLED'}
        _finish_mode(context)
        return {'FINISHED'}


class VOXEL_OT_cancel_scale_voxels(Operator):
    """Cancel and undo every stretch/squash change made this session."""
    bl_idname = 'voxel.cancel_scale_voxels'
    bl_label = 'Cancel'
    bl_options = {'REGISTER'}

    def execute(self, context: Any) -> set:
        if _STATE is None:
            return {'CANCELLED'}
        try:
            _restore_snapshot(_STATE['mesh'], _STATE['entry'],
                              _STATE['session_snapshot'])
        except Exception:
            pass
        _finish_mode(context)
        return {'FINISHED'}


class VOXEL_OT_scale_voxels(Operator):
    """Drag an outward arrow to stretch/squash the interior voxels of that axis."""
    bl_idname = 'voxel.scale_voxels'
    bl_label = 'Stretch / Squash Interior'
    bl_description = ('Drag an arrow to scale one axis; the voxels inside '
                      'stretch (filled solid) or squash with the boundary')
    bl_options = {'REGISTER'}

    def invoke(self, context: Any, event: Any) -> set:
        global _STATE, _HANDLER
        vctx = resolve_volume_context(context)
        if vctx is None or vctx.mesh is None or vctx.root is None:
            self.report({'ERROR'}, 'Active object is not a valid voxel volume')
            return {'CANCELLED'}
        entry = get_or_load(vctx.mesh)
        if entry is None or entry.grid is None:
            self.report({'ERROR'}, 'No voxel grid available for this volume')
            return {'CANCELLED'}
        props = vctx.mesh.voxel_workspace
        if _STATE is not None:
            return {'CANCELLED'}
        request_brush_modal_stop()
        from ..blender.gpu_preview import start_editing
        start_editing(props.uuid, context)
        props.active_tool = 'SCALE'
        extent = (tuple(props.extent_min), tuple(props.extent_max))
        # PRISTINE session snapshot: never replaced by later drags, so
        # Accept/Cancel always measure against and restore the exact state
        # from when this mode was entered.
        _STATE = {
            'extent': extent,
            'original_extent': extent,
            'session_snapshot': (_snapshot_grid(entry.grid), extent),
            'matrix': vctx.root.matrix_world.copy(),
            'voxel_size': float(entry.voxel_size),
            'entry': entry,
            'mesh': vctx.mesh,
            'handle': None,
            'drag_start': None,
            'drag_extent': extent,
        }
        if _HANDLER is None:
            _HANDLER = bpy.types.SpaceView3D.draw_handler_add(
                _draw_callback, (), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        _tag_all_areas()
        return {'RUNNING_MODAL'}

    @staticmethod
    def _region(context, event):
        area = context.area if context.area and context.area.type == 'VIEW_3D' else next(
            (a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
        if area is None:
            return None, None, 0, 0
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            return None, None, 0, 0
        return region, area.spaces.active.region_3d, event.mouse_x - region.x, event.mouse_y - region.y

    def _pick(self, context, event):
        region, rv3d, rx, ry = self._region(context, event)
        if region is None:
            return None
        best = (18.0, None)
        emin, emax = _STATE['extent']
        for corner in _corners(emin, emax):
            for axis in range(3):
                side = 1 if corner[axis] == emax[axis] else -1
                end = list(corner)
                end[axis] += side * _handle_length(emin, emax, axis)
                for point in (corner, tuple(end)):
                    screen = view3d_utils.location_3d_to_region_2d(
                        region, rv3d, _world(_STATE['matrix'], point,
                                             _STATE['voxel_size']))
                    if screen is not None:
                        distance = ((screen.x - rx) ** 2 +
                                    (screen.y - ry) ** 2) ** 0.5
                        if distance < best[0]:
                            best = (distance, (axis, side))
        return best[1]

    def _drag_face(self, context, event):
        axis, side = _STATE['handle']
        region, rv3d, rx, ry = self._region(context, event)
        if region is None:
            return
        local_axis = Vector((0.0, 0.0, 0.0))
        local_axis[axis] = 1.0
        center = tuple((a + b) * 0.5 for a, b in zip(*_STATE['drag_extent']))
        center_screen = view3d_utils.location_3d_to_region_2d(
            region, rv3d, _world(_STATE['matrix'], center, _STATE['voxel_size']))
        axis_screen = view3d_utils.location_3d_to_region_2d(
            region, rv3d, _world(_STATE['matrix'],
                                  tuple(center[i] + local_axis[i] for i in range(3)),
                                  _STATE['voxel_size']))
        if center_screen is None or axis_screen is None:
            return
        screen_axis = Vector((axis_screen.x - center_screen.x,
                              axis_screen.y - center_screen.y))
        pixels_per_voxel = screen_axis.length
        if pixels_per_voxel <= 1e-6 or _STATE['drag_start'] is None:
            return
        screen_axis.normalize()
        delta = Vector((event.mouse_x - _STATE['drag_start'][0],
                        event.mouse_y - _STATE['drag_start'][1]))
        voxel_delta = round(delta.dot(screen_axis) / pixels_per_voxel)
        initial_face = (_STATE['drag_extent'][1][axis]
                        if side > 0 else _STATE['drag_extent'][0][axis])
        face = initial_face + voxel_delta
        emin, emax = _STATE['drag_extent']
        o_lo, o_hi = emin[axis], emax[axis]
        if side > 0:
            face = max(o_lo + 1, min(o_lo + 512, int(face)))
        else:
            face = max(o_hi - 512, min(o_hi - 1, int(face)))
        lo, hi = list(emin), list(emax)
        if side > 0:
            hi[axis] = face
        else:
            lo[axis] = face
        _STATE['extent'] = (tuple(lo), tuple(hi))

    def modal(self, context: Any, event: Any) -> set:
        if _STATE is None:
            return {'CANCELLED'}
        if event.type == 'ESC':
            bpy.ops.voxel.cancel_scale_voxels()
            return {'CANCELLED'}
        if is_event_over_ui_region(context, event):
            return {'PASS_THROUGH'}
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                          'NDOF_MOTION'} or event.type.startswith('NUMPAD'):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and _STATE['handle'] is None:
            _STATE['handle'] = self._pick(context, event)
            if _STATE['handle'] is not None:
                _STATE['drag_start'] = (event.mouse_x, event.mouse_y)
                _STATE['drag_extent'] = _STATE['extent']
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and _STATE['handle'] is not None:
            self._drag_face(context, event)
            _tag_all_areas()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and _STATE['handle'] is not None:
            axis, _side = _STATE['handle']
            # The remap source is the extent the grid CURRENTLY has (i.e. the
            # state before THIS drag), not the session-start one; the pristine
            # session snapshot stays untouched for Cancel.
            old_extent = _STATE['drag_extent']
            new_extent = _STATE['extent']
            if new_extent != old_extent:
                err = validate_scale_extent(new_extent[0], new_extent[1])
                if err is None:
                    apply_scaled_axis(_STATE['mesh'], _STATE['entry'].grid,
                                      _STATE['entry'], old_extent, new_extent,
                                      axis)
                    _refresh_edit_preview(_STATE['entry'])
                    try:
                        bpy.ops.ed.undo_push(message='Stretch/Squash Interior')
                    except Exception:
                        pass
                else:
                    self.report({'ERROR'}, err)
                    _STATE['extent'] = old_extent
            _STATE['handle'] = None
            _STATE['drag_start'] = None
            _STATE['drag_extent'] = _STATE['extent']
            _tag_all_areas()
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


SCALE_OPERATOR_CLASSES = [
    VOXEL_OT_scale_voxels,
    VOXEL_OT_accept_scale_voxels,
    VOXEL_OT_cancel_scale_voxels,
]
