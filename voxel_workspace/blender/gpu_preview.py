"""Depth-aware per-brick GPU viewport preview, overlay management, and work grid drawing."""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader
except ImportError:
    bpy = None
    gpu = None
    batch_for_shader = None

from ..constants import BrickCoord, DEFAULT_PALETTE
from ..geometry.buffers import MeshBuffers
from ..geometry.visible_faces import mesh_visible_faces
from .materials import PALETTE_COLORS


# Fallback default 256x4 float32 palette color table
_DEFAULT_PALETTE_RGBA_LUT: Optional[np.ndarray] = None


def _build_default_palette_rgba_lut() -> np.ndarray:
    """Return a default (256, 4) float32 RGBA palette lookup array."""
    global _DEFAULT_PALETTE_RGBA_LUT
    if _DEFAULT_PALETTE_RGBA_LUT is None:
        lut = np.zeros((256, 4), dtype=np.float32)
        for i, col in enumerate(DEFAULT_PALETTE):
            if i < 256:
                lut[i] = col
        _DEFAULT_PALETTE_RGBA_LUT = lut
    return _DEFAULT_PALETTE_RGBA_LUT


def drop_palette_lut(target: Any = None) -> None:
    """Drop the cached palette RGBA LUT and clear GPU batches so they regenerate with new colors."""
    global _DEFAULT_PALETTE_RGBA_LUT
    _DEFAULT_PALETTE_RGBA_LUT = None

    if target is None:
        from .runtime import all_volumes
        for entry in all_volumes().values():
            entry.palette_lut = None
            if hasattr(entry, "gpu_batches"):
                entry.gpu_batches.clear()
        return

    entry = None
    if isinstance(target, str):
        from .runtime import get_volume
        entry = get_volume(target)
    elif hasattr(target, "palette_lut"):
        entry = target

    if entry is not None:
        entry.palette_lut = None
        if hasattr(entry, "gpu_batches"):
            entry.gpu_batches.clear()


def get_palette_rgba_lut(target: Any = None) -> np.ndarray:
    """Return the (256, 4) float32 RGBA palette lookup array for a volume entry or target.
    
    Lazily built from the Mesh palette collection and cached on VoxelVolumeEntry.palette_lut.
    Always emits 256 rows; unallocated rows are transparent black [0, 0, 0, 0]
    (with row 0 reserved empty).
    """
    entry = None
    mesh = None

    if target is not None:
        if hasattr(target, "palette_lut"):
            entry = target
        elif isinstance(target, str):
            from .runtime import get_volume
            entry = get_volume(target)
        elif bpy is not None and hasattr(target, "voxel_workspace"):
            mesh = target
        elif bpy is not None and hasattr(target, "data") and hasattr(target.data, "voxel_workspace"):
            mesh = target.data

    if entry is not None and entry.palette_lut is not None:
        return entry.palette_lut

    # Resolve mesh from entry if not already found
    if mesh is None and entry is not None and bpy is not None and hasattr(bpy, "data") and hasattr(bpy.data, "meshes"):
        for m in bpy.data.meshes:
            if hasattr(m, "voxel_workspace") and m.voxel_workspace.uuid == entry.uuid:
                mesh = m
                break

    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        props = mesh.voxel_workspace
        lut = np.zeros((256, 4), dtype=np.float32)
        if len(props.palette) == 0:
            from .properties import ensure_palette
            ensure_palette(mesh)
        for p_entry in props.palette:
            idx = int(p_entry.index)
            if 0 <= idx < 256:
                col = list(p_entry.color)
                # If VOLUME domain, use editing preview alpha (0.35)
                if getattr(p_entry, "material_domain", "SURFACE") == "VOLUME":
                    col[3] = 0.35
                lut[idx] = col
        if entry is not None:
            entry.palette_lut = lut
        return lut

    if entry is not None:
        default_lut = _build_default_palette_rgba_lut().copy()
        entry.palette_lut = default_lut
        return default_lut

    return _build_default_palette_rgba_lut()


def palette_indices_to_rgba(palette_indices: np.ndarray, lut: Optional[np.ndarray] = None) -> np.ndarray:
    """Map an array of integer palette indices to an (N, 4) float32 RGBA array using a LUT."""
    if lut is None:
        lut = get_palette_rgba_lut()
    clipped = np.clip(palette_indices, 0, 255)
    return lut[clipped]


# --- Geometric Data Generators ---

def build_bounds_mesh_data(
    extent_min: Tuple[int, int, int],
    extent_max: Tuple[int, int, int],
    voxel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertices and line indices for the bounding box wireframe.
    
    Returns:
        verts: (8, 3) float32 array
        edges: (12, 2) int32 array of line indices
    """
    p0 = np.array(
        [extent_min[0] * voxel_size, extent_min[1] * voxel_size, extent_min[2] * voxel_size],
        dtype=np.float32,
    )
    p1 = np.array(
        [extent_max[0] * voxel_size, extent_max[1] * voxel_size, extent_max[2] * voxel_size],
        dtype=np.float32,
    )

    verts = np.array(
        [
            [p0[0], p0[1], p0[2]],  # 0
            [p1[0], p0[1], p0[2]],  # 1
            [p1[0], p1[1], p0[2]],  # 2
            [p0[0], p1[1], p0[2]],  # 3
            [p0[0], p0[1], p1[2]],  # 4
            [p1[0], p0[1], p1[2]],  # 5
            [p1[0], p1[1], p1[2]],  # 6
            [p0[0], p1[1], p1[2]],  # 7
        ],
        dtype=np.float32,
    )

    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom quad
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top quad
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical pillars
        ],
        dtype=np.int32,
    )

    return verts, edges


def build_work_grid_mesh_data(
    extent_min: Tuple[int, int, int],
    extent_max: Tuple[int, int, int],
    voxel_size: float = 1.0,
    z_coord: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertices and line indices for the Z=0 work plane grid across volume extent.
    
    Returns:
        verts: (2 * (nx + ny), 3) float32 array
        lines: ((nx + ny), 2) int32 array of line indices
    """
    min_x = extent_min[0] * voxel_size
    max_x = extent_max[0] * voxel_size
    min_y = extent_min[1] * voxel_size
    max_y = extent_max[1] * voxel_size

    step = voxel_size
    x_vals = np.arange(extent_min[0], extent_max[0] + 1, dtype=np.float32) * step
    y_vals = np.arange(extent_min[1], extent_max[1] + 1, dtype=np.float32) * step

    verts_list = []
    lines_list = []
    idx = 0

    # Lines parallel to Y (at constant X)
    for x in x_vals:
        verts_list.append([x, min_y, z_coord])
        verts_list.append([x, max_y, z_coord])
        lines_list.append([idx, idx + 1])
        idx += 2

    # Lines parallel to X (at constant Y)
    for y in y_vals:
        verts_list.append([min_x, y, z_coord])
        verts_list.append([max_x, y, z_coord])
        lines_list.append([idx, idx + 1])
        idx += 2

    verts = np.array(verts_list, dtype=np.float32)
    lines = np.array(lines_list, dtype=np.int32)
    return verts, lines


def build_hover_face_mesh_data(
    voxel_coord: Tuple[int, int, int],
    face_normal: Tuple[int, int, int] = (0, 0, 1),
    voxel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate vertices and triangle indices for a hovered voxel face highlight.
    
    Returns:
        verts: (4, 3) float32 array
        tris: (2, 3) int32 array
    """
    vx, vy, vz = voxel_coord
    nx, ny, nz = face_normal

    # Match FACE_SPECS orientations
    face_specs = {
        (1, 0, 0): np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.float32),
        (-1, 0, 0): np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.float32),
        (0, 1, 0): np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float32),
        (0, -1, 0): np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32),
        (0, 0, 1): np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32),
        (0, 0, -1): np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32),
    }

    norm_key = (
        1 if nx > 0 else (-1 if nx < 0 else 0),
        1 if ny > 0 else (-1 if ny < 0 else 0),
        1 if nz > 0 else (-1 if nz < 0 else 0),
    )
    tpl = face_specs.get(norm_key, face_specs[(0, 0, 1)])

    origin = np.array([vx, vy, vz], dtype=np.float32)
    verts = (origin[None, :] + tpl) * voxel_size

    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, tris


# --- GPU Batch Builders ---

def build_brick_gpu_batch(
    mesh_buffers: Optional[MeshBuffers],
    lut: Optional[np.ndarray] = None,
) -> Optional[Any]:
    """Create a single indexed GPUBatch from visible-face MeshBuffers with vertex RGBA colors."""
    if mesh_buffers is None or mesh_buffers.quad_count == 0 or len(mesh_buffers.positions) == 0:
        return None
    if gpu is None or batch_for_shader is None:
        return None

    try:
        shader = gpu.shader.from_builtin('FLAT_COLOR')
    except Exception:
        return None

    pos = mesh_buffers.positions
    ind = mesh_buffers.indices
    colors = palette_indices_to_rgba(mesh_buffers.palette_indices, lut=lut)

    return batch_for_shader(shader, 'TRIS', {'pos': pos, 'color': colors}, indices=ind)


def build_voxel_edge_mesh_data(
    mesh_buffers: Optional[MeshBuffers],
    surface_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return perimeter segments for exposed one-voxel preview quads."""
    if mesh_buffers is None or mesh_buffers.quad_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    quads = mesh_buffers.positions.reshape(-1, 4, 3)
    if surface_offset:
        normals = np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        quads = quads + normals[:, None, :] * surface_offset
    # Preserve each exposed quad perimeter directly. Shared opaque segments
    # are visually idempotent and avoiding Python/global deduplication keeps
    # dirty-brick rebuilds fast.
    segments = np.concatenate(
        (
            quads[:, (0, 1)],
            quads[:, (1, 2)],
            quads[:, (2, 3)],
            quads[:, (3, 0)],
        ),
        axis=0,
    )
    positions = np.ascontiguousarray(segments.reshape(-1, 3), dtype=np.float32)
    indices = np.arange(len(positions), dtype=np.int32).reshape(-1, 2)
    return positions, indices


def build_voxel_edge_gpu_batch(
    mesh_buffers: Optional[MeshBuffers],
    surface_offset: float = 0.0,
) -> Optional[Any]:
    """Create a line batch showing exposed voxel-cell boundaries."""
    if gpu is None or batch_for_shader is None:
        return None
    positions, indices = build_voxel_edge_mesh_data(mesh_buffers, surface_offset)
    if len(indices) == 0:
        return None
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return None
    return batch_for_shader(shader, 'LINES', {'pos': positions}, indices=indices)


def build_bounds_gpu_batch(
    extent_min: Tuple[int, int, int],
    extent_max: Tuple[int, int, int],
    voxel_size: float = 1.0,
) -> Optional[Any]:
    """Create a GPUBatch for the volume bounds wireframe."""
    if gpu is None or batch_for_shader is None:
        return None
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return None

    verts, edges = build_bounds_mesh_data(extent_min, extent_max, voxel_size)
    return batch_for_shader(shader, 'LINES', {'pos': verts}, indices=edges)


def build_work_grid_gpu_batch(
    extent_min: Tuple[int, int, int],
    extent_max: Tuple[int, int, int],
    voxel_size: float = 1.0,
) -> Optional[Any]:
    """Create a GPUBatch for the Z=0 work plane grid."""
    if gpu is None or batch_for_shader is None:
        return None
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return None

    verts, lines = build_work_grid_mesh_data(extent_min, extent_max, voxel_size)
    return batch_for_shader(shader, 'LINES', {'pos': verts}, indices=lines)


def build_hover_face_gpu_batch(
    voxel_coord: Tuple[int, int, int],
    face_normal: Tuple[int, int, int] = (0, 0, 1),
    voxel_size: float = 1.0,
) -> Optional[Any]:
    """Create a GPUBatch for a hovered face highlight."""
    if gpu is None or batch_for_shader is None:
        return None
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return None

    verts, tris = build_hover_face_mesh_data(voxel_coord, face_normal, voxel_size)
    return batch_for_shader(shader, 'TRIS', {'pos': verts}, indices=tris)


# --- Runtime State & Caching ---

_HOVER_STATE: Optional[Dict[str, Any]] = None
_SAVED_OVERLAYS: Dict[int, Dict[str, Any]] = {}
_DRAW_HANDLER: Optional[Any] = None
_ACTIVE_BOUNDS_BATCH: Optional[Any] = None
_ACTIVE_GRID_BATCH: Optional[Any] = None
_CACHED_VOLUME_KEY: Optional[Tuple[str, Tuple[int, int, int], Tuple[int, int, int], float]] = None


def set_hover_state(
    coord: Optional[Tuple[int, int, int]],
    normal: Tuple[int, int, int] = (0, 0, 1),
    color: Tuple[float, float, float, float] = (1.0, 0.9, 0.2, 0.6),
) -> None:
    """Set the active hover face coordinate, normal, and highlight color."""
    global _HOVER_STATE
    if coord is None:
        _HOVER_STATE = None
    else:
        _HOVER_STATE = {
            "coord": tuple(coord),
            "normal": tuple(normal),
            "color": tuple(color),
            "batch": None,
        }


def get_hover_state() -> Optional[Dict[str, Any]]:
    """Return the active hover state dictionary or None."""
    return _HOVER_STATE


def clear_hover_state() -> None:
    """Clear the active hover state."""
    global _HOVER_STATE
    _HOVER_STATE = None


def update_volume_gpu_preview(
    entry: Any,
    dirty_only: bool = True,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
) -> None:
    """Rebuild visible-face buffers and GPU batches for dirty bricks of the given volume."""
    if entry is None or not hasattr(entry, "grid"):
        return

    grid = entry.grid
    cpu_buffers = entry.cpu_buffers
    gpu_batches = entry.gpu_batches
    gpu_edge_batches = entry.gpu_edge_batches

    if dirty_only and (dirty_bricks is not None or len(grid.dirty_bricks) > 0 or len(entry.dirty_bricks) > 0) and len(gpu_batches) > 0:
        base_dirty: Set[BrickCoord] = set(dirty_bricks or set()) | set(grid.dirty_bricks) | set(entry.dirty_bricks)
        remesh_targets: Set[BrickCoord] = set()
        for bx, by, bz in base_dirty:
            remesh_targets.add((bx, by, bz))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n_coord = (bx + dx, by + dy, bz + dz)
                        if n_coord in grid.bricks or n_coord in gpu_batches or n_coord in gpu_edge_batches:
                            remesh_targets.add(n_coord)
    else:
        remesh_targets = set(grid.bricks.keys()) | set(gpu_batches.keys()) | set(gpu_edge_batches.keys())

    s = grid.brick_size
    v_size = entry.voxel_size
    lut = get_palette_rgba_lut(entry)

    for coord in remesh_targets:
        brick = grid.bricks.get(coord)
        if brick is not None and np.any(brick):
            apron = grid.read_apron(coord)
            origin = (
                float(coord[0] * s) * v_size,
                float(coord[1] * s) * v_size,
                float(coord[2] * s) * v_size,
            )
            buf = mesh_visible_faces(apron, origin=origin, voxel_size=v_size, brick=brick)
            if buf.quad_count > 0:
                cpu_buffers[coord] = buf
                batch = build_brick_gpu_batch(buf, lut=lut)
                if batch is not None:
                    gpu_batches[coord] = batch
                else:
                    gpu_batches.pop(coord, None)
                edge_batch = build_voxel_edge_gpu_batch(buf, surface_offset=v_size * 0.001)
                if edge_batch is not None:
                    gpu_edge_batches[coord] = edge_batch
                else:
                    gpu_edge_batches.pop(coord, None)
            else:
                cpu_buffers.pop(coord, None)
                gpu_batches.pop(coord, None)
                gpu_edge_batches.pop(coord, None)
        else:
            cpu_buffers.pop(coord, None)
            gpu_batches.pop(coord, None)
            gpu_edge_batches.pop(coord, None)

    entry.dirty_bricks.clear()
    grid.dirty_bricks.clear()


def clear_volume_gpu_preview(entry: Any) -> None:
    """Clear all GPU batches and CPU preview buffers for a volume."""
    if entry is None:
        return
    if hasattr(entry, "gpu_batches"):
        entry.gpu_batches.clear()
    if hasattr(entry, "gpu_edge_batches"):
        entry.gpu_edge_batches.clear()
    if hasattr(entry, "cpu_buffers"):
        entry.cpu_buffers.clear()


# --- Overlay State Management ---

def hide_view3d_overlays(context: Optional[Any] = None) -> None:
    """Save 3D view floor and axis overlay states and hide them for voxel editing."""
    if bpy is None:
        return

    wm = getattr(bpy.context, "window_manager", None) if context is None else getattr(context, "window_manager", None)
    if wm is None or not hasattr(wm, "windows"):
        return

    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and hasattr(space, "overlay"):
                        space_id = hash(space)
                        if space_id not in _SAVED_OVERLAYS:
                            _SAVED_OVERLAYS[space_id] = {
                                'space': space,
                                'show_floor': space.overlay.show_floor,
                                'show_axis_x': space.overlay.show_axis_x,
                                'show_axis_y': space.overlay.show_axis_y,
                                'show_axis_z': space.overlay.show_axis_z,
                            }
                        space.overlay.show_floor = False
                        space.overlay.show_axis_x = False
                        space.overlay.show_axis_y = False
                        space.overlay.show_axis_z = False


def restore_view3d_overlays(context: Optional[Any] = None) -> None:
    """Restore previously saved 3D view floor and axis overlay states."""
    global _SAVED_OVERLAYS
    for space_id, state in list(_SAVED_OVERLAYS.items()):
        try:
            space = state.get('space')
            if space is not None and hasattr(space, "overlay"):
                space.overlay.show_floor = state.get('show_floor', True)
                space.overlay.show_axis_x = state.get('show_axis_x', True)
                space.overlay.show_axis_y = state.get('show_axis_y', True)
                space.overlay.show_axis_z = state.get('show_axis_z', False)
        except Exception:
            pass
    _SAVED_OVERLAYS.clear()


# --- Draw Handler Callback ---

def _draw_callback() -> None:
    """POST_VIEW GPU draw handler for active voxel volume preview, bounds, grid, and hover face.
    
    CRITICAL: This function NEVER mutates bpy.data.
    Strictly restores all changed GPU states in a finally block.
    """
    if bpy is None or gpu is None:
        return

    from .runtime import get_active_volume_uuid, get_or_load
    from .object_graph import resolve_volume_context, resolve_authoritative_mesh, resolve_voxel_root, resolve_surface_object

    active_uuid = get_active_volume_uuid()
    if not active_uuid:
        return

    context = bpy.context
    if context is None:
        return

    # Find the object corresponding to active_uuid
    v_ctx = resolve_volume_context(context)
    obj = None
    if v_ctx is not None and v_ctx.mesh_uuid == active_uuid:
        obj = v_ctx.root if v_ctx.root is not None else v_ctx.surface_object
        mesh = v_ctx.mesh
    else:
        mesh = None
        if hasattr(context, "scene") and hasattr(context.scene, "objects"):
            for o in context.scene.objects:
                if (
                    hasattr(o, "data")
                    and hasattr(o.data, "voxel_workspace")
                    and o.data.voxel_workspace.uuid == active_uuid
                ):
                    root = resolve_voxel_root(o)
                    obj = root if root is not None else o
                    mesh = o.data
                    break
    if obj is None or mesh is None:
        return

    entry = get_or_load(mesh)
    if entry is None:
        return

    if not entry.gpu_batches:
        update_volume_gpu_preview(entry, dirty_only=False)

    try:
        flat_shader = gpu.shader.from_builtin('FLAT_COLOR')
        uniform_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        return

    # Helper batch lookups for bounds and work grid
    global _ACTIVE_BOUNDS_BATCH, _ACTIVE_GRID_BATCH, _CACHED_VOLUME_KEY
    grid = entry.grid
    extent_min = grid.extent_min
    extent_max = grid.extent_max_exclusive
    v_size = entry.voxel_size
    cache_key = (active_uuid, extent_min, extent_max, v_size)

    if _CACHED_VOLUME_KEY != cache_key or _ACTIVE_BOUNDS_BATCH is None:
        _ACTIVE_BOUNDS_BATCH = build_bounds_gpu_batch(extent_min, extent_max, v_size)
        _ACTIVE_GRID_BATCH = build_work_grid_gpu_batch(extent_min, extent_max, v_size)
        _CACHED_VOLUME_KEY = cache_key

    # Preserve all queryable GPU state touched by this handler. Blender's
    # Python API has no face_culling_get(), so culling is restored to NONE,
    # the documented/default state shared by the surrounding viewport path.
    previous_depth_test = gpu.state.depth_test_get()
    previous_depth_mask = gpu.state.depth_mask_get()
    previous_blend = gpu.state.blend_get()
    try:
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set('ALPHA')
        gpu.state.face_culling_set('NONE')

        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(obj.matrix_world)

            # 1. Draw Z=0 work grid
            if _ACTIVE_GRID_BATCH is not None:
                uniform_shader.bind()
                uniform_shader.uniform_float("color", (0.35, 0.35, 0.4, 0.6))
                _ACTIVE_GRID_BATCH.draw(uniform_shader)

            # 2. Draw volume bounding box wireframe
            if _ACTIVE_BOUNDS_BATCH is not None:
                uniform_shader.bind()
                uniform_shader.uniform_float("color", (0.2, 0.6, 1.0, 0.75))
                _ACTIVE_BOUNDS_BATCH.draw(uniform_shader)

            # 3. Draw active volume voxel brick batches (FLAT_COLOR)
            flat_shader.bind()
            for coord, batch in list(entry.gpu_batches.items()):
                if batch is not None:
                    batch.draw(flat_shader)

            # 4. Draw exposed voxel-cell boundaries over the colored faces.
            scene_props = getattr(getattr(context, "scene", None), "voxel_workspace", None)
            if scene_props is None or scene_props.show_voxel_edges:
                gpu.state.depth_mask_set(False)
                uniform_shader.bind()
                uniform_shader.uniform_float("color", (0.025, 0.025, 0.03, 1.0))
                for coord, batch in list(entry.gpu_edge_batches.items()):
                    if batch is not None:
                        batch.draw(uniform_shader)
                gpu.state.depth_mask_set(True)

            # 5. Draw hover face highlight if active
            if _HOVER_STATE is not None and _HOVER_STATE.get("coord") is not None:
                h_coord = _HOVER_STATE["coord"]
                h_norm = _HOVER_STATE.get("normal", (0, 0, 1))
                h_color = _HOVER_STATE.get("color", (1.0, 0.9, 0.2, 0.6))
                hover_batch = _HOVER_STATE.get("batch")
                if hover_batch is None:
                    hover_batch = build_hover_face_gpu_batch(h_coord, h_norm, v_size)
                    _HOVER_STATE["batch"] = hover_batch
                if hover_batch is not None:
                    uniform_shader.bind()
                    uniform_shader.uniform_float("color", h_color)
                    hover_batch.draw(uniform_shader)

    except Exception:
        # Drawing handler must never crash viewport rendering
        pass
    finally:
        gpu.state.depth_mask_set(previous_depth_mask)
        gpu.state.depth_test_set(previous_depth_test)
        gpu.state.blend_set(previous_blend)
        gpu.state.face_culling_set('NONE')


# --- Handler Installation & Editing Lifecycle ---

def is_draw_handler_installed() -> bool:
    """Return True if the viewport draw handler is currently installed."""
    return _DRAW_HANDLER is not None


def install_draw_handler() -> bool:
    """Install the POST_VIEW draw handler idempotently."""
    global _DRAW_HANDLER
    if _DRAW_HANDLER is not None:
        return False
    if bpy is None or not hasattr(bpy.types, "SpaceView3D"):
        return False

    _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(_draw_callback, (), 'WINDOW', 'POST_VIEW')
    return True


def remove_draw_handler() -> bool:
    """Remove the POST_VIEW draw handler idempotently."""
    global _DRAW_HANDLER, _ACTIVE_BOUNDS_BATCH, _ACTIVE_GRID_BATCH, _CACHED_VOLUME_KEY
    if _DRAW_HANDLER is None:
        return False

    if bpy is not None and hasattr(bpy.types, "SpaceView3D"):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, 'WINDOW')
        except Exception:
            pass

    _DRAW_HANDLER = None
    _ACTIVE_BOUNDS_BATCH = None
    _ACTIVE_GRID_BATCH = None
    _CACHED_VOLUME_KEY = None
    return True


def is_editing_active() -> bool:
    """Return True if a voxel volume is currently actively being edited."""
    from .runtime import get_active_volume_uuid
    return get_active_volume_uuid() is not None


def start_editing(volume_uuid: str, context: Optional[Any] = None) -> None:
    """Start editing session for a voxel volume: activate UUID, hide overlays, install handler."""
    from .runtime import set_active_volume_uuid, get_volume, tag_redraw_all_viewports

    set_active_volume_uuid(volume_uuid)

    # Set is_editing flag on object if found
    if bpy is not None:
        ctx = context if context is not None else bpy.context
        if ctx is not None and hasattr(ctx, "scene") and hasattr(ctx.scene, "objects"):
            for obj in ctx.scene.objects:
                if hasattr(obj, "data") and hasattr(obj.data, "voxel_workspace") and obj.data.voxel_workspace.uuid == volume_uuid:
                    if hasattr(obj, "voxel_workspace"):
                        obj.voxel_workspace.is_editing = True
                elif hasattr(obj, "voxel_workspace") and obj.voxel_workspace.is_editing:
                    obj.voxel_workspace.is_editing = False

    hide_view3d_overlays(context)
    install_draw_handler()

    entry = get_volume(volume_uuid)
    if entry is not None:
        update_volume_gpu_preview(entry)

    tag_redraw_all_viewports()


def stop_editing(context: Optional[Any] = None) -> None:
    """Stop active editing session: restore overlays and deactivate editing state."""
    from .runtime import get_active_volume_uuid, set_active_volume_uuid, tag_redraw_all_viewports

    active_uuid = get_active_volume_uuid()
    ctx = context if context is not None else (bpy.context if bpy is not None else None)
    if ctx is not None and getattr(ctx, "scene", None) is not None:
        scene_props = getattr(ctx.scene, "voxel_workspace", None)
        if scene_props is not None:
            scene_props.active_tool = 'NONE'
    if active_uuid and bpy is not None:
        if ctx is not None and hasattr(ctx, "scene") and hasattr(ctx.scene, "objects"):
            for obj in ctx.scene.objects:
                if hasattr(obj, "voxel_workspace") and obj.voxel_workspace.is_editing:
                    obj.voxel_workspace.is_editing = False

    set_active_volume_uuid(None)
    clear_hover_state()
    remove_draw_handler()
    restore_view3d_overlays(context)
    tag_redraw_all_viewports()


def cleanup_gpu_preview(context: Optional[Any] = None) -> None:
    """Idempotently remove all preview UI/GPU state for unload or errors."""
    global _ACTIVE_BOUNDS_BATCH, _ACTIVE_GRID_BATCH, _CACHED_VOLUME_KEY
    clear_hover_state()
    remove_draw_handler()
    restore_view3d_overlays(context)
    _ACTIVE_BOUNDS_BATCH = None
    _ACTIVE_GRID_BATCH = None
    _CACHED_VOLUME_KEY = None
