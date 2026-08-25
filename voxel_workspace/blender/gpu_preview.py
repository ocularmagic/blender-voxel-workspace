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


def build_typed_palette_lut(entries: Any, volume_alpha: Optional[float] = None) -> np.ndarray:
    """Build one independent 256-row display LUT from a typed palette collection."""
    from .material_domains import display_rgba_from_entry, linear_to_srgb_rgba
    lut = np.zeros((256, 4), dtype=np.float32)
    palette_type = "VOLUME" if volume_alpha is not None else "SURFACE"
    for palette_entry in entries:
        idx = int(palette_entry.index)
        if not (1 <= idx <= 255):
            continue
        color = np.asarray(linear_to_srgb_rgba(display_rgba_from_entry(palette_entry, palette_type)), dtype=np.float32).copy()
        if volume_alpha is not None:
            color[3] = float(volume_alpha)
        lut[idx] = color
    return lut


def drop_palette_lut(target: Any = None, palette_type: Optional[str] = None) -> None:
    """Drop one or both typed display LUTs and color-baked GPU batches."""
    global _DEFAULT_PALETTE_RGBA_LUT
    _DEFAULT_PALETTE_RGBA_LUT = None
    domain = str(palette_type).upper() if palette_type else None

    if target is None:
        from .runtime import all_volumes
        entries = list(all_volumes().values())
    else:
        entry = None
        if isinstance(target, str):
            from .runtime import get_volume
            entry = get_volume(target)
        elif hasattr(target, "palette_lut"):
            entry = target
        entries = [entry] if entry is not None else []

    for entry in entries:
        if domain in {None, "SURFACE"}:
            entry.surface_palette_lut = None
            entry.palette_lut = None
            entry.gpu_batches.clear()
        if domain in {None, "VOLUME"}:
            entry.volume_palette_lut = None
            if hasattr(entry, "volume_gpu_batches"):
                entry.volume_gpu_batches.clear()


def get_palette_rgba_lut(target: Any = None, palette_type: str = "SURFACE") -> np.ndarray:
    """Return an independent 256-row display LUT for Surface or Volume Palette."""
    domain = str(palette_type).upper()
    if domain not in {"SURFACE", "VOLUME"}:
        raise ValueError(f"Unknown palette type: {palette_type}")
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

    cache_name = "volume_palette_lut" if domain == "VOLUME" else "surface_palette_lut"
    if entry is not None and getattr(entry, cache_name, None) is not None:
        return getattr(entry, cache_name)

    if mesh is None and entry is not None and bpy is not None:
        for candidate in bpy.data.meshes:
            if hasattr(candidate, "voxel_workspace") and candidate.voxel_workspace.uuid == entry.uuid:
                mesh = candidate
                break

    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        from .properties import ensure_palette
        ensure_palette(mesh)
        props = mesh.voxel_workspace
        entries = props.volume_palette if domain == "VOLUME" else props.surface_palette
        lut = build_typed_palette_lut(entries, volume_alpha=0.35 if domain == "VOLUME" else None)
        if entry is not None:
            setattr(entry, cache_name, lut)
            if domain == "SURFACE":
                entry.palette_lut = lut
        return lut

    default_lut = _build_default_palette_rgba_lut().copy()
    if domain == "VOLUME":
        default_lut[:, 3] = np.where(default_lut[:, 3] > 0.0, 0.35, 0.0)
    if entry is not None:
        setattr(entry, cache_name, default_lut)
        if domain == "SURFACE":
            entry.palette_lut = default_lut
    return default_lut


def palette_indices_to_rgba(palette_indices: np.ndarray, lut: Optional[np.ndarray] = None) -> np.ndarray:
    """Map an array of integer palette indices to an (N, 4) float32 RGBA array using a LUT."""
    if lut is None:
        lut = get_palette_rgba_lut()
    clipped = np.clip(palette_indices, 0, 255)
    return lut[clipped]


def recolor_preview_batches(
    entry: Any,
    palette_type: str,
    lut: Optional[np.ndarray] = None,
) -> None:
    """Rebuild color-baked batches from cached CPU buffers without remeshing.

    Edge batches are rebuilt alongside the fill batches so auto-contrast seam
    colors follow material edits immediately.
    """
    if entry is None:
        return
    domain = str(palette_type).upper()
    if domain == "VOLUME":
        buffers = entry.volume_preview_buffers
        batches = entry.volume_gpu_batches
        edge_batches = getattr(entry, "volume_gpu_edge_batches", {})
    else:
        buffers = entry.surface_preview_buffers
        batches = entry.gpu_batches
        edge_batches = getattr(entry, "gpu_edge_batches", {})
    if lut is None:
        lut = get_palette_rgba_lut(entry, domain)
    batches.clear()
    edge_batches.clear()
    for coord, mesh_buffers in buffers.items():
        batch = build_brick_gpu_batch(mesh_buffers, lut=lut)
        if batch is not None:
            batches[coord] = batch
        edge_batch = build_voxel_edge_gpu_batch_auto(mesh_buffers, surface_offset=0.001, lut=lut)
        if edge_batch is not None:
            edge_batches[coord] = edge_batch


def refresh_material_display_colors(entry: Any) -> bool:
    """Refresh placement colors when bound material sockets change.

    Material node inputs are edited directly by Blender, so palette-entry RNA
    callbacks do not run. Compare fresh material-derived LUTs with the cached
    Surface/Volume LUTs and recolor existing GPU batches only when needed.
    """
    if entry is None or bpy is None:
        return False
    mesh = next(
        (
            candidate
            for candidate in bpy.data.meshes
            if hasattr(candidate, "voxel_workspace")
            and candidate.voxel_workspace.uuid == entry.uuid
        ),
        None,
    )
    if mesh is None:
        return False

    from .properties import ensure_palette
    ensure_palette(mesh)
    props = mesh.voxel_workspace
    changed = False
    for domain, entries, alpha, cache_name in (
        ("SURFACE", props.surface_palette, None, "surface_palette_lut"),
        ("VOLUME", props.volume_palette, 0.35, "volume_palette_lut"),
    ):
        fresh_lut = build_typed_palette_lut(entries, volume_alpha=alpha)
        cached_lut = getattr(entry, cache_name, None)
        if cached_lut is not None and np.array_equal(cached_lut, fresh_lut):
            continue
        setattr(entry, cache_name, fresh_lut)
        if domain == "SURFACE":
            entry.palette_lut = fresh_lut
        recolor_preview_batches(entry, domain, lut=fresh_lut)
        changed = True
    return changed


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


def build_hover_face_outline_mesh_data(
    voxel_coord: Tuple[int, int, int],
    face_normal: Tuple[int, int, int] = (0, 0, 1),
    voxel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate perimeter line segments for a hovered voxel face highlight.

    Returns:
        verts: (4, 3) float32 array
        lines: (4, 2) int32 array
    """
    verts, _tris = build_hover_face_mesh_data(voxel_coord, face_normal, voxel_size)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    return verts, lines


def build_hover_face_x_mesh_data(
    voxel_coord: Tuple[int, int, int],
    face_normal: Tuple[int, int, int] = (0, 0, 1),
    voxel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an X (corner-to-corner cross) for the hovered face.

    Used as a mode-specific erase marker that stays recognizable no matter
    what color the underlying voxel is.

    Returns:
        verts: (4, 3) float32 array
        lines: (2, 2) int32 array
    """
    verts, _tris = build_hover_face_mesh_data(voxel_coord, face_normal, voxel_size)
    lines = np.array([[0, 2], [1, 3]], dtype=np.int32)
    return verts, lines


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


# Auto-contrast edge tuning: edges invert against the voxel fill luminance.
EDGE_AUTO_DARK = (0.02, 0.02, 0.03)
EDGE_AUTO_LIGHT = (0.97, 0.97, 0.95)


def _edge_contrast_rgba(rgba: np.ndarray) -> np.ndarray:
    """Map an (N, 4) sRGB RGBA array to contrast-safe edge colors.

    Dark fills get light edges and vice versa; mid-tones get the dark edge,
    which reads better over Blender's default mid-gray viewport background
    than white lines on a light model.
    """
    rgb = rgba[:, :3]
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    fill_is_light = (lum > 0.5).astype(np.float32)[:, None]
    dark = np.array(EDGE_AUTO_DARK, dtype=np.float32)[None, :]
    bright = np.array(EDGE_AUTO_LIGHT, dtype=np.float32)[None, :]
    # Light fills take the dark edge; dark fills take the light edge.
    color = bright * (1.0 - fill_is_light) + dark * fill_is_light
    alpha = rgba[:, 3:4] if rgba.shape[1] > 3 else np.ones((len(rgba), 1), dtype=np.float32)
    return np.concatenate([color, alpha], axis=1)


def build_voxel_edge_gpu_batch_auto(
    mesh_buffers: Optional[MeshBuffers],
    surface_offset: float = 0.0,
    lut: Optional[np.ndarray] = None,
) -> Optional[Any]:
    """Create a per-vertex-colored line batch with auto-contrast edge colors.

    Each segment inherits its voxel's palette color via the LUT, then flips to
    the inverse luminance band so black voxels show light seams and light
    voxels keep dark seams.
    """
    if gpu is None or batch_for_shader is None:
        return None
    positions, indices = build_voxel_edge_mesh_data(mesh_buffers, surface_offset)
    if len(indices) == 0:
        return None
    # build_voxel_edge_mesh_data emits quads in order: each source quad becomes
    # exactly four consecutive 2-point segments (8 line vertices). Colors must
    # match the VERTEX count: one edge color per segment, duplicated onto its
    # two endpoint vertices.
    seg_per_quad = 4
    verts_per_segment = 2
    verts_per_quad = seg_per_quad * verts_per_segment
    quad_count = len(mesh_buffers.positions) // 4 if mesh_buffers is not None else 0
    # mesh_visible_faces stores palette_indices PER VERTEX (4 per quad), so
    # take every fourth entry to recover one color per quad.
    if (
        lut is not None
        and mesh_buffers is not None
        and len(mesh_buffers.palette_indices) == len(mesh_buffers.positions)
        and quad_count > 0
    ):
        quad_indices = np.asarray(mesh_buffers.palette_indices).reshape(-1, 4)[:, 0]
        quad_rgba = palette_indices_to_rgba(quad_indices, lut=lut)
        seg_rgba = np.repeat(quad_rgba, verts_per_quad, axis=0)
        colors = _edge_contrast_rgba(seg_rgba)
    else:
        colors = None
    try:
        shader = gpu.shader.from_builtin('FLAT_COLOR')
    except Exception:
        return None
    attrs = {'pos': positions}
    if colors is not None and len(colors) == len(positions):
        attrs['color'] = np.ascontiguousarray(colors, dtype=np.float32)
        try:
            return batch_for_shader(shader, 'LINES', attrs, indices=indices)
        except Exception:
            # Never let a colored-batch mismatch break the edit session; fall
            # back to the uniform-color edge batch.
            pass
    return build_voxel_edge_gpu_batch(mesh_buffers, surface_offset)


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
_PENDING_ERASE: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
_SAVED_OVERLAYS: Dict[int, Dict[str, Any]] = {}
_DRAW_HANDLER: Optional[Any] = None
_ACTIVE_BOUNDS_BATCH: Optional[Any] = None
_ACTIVE_GRID_BATCH: Optional[Any] = None
_CACHED_VOLUME_KEY: Optional[Tuple[str, Tuple[int, int, int], Tuple[int, int, int], float]] = None


def set_hover_state(
    coord: Optional[Tuple[int, int, int]],
    normal: Tuple[int, int, int] = (0, 0, 1),
    color: Tuple[float, float, float, float] = (1.0, 0.9, 0.2, 0.6),
    mode: Optional[str] = None,
) -> None:
    """Set the active hover face coordinate, normal, color, and brush mode."""
    global _HOVER_STATE
    if coord is None:
        _HOVER_STATE = None
    else:
        _HOVER_STATE = {
            "coord": tuple(coord),
            "normal": tuple(normal),
            "color": tuple(color),
            "mode": str(mode).upper() if mode else None,
            "batch": None,
        }


def get_hover_state() -> Optional[Dict[str, Any]]:
    """Return the active hover state dictionary or None."""
    return _HOVER_STATE


def clear_hover_state() -> None:
    """Clear the active hover state."""
    global _HOVER_STATE
    _HOVER_STATE = None


def set_pending_erase(cells: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> None:
    """Set the list of (coord, normal) faces marked for deferred erase."""
    global _PENDING_ERASE
    _PENDING_ERASE = [(tuple(c), tuple(n)) for c, n in cells]


def get_pending_erase() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return the pending erase face list."""
    return _PENDING_ERASE


def clear_pending_erase() -> None:
    """Clear all pending erase marks."""
    global _PENDING_ERASE
    _PENDING_ERASE = []


def update_volume_gpu_preview(
    entry: Any,
    dirty_only: bool = True,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
) -> None:
    """Rebuild independent Surface and Volume editing-preview batches."""
    if entry is None or not hasattr(entry, "grid"):
        return

    from ..core.tagged_grid import TaggedVoxelGrid, VoxelDomain

    grid = entry.grid
    surface_buffers = entry.surface_preview_buffers
    surface_batches = entry.gpu_batches
    surface_edges = entry.gpu_edge_batches
    volume_buffers = entry.volume_preview_buffers
    volume_batches = entry.volume_gpu_batches
    volume_edges = entry.volume_gpu_edge_batches

    all_cached = set(surface_batches) | set(surface_edges) | set(volume_batches) | set(volume_edges)
    if dirty_only and (dirty_bricks is not None or grid.dirty_bricks or entry.dirty_bricks) and all_cached:
        base_dirty = set(dirty_bricks or set()) | set(grid.dirty_bricks) | set(entry.dirty_bricks)
        remesh_targets: Set[BrickCoord] = set()
        for bx, by, bz in base_dirty:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        coord = (bx + dx, by + dy, bz + dz)
                        if coord in grid.bricks or coord in all_cached or coord == (bx, by, bz):
                            remesh_targets.add(coord)
    else:
        remesh_targets = set(grid.bricks) | all_cached

    s = grid.brick_size
    v_size = entry.voxel_size
    surface_lut = get_palette_rgba_lut(entry, "SURFACE")
    volume_lut = get_palette_rgba_lut(entry, "VOLUME")

    def rebuild_domain(coord, domain, buffers, batches, edges, lut):
        brick = grid.bricks.get(coord)
        if brick is None:
            buffers.pop(coord, None); batches.pop(coord, None); edges.pop(coord, None)
            return
        if isinstance(grid, TaggedVoxelGrid):
            core = np.where(brick.domains == int(domain), brick.indices, 0).astype(np.uint8)
            # Mesh each domain independently with a domain-filtered apron, matching
            # sync_volume_mesh's committed meshing. An unfiltered apron would cull
            # faces at cross-domain contacts (e.g. SURFACE voxels flush against
            # VOLUME voxels after an interior fill), hiding them in the viewport
            # even though they exist in the rendered mesh.
            apron = grid.read_index_apron(coord, domain_filter=domain)
        else:
            if domain == VoxelDomain.VOLUME:
                core = np.zeros((s, s, s), dtype=np.uint8)
                apron = np.zeros((s + 2, s + 2, s + 2), dtype=np.uint8)
            else:
                core = brick
                apron = grid.read_apron(coord)
        if not np.any(core):
            buffers.pop(coord, None); batches.pop(coord, None); edges.pop(coord, None)
            return
        origin = tuple(float(coord[axis] * s) * v_size for axis in range(3))
        buf = mesh_visible_faces(apron, origin=origin, voxel_size=v_size, brick=core)
        if buf.quad_count == 0:
            buffers.pop(coord, None); batches.pop(coord, None); edges.pop(coord, None)
            return
        buffers[coord] = buf
        batch = build_brick_gpu_batch(buf, lut=lut)
        if batch is not None:
            batches[coord] = batch
        else:
            batches.pop(coord, None)
        edge_batch = build_voxel_edge_gpu_batch_auto(buf, surface_offset=v_size * 0.001, lut=lut)
        if edge_batch is not None:
            edges[coord] = edge_batch
        else:
            edges.pop(coord, None)

    for coord in remesh_targets:
        rebuild_domain(coord, VoxelDomain.SURFACE, surface_buffers, surface_batches, surface_edges, surface_lut)
        rebuild_domain(coord, VoxelDomain.VOLUME, volume_buffers, volume_batches, volume_edges, volume_lut)

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
    if hasattr(entry, "surface_preview_buffers"):
        entry.surface_preview_buffers.clear()
    if hasattr(entry, "volume_preview_buffers"):
        entry.volume_preview_buffers.clear()
    if hasattr(entry, "volume_gpu_batches"):
        entry.volume_gpu_batches.clear()
    if hasattr(entry, "volume_gpu_edge_batches"):
        entry.volume_gpu_edge_batches.clear()


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

def _draw_voxel_edge_batches(
    entry: Any,
    scene_props: Any,
    uniform_shader: Any,
    flat_shader: Any,
) -> None:
    """Draw all editing edge batches honoring the edge color mode.

    Auto Contrast on: batches carry per-vertex contrast colors, drawn with the
    FLAT_COLOR shader. Off: every batch is drawn with the user's fixed override
    color through UNIFORM_COLOR. Shared by the base pass and the hover redraw
    so a highlighted face can never repaint edges in a different color mode.
    """
    auto = scene_props is None or bool(scene_props.voxel_edge_auto_contrast)
    if auto:
        flat_shader.bind()
        for coord, batch in list(entry.gpu_edge_batches.items()):
            if batch is not None:
                batch.draw(flat_shader)
        for coord, batch in list(getattr(entry, "volume_gpu_edge_batches", {}).items()):
            if batch is not None:
                batch.draw(flat_shader)
    else:
        # Picker value is RGBA; uniforms take exactly 4 components.
        color = tuple(scene_props.voxel_edge_color)[:4]
        uniform_shader.bind()
        uniform_shader.uniform_float("color", color)
        for coord, batch in list(entry.gpu_edge_batches.items()):
            if batch is not None:
                batch.draw(uniform_shader)
        for coord, batch in list(getattr(entry, "volume_gpu_edge_batches", {}).items()):
            if batch is not None:
                batch.draw(uniform_shader)


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
            # Volume preview uses a separately color-baked, translucent LUT.
            for coord, batch in list(getattr(entry, "volume_gpu_batches", {}).items()):
                if batch is not None:
                    batch.draw(flat_shader)

            # 4. Draw exposed voxel-cell boundaries over the colored faces.
            # Batches carry per-vertex auto-contrast colors (light seams on
            # dark voxels, dark seams on light ones) unless the user pinned a
            # manual override color in the Voxel N-panel.
            scene_props = getattr(getattr(context, "scene", None), "voxel_workspace", None)
            if scene_props is None or scene_props.show_voxel_edges:
                gpu.state.depth_mask_set(False)
                _draw_voxel_edge_batches(entry, scene_props, uniform_shader, flat_shader)
                gpu.state.depth_mask_set(True)

            # 5. Draw hover face highlight if active
            if _HOVER_STATE is not None and _HOVER_STATE.get("coord") is not None:
                h_coord = _HOVER_STATE["coord"]
                h_norm = _HOVER_STATE.get("normal", (0, 0, 1))
                h_color = _HOVER_STATE.get("color", (1.0, 0.9, 0.2, 0.6))
                h_mode = _HOVER_STATE.get("mode")
                hover_batch = _HOVER_STATE.get("batch")
                if hover_batch is None:
                    hover_batch = build_hover_face_gpu_batch(h_coord, h_norm, v_size)
                    _HOVER_STATE["batch"] = hover_batch
                if hover_batch is not None:
                    uniform_shader.bind()
                    uniform_shader.uniform_float("color", h_color)
                    # Additive blending guarantees the tint brightens the face
                    # no matter how dark the underlying voxel color is.
                    gpu.state.blend_set('ADDITIVE')
                    gpu.state.depth_test_set('LESS_EQUAL')
                    hover_batch.draw(uniform_shader)
                    gpu.state.blend_set('ALPHA')
                    # Perimeter outline in a color guaranteed to contrast with
                    # the tint (its inverse), offset slightly along the face
                    # normal so it never z-fights with the fill quad.
                    try:
                        outline_verts, outline_lines = build_hover_face_outline_mesh_data(h_coord, h_norm, v_size)
                        eps = v_size * 0.002
                        n_vec = np.array(h_norm, dtype=np.float32)
                        outline_verts = outline_verts + n_vec * eps
                        outline_batch = batch_for_shader(
                            uniform_shader, 'LINES', {"pos": outline_verts}, indices=outline_lines
                        )
                        lum = 0.2126 * h_color[0] + 0.7152 * h_color[1] + 0.0722 * h_color[2]
                        o_color = (0.02,) * 3 if lum > 0.5 else (0.98,) * 3
                        uniform_shader.uniform_float("color", (*o_color, 0.95))
                        outline_batch.draw(uniform_shader)

                        # Erase mode: draw an X across the face so erase stays
                        # recognizable even when it lands on a same-colored voxel.
                        if h_mode == "ERASE":
                            x_verts, x_lines = build_hover_face_x_mesh_data(h_coord, h_norm, v_size)
                            x_verts = x_verts + n_vec * (eps * 2.0)
                            x_batch = batch_for_shader(
                                uniform_shader, 'LINES', {"pos": x_verts}, indices=x_lines
                            )
                            # White halo underneath + black X on top reads on
                            # any voxel color, including white.
                            uniform_shader.uniform_float("color", (1.0, 1.0, 1.0, 0.85))
                            x_batch.draw(uniform_shader)
                            x_batch.draw(uniform_shader)
                            x_batch.draw(uniform_shader)
                            uniform_shader.uniform_float("color", (0.05, 0.05, 0.08, 1.0))
                            x_batch.draw(uniform_shader)
                    except Exception:
                        pass

                # 5b. Redraw cell-boundary edges over the highlight so voxels
                # keep their definition inside a traced/highlighted region.
                # Same color mode as step 4: auto-contrast per-vertex colors,
                # or the user's fixed override. Never the old hardcoded black.
                if scene_props is None or scene_props.show_voxel_edges:
                    gpu.state.depth_mask_set(False)
                    _draw_voxel_edge_batches(entry, scene_props, uniform_shader, flat_shader)
                    gpu.state.depth_mask_set(True)

            # 5c. Draw pending-erase marks: every voxel touched this stroke
            # keeps a red tint + X until the mouse is released.
            if _PENDING_ERASE and entry is not None:
                erase_color = (1.0, 0.2, 0.2, 0.6)
                try:
                    for p_coord, p_norm in _PENDING_ERASE:
                        n_vec = np.array(p_norm, dtype=np.float32)
                        eps = v_size * 0.002
                        # Darken pass: subtractive so red tint reads even on
                        # near-white voxels where additive red saturates.
                        d_batch = build_hover_face_gpu_batch(p_coord, p_norm, v_size)
                        if d_batch is not None:
                            uniform_shader.bind()
                            uniform_shader.uniform_float("color", (0.35, 0.35, 0.35, 1.0))
                            gpu.state.blend_set('MULTIPLY')
                            gpu.state.depth_test_set('LESS_EQUAL')
                            d_batch.draw(uniform_shader)
                            gpu.state.blend_set('ALPHA')
                        # Red tint
                        p_batch = build_hover_face_gpu_batch(p_coord, p_norm, v_size)
                        if p_batch is not None:
                            uniform_shader.bind()
                            uniform_shader.uniform_float("color", erase_color)
                            gpu.state.blend_set('ADDITIVE')
                            gpu.state.depth_test_set('LESS_EQUAL')
                            p_batch.draw(uniform_shader)
                            gpu.state.blend_set('ALPHA')
                        # Outline
                        o_verts, o_lines = build_hover_face_outline_mesh_data(p_coord, p_norm, v_size)
                        o_verts = o_verts + n_vec * eps
                        o_batch = batch_for_shader(
                            uniform_shader, 'LINES', {"pos": o_verts}, indices=o_lines
                        )
                        uniform_shader.uniform_float("color", (0.02, 0.02, 0.03, 0.95))
                        o_batch.draw(uniform_shader)
                        # X marker: white halo underneath + black X on top
                        # reads on any voxel color, including white.
                        x_verts, x_lines = build_hover_face_x_mesh_data(p_coord, p_norm, v_size)
                        x_verts = x_verts + n_vec * (eps * 2.0)
                        x_batch = batch_for_shader(
                            uniform_shader, 'LINES', {"pos": x_verts}, indices=x_lines
                        )
                        uniform_shader.uniform_float("color", (1.0, 1.0, 1.0, 0.85))
                        x_batch.draw(uniform_shader)
                        x_batch.draw(uniform_shader)
                        x_batch.draw(uniform_shader)
                        uniform_shader.uniform_float("color", (0.05, 0.05, 0.08, 1.0))
                        x_batch.draw(uniform_shader)
                except Exception:
                    pass

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

    # Self-heal a Surface child that was transformed directly outside the voxel
    # tools (e.g. the user rotated the model in object mode). The edit preview
    # and brush picking transform through Voxel Root while the committed mesh
    # renders through the Surface object; a non-identity child local transform
    # makes those disagree and shows a rotated ghost. Fold it up into the root
    # before any preview batches are built.
    try:
        from .object_graph import normalize_voxel_child_transforms
        if bpy is not None:
            ctx = context if context is not None else bpy.context
            if ctx is not None and hasattr(ctx, "scene") and hasattr(ctx.scene, "objects"):
                for obj in ctx.scene.objects:
                    if (
                        hasattr(obj, "data")
                        and hasattr(obj.data, "voxel_workspace")
                        and obj.data.voxel_workspace.uuid == volume_uuid
                    ):
                        normalize_voxel_child_transforms(obj)
                        break
    except Exception:
        pass

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
