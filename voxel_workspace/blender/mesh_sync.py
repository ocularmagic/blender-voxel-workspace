"""Committed mesh synchronization and bulk datablock writing for voxel volumes."""
from typing import Any, Callable, Dict, Optional, Set, Tuple
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

from ..constants import BrickCoord
from ..core.grid import VoxelGrid
from ..geometry.buffers import MeshBuffers
from ..geometry.greedy import mesh_greedy
from .materials import ensure_voxel_material, PALETTE_ATTRIBUTE_NAME


def sync_volume_mesh(
    mesh: Any,
    grid: Optional[VoxelGrid] = None,
    entry: Optional[Any] = None,
    dirty_only: bool = True,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    mesher: Callable[..., MeshBuffers] = mesh_greedy,
    ensure_material: bool = True,
    voxel_size: Optional[float] = None,
) -> Dict[BrickCoord, MeshBuffers]:
    """Synchronize a Blender Mesh datablock with the authoritative VoxelGrid.
    
    Rebuilds only dirty bricks plus their direct 26 neighbors when dirty_only=True.
    Caches MeshBuffers in the runtime entry (or local dict) and bulk-writes vertices,
    loops, polygons, and the CORNER-domain INT palette_index attribute using foreach_set.
    Preserves all existing IDProperties and custom properties on the mesh.
    """
    if mesh is None or bpy is None:
        return {}

    vol_uuid = ""
    if hasattr(mesh, "voxel_workspace"):
        vol_uuid = mesh.voxel_workspace.uuid
        if voxel_size is None:
            voxel_size = float(mesh.voxel_workspace.voxel_size)

    if voxel_size is None:
        voxel_size = 1.0

    if entry is None and vol_uuid:
        from .runtime import get_volume
        entry = get_volume(vol_uuid)

    if grid is None:
        if entry is not None:
            grid = entry.grid
        else:
            from .persistence import deserialize_volume
            grid = deserialize_volume(mesh)

    cpu_buffers: Dict[BrickCoord, MeshBuffers]
    if entry is not None:
        cpu_buffers = entry.cpu_buffers
    else:
        cpu_buffers = {}

    # Determine which brick coordinates must be remeshed
    if dirty_only and (dirty_bricks is not None or len(grid.dirty_bricks) > 0) and len(cpu_buffers) > 0:
        base_dirty: Set[BrickCoord] = set(dirty_bricks) if dirty_bricks is not None else set(grid.dirty_bricks)
        remesh_targets: Set[BrickCoord] = set()
        for bx, by, bz in base_dirty:
            remesh_targets.add((bx, by, bz))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n_coord = (bx + dx, by + dy, bz + dz)
                        if n_coord in grid.bricks or n_coord in cpu_buffers:
                            remesh_targets.add(n_coord)
    else:
        remesh_targets = set(grid.bricks.keys()) | set(cpu_buffers.keys())

    # Remesh target bricks
    s = grid.brick_size
    v_size = voxel_size
    for coord in remesh_targets:
        brick = grid.bricks.get(coord)
        if brick is not None and np.any(brick):
            apron = grid.read_apron(coord)
            origin = (
                float(coord[0] * s) * v_size,
                float(coord[1] * s) * v_size,
                float(coord[2] * s) * v_size,
            )
            buf = mesher(apron, origin=origin, voxel_size=v_size, brick=brick)
            if buf.quad_count > 0:
                cpu_buffers[coord] = buf
            else:
                cpu_buffers.pop(coord, None)
        else:
            cpu_buffers.pop(coord, None)

    # Concatenate cached brick buffers in deterministic sorted order
    sorted_coords = sorted(cpu_buffers.keys())
    all_positions = []
    all_indices = []
    all_palette_per_vert = []
    vert_offset = 0

    for coord in sorted_coords:
        buf = cpu_buffers[coord]
        if buf.quad_count == 0 or len(buf.positions) == 0:
            continue
        all_positions.append(buf.positions)
        all_indices.append(buf.indices + vert_offset)
        all_palette_per_vert.append(buf.palette_indices)
        vert_offset += len(buf.positions)

    # Clear geometry while preserving IDProperties and datablock identity
    mesh.clear_geometry()

    # Reconcile native surface material slots
    from .material_domains import reconcile_surface_slots
    slot_map = reconcile_surface_slots(mesh, grid)

    if ensure_material:
        # If no surface slots created (e.g. legacy/fallback), ensure legacy material
        if len(mesh.materials) == 0:
            ensure_voxel_material(mesh)

    if not all_positions or vert_offset == 0:
        mesh.update()
        return cpu_buffers

    positions = np.concatenate(all_positions, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    palette_per_vert = np.concatenate(all_palette_per_vert, axis=0)

    total_verts = len(positions)
    total_tris = len(indices)
    total_loops = total_tris * 3

    # Flatten triangle indices to loop vertex indices
    loop_vertex_indices = indices.ravel().astype(np.int32)

    # Map each corner loop to the palette index of its referenced vertex
    corner_palette = palette_per_vert[loop_vertex_indices].astype(np.int32)

    # Triangle palette index is the palette index of its first vertex
    tri_first_vert_indices = indices[:, 0]
    tri_palette_indices = palette_per_vert[tri_first_vert_indices].astype(np.int32)

    # Map each triangle to its corresponding material slot
    if slot_map:
        # Vectorized lookup via mapping array
        max_pal_idx = int(np.max(tri_palette_indices)) if len(tri_palette_indices) > 0 else 0
        lut_size = max(256, max_pal_idx + 1)
        slot_lut = np.zeros(lut_size, dtype=np.int32)
        for pal_idx, slot_idx in slot_map.items():
            if pal_idx < lut_size:
                slot_lut[pal_idx] = slot_idx
        tri_material_slots = slot_lut[tri_palette_indices]
    else:
        tri_material_slots = np.zeros(total_tris, dtype=np.int32)

    # Bulk datablock write using foreach_set
    mesh.vertices.add(total_verts)
    mesh.vertices.foreach_set("co", positions.ravel())

    mesh.loops.add(total_loops)
    mesh.loops.foreach_set("vertex_index", loop_vertex_indices)

    mesh.polygons.add(total_tris)
    mesh.polygons.foreach_set("loop_start", np.arange(0, total_loops, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(total_tris, 3, dtype=np.int32))
    mesh.polygons.foreach_set("material_index", tri_material_slots)

    # Create/update CORNER INT palette_index attribute
    attr = mesh.attributes.get(PALETTE_ATTRIBUTE_NAME)
    if attr is None or attr.domain != "CORNER" or attr.data_type != "INT":
        if attr is not None:
            mesh.attributes.remove(attr)
        attr = mesh.attributes.new(name=PALETTE_ATTRIBUTE_NAME, type="INT", domain="CORNER")

    attr.data.foreach_set("value", corner_palette)

    # Clean up and remove unreferenced legacy atlas material/image
    try:
        from .materials import PALETTE_IMAGE_NAME, PALETTE_MATERIAL_NAME
        for mat_name in [f"{PALETTE_MATERIAL_NAME}_{vol_uuid}", PALETTE_MATERIAL_NAME]:
            m = bpy.data.materials.get(mat_name)
            if m and m.users == 0:
                bpy.data.materials.remove(m)
        for img_name in [f"{PALETTE_IMAGE_NAME}_{vol_uuid}", PALETTE_IMAGE_NAME]:
            im = bpy.data.images.get(img_name)
            if im and im.users == 0:
                bpy.data.images.remove(im)
    except Exception:
        pass

    mesh.update()
    return cpu_buffers
