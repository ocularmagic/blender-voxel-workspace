"""Volume proxy object lifecycle and reconciliation for volumetric palette entries."""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import bpy
    from bpy.types import Mesh, Object
except ImportError:
    bpy = None
    Mesh = Object = object

from ..constants import BrickCoord
from ..core.grid import VoxelGrid
from ..geometry.buffers import MeshBuffers
from ..geometry.greedy import mesh_greedy
from .material_domains import used_volume_indices, ensure_entry_material


PROXY_OBJECT_FLAG = "is_voxel_volume_proxy"
PROXY_SOURCE_UUID_FLAG = "source_mesh_uuid"
PROXY_PALETTE_INDEX_FLAG = "palette_index"


def iter_primary_objects_for_mesh(mesh: Any) -> List[Any]:
    """Find all Blender objects in the scene referencing the given mesh."""
    if bpy is None or mesh is None:
        return []
    primary_objs = []
    for obj in bpy.data.objects:
        if (
            obj.type == 'MESH'
            and obj.data == mesh
            and not obj.get(PROXY_OBJECT_FLAG, False)
        ):
            primary_objs.append(obj)
    return primary_objs


def find_proxy(parent_obj: Any, palette_index: int) -> Optional[Any]:
    """Find an existing volume proxy child object for the given parent object and palette index."""
    if parent_obj is None:
        return None
    for child in parent_obj.children:
        if (
            child.get(PROXY_OBJECT_FLAG, False)
            and child.get(PROXY_PALETTE_INDEX_FLAG, -1) == palette_index
        ):
            return child
    return None


def ensure_proxy(parent_obj: Any, mesh: Any, entry_item: Any) -> Any:
    """Ensure a child proxy object exists for parent_obj and the specified volume palette entry."""
    if bpy is None or parent_obj is None or entry_item is None:
        return None

    pal_idx = int(entry_item.index)
    proxy_obj = find_proxy(parent_obj, pal_idx)
    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "")

    if proxy_obj is None:
        proxy_mesh = bpy.data.meshes.new(name=f"VoxelVolumeProxyMesh_{pal_idx}")
        proxy_obj = bpy.data.objects.new(name=f"{parent_obj.name}_VolumeProxy_{pal_idx}", object_data=proxy_mesh)
        
        # Link to parent object's collection
        target_col = None
        if parent_obj.users_collection:
            target_col = parent_obj.users_collection[0]
        elif hasattr(bpy.context, "scene") and bpy.context.scene:
            target_col = bpy.context.scene.collection
        if target_col:
            target_col.objects.link(proxy_obj)

        # Parent to primary object with identity local transform
        proxy_obj.parent = parent_obj
        proxy_obj.matrix_local = np.eye(4)
        
        # Configure display mode and metadata
        proxy_obj.display_type = 'WIRE'
        proxy_obj.show_wire = True
        proxy_obj[PROXY_OBJECT_FLAG] = True
        proxy_obj[PROXY_SOURCE_UUID_FLAG] = mesh_uuid
        proxy_obj[PROXY_PALETTE_INDEX_FLAG] = pal_idx

    # Ensure proxy material slot 0
    mat = ensure_entry_material(mesh, entry_item)
    if len(proxy_obj.data.materials) == 0:
        proxy_obj.data.materials.append(mat)
    else:
        proxy_obj.data.materials[0] = mat

    return proxy_obj


def rebuild_proxy_geometry(
    proxy_obj: Any,
    grid: VoxelGrid,
    palette_index: int,
    volume_entry: Optional[Any] = None,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    voxel_size: float = 1.0,
) -> None:
    """Rebuild the closed-hull proxy geometry for a single volume index."""
    if proxy_obj is None or grid is None or proxy_obj.data is None:
        return

    proxy_mesh = proxy_obj.data
    s = grid.brick_size
    v_size = voxel_size

    # Cache lookup
    proxy_buffers: Dict[BrickCoord, MeshBuffers]
    if volume_entry is not None:
        if palette_index not in volume_entry.volume_proxy_buffers:
            volume_entry.volume_proxy_buffers[palette_index] = {}
        proxy_buffers = volume_entry.volume_proxy_buffers[palette_index]
    else:
        proxy_buffers = {}

    # Bricks to remesh
    if dirty_bricks is not None and len(proxy_buffers) > 0:
        remesh_targets: Set[BrickCoord] = set()
        for bx, by, bz in dirty_bricks:
            remesh_targets.add((bx, by, bz))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n_coord = (bx + dx, by + dy, bz + dz)
                        if n_coord in grid.bricks or n_coord in proxy_buffers:
                            remesh_targets.add(n_coord)
    else:
        remesh_targets = set(grid.bricks.keys()) | set(proxy_buffers.keys())

    for coord in remesh_targets:
        brick = grid.bricks.get(coord)
        if brick is not None and np.any(brick == palette_index):
            apron = grid.read_apron(coord)
            origin = (
                float(coord[0] * s) * v_size,
                float(coord[1] * s) * v_size,
                float(coord[2] * s) * v_size,
            )
            buf = mesh_greedy(
                apron,
                origin=origin,
                voxel_size=v_size,
                brick=brick,
                only_index=palette_index,
            )
            if buf.quad_count > 0:
                proxy_buffers[coord] = buf
            else:
                proxy_buffers.pop(coord, None)
        else:
            proxy_buffers.pop(coord, None)

    # Concatenate buffers
    sorted_coords = sorted(proxy_buffers.keys())
    all_positions = []
    all_indices = []
    vert_offset = 0

    for coord in sorted_coords:
        buf = proxy_buffers[coord]
        if buf.quad_count == 0 or len(buf.positions) == 0:
            continue
        all_positions.append(buf.positions)
        all_indices.append(buf.indices + vert_offset)
        vert_offset += len(buf.positions)

    proxy_mesh.clear_geometry()
    if not all_positions or vert_offset == 0:
        proxy_mesh.update()
        return

    positions = np.concatenate(all_positions, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    total_verts = len(positions)
    total_tris = len(indices)
    total_loops = total_tris * 3
    loop_vertex_indices = indices.ravel().astype(np.int32)

    proxy_mesh.vertices.add(total_verts)
    proxy_mesh.vertices.foreach_set("co", positions.ravel())

    proxy_mesh.loops.add(total_loops)
    proxy_mesh.loops.foreach_set("vertex_index", loop_vertex_indices)

    proxy_mesh.polygons.add(total_tris)
    proxy_mesh.polygons.foreach_set("loop_start", np.arange(0, total_loops, 3, dtype=np.int32))
    proxy_mesh.polygons.foreach_set("loop_total", np.full(total_tris, 3, dtype=np.int32))
    proxy_mesh.polygons.foreach_set("material_index", np.zeros(total_tris, dtype=np.int32))

    proxy_mesh.update()


def remove_proxy(parent_obj: Any, palette_index: int) -> bool:
    """Remove a proxy child object for the given parent and palette index."""
    if parent_obj is None:
        return False
    proxy = find_proxy(parent_obj, palette_index)
    if proxy is not None:
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        return True
    return False


def reconcile_volume_proxies(
    parent_obj: Any,
    mesh: Any,
    grid: VoxelGrid,
    volume_entry: Optional[Any] = None,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    voxel_size: float = 1.0,
) -> None:
    """Reconcile all volume proxies for a single primary object instance."""
    if parent_obj is None or mesh is None or grid is None:
        return

    props = mesh.voxel_workspace
    vol_indices = used_volume_indices(mesh, grid)
    current_proxies = {
        c.get(PROXY_PALETTE_INDEX_FLAG, -1): c
        for c in parent_obj.children
        if c.get(PROXY_OBJECT_FLAG, False)
    }

    # 1. Remove proxies for indices that are no longer VOLUME or no longer used
    for pal_idx, proxy_obj in list(current_proxies.items()):
        if pal_idx not in vol_indices:
            remove_proxy(parent_obj, pal_idx)
            if volume_entry and pal_idx in volume_entry.volume_proxy_buffers:
                del volume_entry.volume_proxy_buffers[pal_idx]

    # 2. Ensure and rebuild geometry for each used VOLUME index
    palette_lookup = {e.index: e for e in props.palette}
    for pal_idx in vol_indices:
        entry_item = palette_lookup.get(pal_idx)
        if entry_item is None:
            continue
        proxy_obj = ensure_proxy(parent_obj, mesh, entry_item)
        rebuild_proxy_geometry(
            proxy_obj,
            grid,
            pal_idx,
            volume_entry=volume_entry,
            dirty_bricks=dirty_bricks,
            voxel_size=voxel_size,
        )


def reconcile_all_instances(
    mesh: Any,
    grid: VoxelGrid,
    volume_entry: Optional[Any] = None,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    voxel_size: float = 1.0,
) -> None:
    """Reconcile volume proxies across all primary Object instances sharing the given mesh."""
    primary_objs = iter_primary_objects_for_mesh(mesh)
    for parent_obj in primary_objs:
        reconcile_volume_proxies(
            parent_obj,
            mesh,
            grid,
            volume_entry=volume_entry,
            dirty_bricks=dirty_bricks,
            voxel_size=voxel_size,
        )


def cleanup_stale_proxies() -> List[str]:
    """Clean up orphan volume proxy objects whose parent or source mesh no longer exists."""
    if bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "objects"):
        return []

    active_mesh_uuids = set()
    for m in bpy.data.meshes:
        if hasattr(m, "voxel_workspace") and m.voxel_workspace.uuid:
            active_mesh_uuids.add(m.voxel_workspace.uuid)

    removed = []
    for obj in list(bpy.data.objects):
        if obj.get(PROXY_OBJECT_FLAG, False):
            source_uuid = obj.get(PROXY_SOURCE_UUID_FLAG, "")
            parent = obj.parent
            if parent is None or source_uuid not in active_mesh_uuids:
                m = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if m and m.users == 0:
                    bpy.data.meshes.remove(m)
                removed.append(obj.name)

    return removed
