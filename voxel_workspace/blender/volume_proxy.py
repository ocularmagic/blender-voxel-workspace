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
PROXY_ROOT_INSTANCE_UUID_FLAG = "voxel_root_instance_uuid"
PROXY_PALETTE_INDEX_FLAG = "palette_index"

# Render-only separation between an opaque Surface boundary and a Volume hull.
# This is deliberately far below a visible pixel at normal voxel scales.
VOLUME_CONTACT_GAP_RATIO = 1.0e-4


def _volume_cells(grid: Any, palette_index: int) -> Set[Tuple[int, int, int]]:
    """Return all cells occupied by one Volume palette index."""
    cells: Set[Tuple[int, int, int]] = set()
    s = int(grid.brick_size)
    tagged = hasattr(grid, "read_index_apron")
    for (bx, by, bz), brick in grid.bricks.items():
        values = brick.indices if tagged else brick
        mask = values == int(palette_index)
        if tagged:
            mask &= brick.domains == 2
        for lx, ly, lz in np.argwhere(mask):
            cells.add((bx * s + int(lx), by * s + int(ly), bz * s + int(lz)))
    return cells


def _face_connected_components(cells: Set[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], int]:
    """Label six-connected voxel components without joining edge/corner contacts."""
    labels: Dict[Tuple[int, int, int], int] = {}
    remaining = set(cells)
    component = 0
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        labels[seed] = component
        while stack:
            x, y, z = stack.pop()
            for neighbor in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z),
                             (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    labels[neighbor] = component
                    stack.append(neighbor)
        component += 1
    return labels


def _weld_proxy_quads(
    positions: np.ndarray,
    grid: Any,
    palette_index: int,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weld proxy quads per six-connected component and inset Surface contacts."""
    if len(positions) == 0:
        return positions, np.empty((0, 3), dtype=np.int32)
    if len(positions) % 4 != 0:
        raise ValueError("Volume proxy buffers must contain four vertices per quad")

    cells = _volume_cells(grid, palette_index)
    components = _face_connected_components(cells)
    epsilon = float(voxel_size) * VOLUME_CONTACT_GAP_RATIO
    tagged = hasattr(grid, "get_domain")
    quads = positions.reshape((-1, 4, 3))
    welded: List[np.ndarray] = []
    vertex_map: Dict[Tuple[int, Tuple[float, float, float]], int] = {}
    contact_normals: Dict[int, Set[Tuple[int, int, int]]] = {}
    quad_indices: List[Tuple[int, int, int, int]] = []

    for quad in quads:
        raw_normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        axis = int(np.argmax(np.abs(raw_normal)))
        sign = 1 if raw_normal[axis] > 0 else -1
        normal = np.zeros(3, dtype=np.int32)
        normal[axis] = sign
        center = np.mean(quad, axis=0)
        inside_point = center - normal.astype(np.float32) * (float(voxel_size) * 1.0e-3)
        inside = tuple(int(np.floor(float(v) / float(voxel_size))) for v in inside_point)
        component = components.get(inside)
        if component is None:
            raise ValueError(f"Could not associate Volume proxy face with cell {inside}")

        touches_surface = False
        if tagged:
            from ..core.tagged_grid import VoxelDomain
            other_axes = [candidate for candidate in range(3) if candidate != axis]
            plane = int(round(float(quad[0][axis]) / float(voxel_size)))
            inside_axis = plane - 1 if sign > 0 else plane
            ranges = []
            for other_axis in other_axes:
                lower = int(round(float(np.min(quad[:, other_axis])) / float(voxel_size)))
                upper = int(round(float(np.max(quad[:, other_axis])) / float(voxel_size)))
                ranges.append(range(lower, upper))
            for first in ranges[0]:
                for second in ranges[1]:
                    inside_cell = [0, 0, 0]
                    inside_cell[axis] = inside_axis
                    inside_cell[other_axes[0]] = first
                    inside_cell[other_axes[1]] = second
                    outside = tuple(inside_cell[i] + int(normal[i]) for i in range(3))
                    if grid.get_domain(outside) == VoxelDomain.SURFACE:
                        touches_surface = True
                        break
                if touches_surface:
                    break

        mapped: List[int] = []
        for point in quad:
            point_key = tuple(round(float(v), 7) for v in point)
            key = (component, point_key)
            vertex_index = vertex_map.get(key)
            if vertex_index is None:
                vertex_index = len(welded)
                vertex_map[key] = vertex_index
                welded.append(point.astype(np.float32).copy())
            mapped.append(vertex_index)
            if touches_surface:
                contact_normals.setdefault(vertex_index, set()).add(tuple(int(v) for v in normal))
        quad_indices.append((mapped[0], mapped[1], mapped[2], mapped[3]))

    for vertex_index, normals in contact_normals.items():
        displacement = np.zeros(3, dtype=np.float32)
        for normal in normals:
            displacement -= np.asarray(normal, dtype=np.float32) * epsilon
        welded[vertex_index] += displacement

    triangles = np.empty((len(quad_indices) * 2, 3), dtype=np.int32)
    for quad_index, (a, b, c, d) in enumerate(quad_indices):
        triangles[quad_index * 2] = (a, b, c)
        triangles[quad_index * 2 + 1] = (a, c, d)
    return np.asarray(welded, dtype=np.float32), triangles


def iter_roots_for_mesh(mesh: Any) -> List[Any]:
    """Find all canonical Voxel Root objects referencing the given mesh."""
    from .object_graph import iter_roots_for_mesh as _iter_roots
    return _iter_roots(mesh)


def iter_primary_objects_for_mesh(mesh: Any) -> List[Any]:
    """Find all canonical root objects (or fallback legacy objects) for the given mesh."""
    roots = iter_roots_for_mesh(mesh)
    if roots:
        return roots
    if bpy is None or mesh is None:
        return []
    legacy_objs = []
    for obj in bpy.data.objects:
        if (
            obj.type == 'MESH'
            and obj.data == mesh
            and not obj.get(PROXY_OBJECT_FLAG, False)
        ):
            legacy_objs.append(obj)
    return legacy_objs


def find_proxy(root_or_parent: Any, palette_index: int) -> Optional[Any]:
    """Find an existing volume proxy child object for the given root and palette index."""
    if root_or_parent is None:
        return None

    # Get owning root instance UUID and authoritative mesh UUID
    from .object_graph import is_voxel_root, resolve_voxel_root, resolve_authoritative_mesh
    root = root_or_parent if is_voxel_root(root_or_parent) else resolve_voxel_root(root_or_parent)
    mesh = resolve_authoritative_mesh(root_or_parent)
    source_uuid = getattr(mesh.voxel_workspace, "uuid", "") if (mesh and hasattr(mesh, "voxel_workspace")) else ""
    root_uuid = root.get("voxel_instance_uuid", "") if (root and hasattr(root, "get")) else ""

    # Inspect children of root (or fallback to root_or_parent)
    parent_target = root if root is not None else root_or_parent
    for child in getattr(parent_target, "children", []):
        if (
            child.get(PROXY_OBJECT_FLAG, False)
            and child.get(PROXY_PALETTE_INDEX_FLAG, -1) == palette_index
        ):
            child_mesh_uuid = child.get(PROXY_SOURCE_UUID_FLAG, "")
            child_root_uuid = child.get(PROXY_ROOT_INSTANCE_UUID_FLAG, "")
            # Validate source mesh UUID and root instance UUID if known
            if source_uuid and child_mesh_uuid and child_mesh_uuid != source_uuid:
                continue
            if root_uuid and child_root_uuid and child_root_uuid != root_uuid:
                continue
            return child

    # Also check fallback if parent_target was a legacy surface mesh and children are attached there
    if parent_target != root_or_parent:
        for child in getattr(root_or_parent, "children", []):
            if (
                child.get(PROXY_OBJECT_FLAG, False)
                and child.get(PROXY_PALETTE_INDEX_FLAG, -1) == palette_index
            ):
                return child

    return None


def ensure_proxy(root_or_parent: Any, mesh: Any, entry_item: Any) -> Any:
    """Ensure a child proxy object exists under Voxel Root for the specified volume palette entry."""
    if bpy is None or root_or_parent is None:
        return None

    from .object_graph import is_voxel_root, resolve_voxel_root
    root = root_or_parent if is_voxel_root(root_or_parent) else resolve_voxel_root(root_or_parent)
    parent_obj = root if root is not None else root_or_parent

    pal_idx = int(getattr(entry_item, "index", 1)) if entry_item is not None else 1
    proxy_obj = find_proxy(parent_obj, pal_idx)
    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "") if (mesh and hasattr(mesh, "voxel_workspace")) else ""
    root_uuid = parent_obj.get("voxel_instance_uuid", "") if hasattr(parent_obj, "get") else ""

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

        # Parent to Voxel Root with identity local transform
        proxy_obj.parent = parent_obj
        proxy_obj.matrix_local = np.eye(4)
        
        # Configure display mode and metadata
        proxy_obj.display_type = 'WIRE'
        proxy_obj.show_wire = True
        proxy_obj.hide_select = True
        proxy_obj[PROXY_OBJECT_FLAG] = True
        proxy_obj[PROXY_SOURCE_UUID_FLAG] = mesh_uuid
        proxy_obj[PROXY_ROOT_INSTANCE_UUID_FLAG] = root_uuid
        proxy_obj[PROXY_PALETTE_INDEX_FLAG] = pal_idx
        proxy_obj["voxel_render_role"] = "VOLUME"
    else:
        # Reparent to root if parent was not root
        if proxy_obj.parent != parent_obj:
            proxy_obj.parent = parent_obj
            proxy_obj.matrix_local = np.eye(4)

    # Repair metadata/display properties on existing proxies too.
    proxy_obj[PROXY_OBJECT_FLAG] = True
    proxy_obj[PROXY_SOURCE_UUID_FLAG] = mesh_uuid
    proxy_obj[PROXY_ROOT_INSTANCE_UUID_FLAG] = root_uuid
    proxy_obj[PROXY_PALETTE_INDEX_FLAG] = pal_idx
    proxy_obj["voxel_render_role"] = "VOLUME"
    proxy_obj.display_type = 'WIRE'
    proxy_obj.show_wire = True
    proxy_obj.hide_select = True
    proxy_obj.matrix_local = np.eye(4)

    # Ensure proxy material slot 0
    if entry_item is not None:
        mat = ensure_entry_material(mesh, entry_item, domain="VOLUME")
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

    is_tagged = hasattr(grid, "read_index_apron")

    for coord in remesh_targets:
        brick = grid.bricks.get(coord)
        has_val = False
        if brick is not None:
            if is_tagged:
                has_val = np.any((brick.domains == 2) & (brick.indices == palette_index))
            else:
                has_val = np.any(brick == palette_index)
        if has_val:
            if is_tagged:
                from ..core.tagged_grid import VoxelDomain
                apron = grid.read_index_apron(
                    coord,
                    domain_filter=VoxelDomain.VOLUME,
                    only_index=palette_index,
                )
                buf = mesh_greedy(
                    apron,
                    origin=(
                        float(coord[0] * s) * v_size,
                        float(coord[1] * s) * v_size,
                        float(coord[2] * s) * v_size,
                    ),
                    voxel_size=v_size,
                    only_index=palette_index,
                )
            else:
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
    vert_offset = 0

    for coord in sorted_coords:
        buf = proxy_buffers[coord]
        if buf.quad_count == 0 or len(buf.positions) == 0:
            continue
        all_positions.append(buf.positions)
        vert_offset += len(buf.positions)

    proxy_mesh.clear_geometry()
    if not all_positions or vert_offset == 0:
        proxy_mesh.update()
        return

    positions = np.concatenate(all_positions, axis=0)
    positions, indices = _weld_proxy_quads(
        positions,
        grid,
        palette_index,
        v_size,
    )
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


def remove_proxy(root_or_parent: Any, palette_index: int) -> bool:
    """Remove a proxy child object for the given root/parent and palette index."""
    if root_or_parent is None:
        return False
    proxy = find_proxy(root_or_parent, palette_index)
    if proxy is not None:
        mesh = proxy.data
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        return True
    return False


def reconcile_volume_proxies(
    root_or_parent: Any,
    mesh: Any,
    grid: VoxelGrid,
    volume_entry: Optional[Any] = None,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    voxel_size: float = 1.0,
) -> None:
    """Reconcile all volume proxies under a single Voxel Root instance."""
    if root_or_parent is None or mesh is None or grid is None:
        return

    from .object_graph import is_voxel_root, resolve_voxel_root
    root = root_or_parent if is_voxel_root(root_or_parent) else resolve_voxel_root(root_or_parent)
    parent_obj = root if root is not None else root_or_parent

    props = mesh.voxel_workspace
    mesh_uuid = props.uuid
    root_uuid = parent_obj.get("voxel_instance_uuid", "") if hasattr(parent_obj, "get") else ""

    for child in list(getattr(parent_obj, "children", [])):
        if child.get(PROXY_OBJECT_FLAG, False):
            child_mesh_uuid = child.get(PROXY_SOURCE_UUID_FLAG, "")
            child_root_uuid = child.get(PROXY_ROOT_INSTANCE_UUID_FLAG, "")
            if (mesh_uuid and child_mesh_uuid and child_mesh_uuid != mesh_uuid):
                child_mesh = child.data
                bpy.data.objects.remove(child, do_unlink=True)
                if child_mesh is not None and child_mesh.users == 0:
                    bpy.data.meshes.remove(child_mesh)

    vol_indices = used_volume_indices(mesh, grid)
    current_proxies = {
        c.get(PROXY_PALETTE_INDEX_FLAG, -1): c
        for c in getattr(parent_obj, "children", [])
        if c.get(PROXY_OBJECT_FLAG, False)
    }

    # 1. Remove proxies for indices that are no longer VOLUME or no longer used
    for pal_idx, proxy_obj in list(current_proxies.items()):
        if pal_idx not in vol_indices:
            remove_proxy(parent_obj, pal_idx)
            if volume_entry and pal_idx in volume_entry.volume_proxy_buffers:
                del volume_entry.volume_proxy_buffers[pal_idx]

    # 2. Ensure and rebuild geometry for each used VOLUME index
    from .material_domains import find_entry
    for pal_idx in vol_indices:
        # Typed Volume Palette is authoritative.
        entry_item = find_entry(mesh, "VOLUME", pal_idx)
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


def reconcile_all_root_instances(
    mesh: Any,
    grid: VoxelGrid,
    volume_entry: Optional[Any] = None,
    dirty_bricks: Optional[Set[BrickCoord]] = None,
    voxel_size: float = 1.0,
) -> None:
    """Reconcile volume proxies across all Voxel Root instances sharing the given mesh."""
    root_objs = iter_roots_for_mesh(mesh)
    if not root_objs:
        # Fallback to primary objects if no root found
        root_objs = iter_primary_objects_for_mesh(mesh)
    for root in root_objs:
        reconcile_volume_proxies(
            root,
            mesh,
            grid,
            volume_entry=volume_entry,
            dirty_bricks=dirty_bricks,
            voxel_size=voxel_size,
        )


reconcile_all_instances = reconcile_all_root_instances


def cleanup_stale_proxies() -> List[str]:
    """Clean up orphan volume proxy objects whose root or source mesh no longer exists."""
    if bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "objects"):
        return []

    active_mesh_uuids = set()
    for m in bpy.data.meshes:
        if hasattr(m, "voxel_workspace") and m.voxel_workspace.uuid:
            active_mesh_uuids.add(m.voxel_workspace.uuid)

    from .object_graph import is_voxel_root, resolve_authoritative_mesh

    removed = []
    for obj in list(bpy.data.objects):
        if obj.get(PROXY_OBJECT_FLAG, False):
            source_uuid = obj.get(PROXY_SOURCE_UUID_FLAG, "")
            parent = obj.parent
            parent_is_valid_root = parent is not None and is_voxel_root(parent)
            parent_mesh = resolve_authoritative_mesh(parent) if parent else None
            parent_mesh_uuid = getattr(parent_mesh.voxel_workspace, "uuid", "") if (parent_mesh and hasattr(parent_mesh, "voxel_workspace")) else ""

            if parent is None or source_uuid not in active_mesh_uuids or (parent_mesh_uuid and parent_mesh_uuid != source_uuid):
                m = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if m and m.users == 0:
                    bpy.data.meshes.remove(m)
                removed.append(obj.name)

    return removed
