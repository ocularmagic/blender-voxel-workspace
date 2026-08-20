"""In-memory runtime cache, active volume management, and lifecycle handling."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid as uuid_lib

try:
    import bpy
    from bpy.app.handlers import persistent
except ImportError:
    bpy = None
    persistent = lambda f: f

from ..constants import BRICK_SIZE, BrickCoord
from ..core.grid import VoxelGrid
from ..geometry.buffers import MeshBuffers


@dataclass
class VoxelVolumeEntry:
    """In-memory runtime state for a single voxel volume."""
    uuid: str
    grid: VoxelGrid
    cpu_buffers: Dict[BrickCoord, MeshBuffers] = field(default_factory=dict)
    gpu_batches: Dict[BrickCoord, Any] = field(default_factory=dict)
    gpu_edge_batches: Dict[BrickCoord, Any] = field(default_factory=dict)
    dirty_bricks: Set[BrickCoord] = field(default_factory=set)
    voxel_size: float = 1.0


_REGISTRY: Dict[str, VoxelVolumeEntry] = {}
_ACTIVE_VOLUME_UUID: Optional[str] = None
_DEDUPE_GUARD: bool = False
_UNDO_GUARD: bool = False


def get_active_volume_uuid() -> Optional[str]:
    """Return the currently active volume UUID for editing, or None."""
    return _ACTIVE_VOLUME_UUID


def set_active_volume_uuid(uuid_str: Optional[str]) -> None:
    """Set the currently active volume UUID for editing (at most one active)."""
    global _ACTIVE_VOLUME_UUID
    _ACTIVE_VOLUME_UUID = uuid_str


def get_active_volume() -> Optional[VoxelVolumeEntry]:
    """Return the active volume's runtime entry, or None."""
    if _ACTIVE_VOLUME_UUID is None:
        return None
    return _REGISTRY.get(_ACTIVE_VOLUME_UUID)


def register_volume(
    uuid_str: str,
    grid: Optional[VoxelGrid] = None,
    voxel_size: float = 1.0,
    brick_size: int = BRICK_SIZE,
    extent_min: tuple[int, int, int] = (0, 0, 0),
    extent_max: tuple[int, int, int] = (32, 32, 32),
) -> VoxelVolumeEntry:
    """Register or replace a volume in the runtime cache."""
    if not uuid_str:
        raise ValueError("Cannot register volume with empty UUID")
    if grid is None:
        grid = VoxelGrid(extent_min=extent_min, extent_max_exclusive=extent_max, brick_size=brick_size)
    entry = VoxelVolumeEntry(
        uuid=uuid_str,
        grid=grid,
        voxel_size=voxel_size,
    )
    _REGISTRY[uuid_str] = entry
    return entry


def invalidate_uuid(uuid_str: str) -> bool:
    """Evict a volume entry from the runtime cache by UUID.
    
    Returns True if the volume was in cache and removed, False otherwise.
    """
    if uuid_str in _REGISTRY:
        del _REGISTRY[uuid_str]
        return True
    return False


def get_or_load(mesh: Any) -> Optional[VoxelVolumeEntry]:
    """Retrieve runtime entry for a voxel mesh datablock, lazily deserializing if not cached."""
    if mesh is None:
        return None
    mesh_data = getattr(mesh, "data", mesh)
    if mesh_data is None or not hasattr(mesh_data, "voxel_workspace"):
        return None
    uuid_str = getattr(mesh_data.voxel_workspace, "uuid", None)
    if not uuid_str:
        return None

    entry = _REGISTRY.get(uuid_str)
    if entry is not None:
        return entry

    from .persistence import deserialize_volume
    grid = deserialize_volume(mesh_data)
    voxel_size = float(getattr(mesh_data.voxel_workspace, "voxel_size", 1.0))
    entry = register_volume(uuid_str, grid=grid, voxel_size=voxel_size)
    return entry


def unregister_volume(uuid_str: str) -> None:
    """Remove a volume from the runtime cache."""
    global _ACTIVE_VOLUME_UUID
    if uuid_str in _REGISTRY:
        del _REGISTRY[uuid_str]
    if _ACTIVE_VOLUME_UUID == uuid_str:
        _ACTIVE_VOLUME_UUID = None


def get_volume(uuid_str: str) -> Optional[VoxelVolumeEntry]:
    """Lookup a volume runtime entry by UUID."""
    return _REGISTRY.get(uuid_str)


def has_volume(uuid_str: str) -> bool:
    """Check if a volume UUID is in the runtime cache."""
    return uuid_str in _REGISTRY


def clear_registry() -> None:
    """Clear all runtime entries and reset active volume."""
    global _ACTIVE_VOLUME_UUID
    _REGISTRY.clear()
    _ACTIVE_VOLUME_UUID = None


def all_volumes() -> Dict[str, VoxelVolumeEntry]:
    """Return a shallow copy of the runtime registry dictionary."""
    return dict(_REGISTRY)


def deduplicate_mesh_uuids(scene=None, depsgraph=None) -> List[Tuple[Any, str, str]]:
    """Scan bpy.data.meshes for duplicate UUIDs and repair them with new UUIDs.
    
    Guarded against reentrancy and idempotent.
    Returns a list of repaired tuples: (mesh, old_uuid, new_uuid).
    """
    global _DEDUPE_GUARD
    if _DEDUPE_GUARD or bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "meshes"):
        return []

    _DEDUPE_GUARD = True
    repaired: List[Tuple[Any, str, str]] = []
    try:
        seen_uuids: Dict[str, Any] = {}
        for mesh in bpy.data.meshes:
            if not hasattr(mesh, "voxel_workspace"):
                continue
            vol_uuid = mesh.voxel_workspace.uuid
            if not vol_uuid:
                continue
            if vol_uuid not in seen_uuids:
                seen_uuids[vol_uuid] = mesh
            else:
                # Duplicate UUID on a distinct Mesh datablock!
                new_uuid = str(uuid_lib.uuid4())
                old_uuid = vol_uuid
                mesh.voxel_workspace.uuid = new_uuid
                repaired.append((mesh, old_uuid, new_uuid))
                seen_uuids[new_uuid] = mesh
    finally:
        _DEDUPE_GUARD = False

    return repaired


def cleanup_stale_volumes() -> List[str]:
    """Remove runtime entries for UUIDs that no longer exist in any bpy.data.meshes."""
    global _ACTIVE_VOLUME_UUID
    if bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "meshes"):
        return []

    active_mesh_uuids = set()
    for mesh in bpy.data.meshes:
        if hasattr(mesh, "voxel_workspace") and mesh.voxel_workspace.uuid:
            active_mesh_uuids.add(mesh.voxel_workspace.uuid)

    stale_uuids = [u for u in _REGISTRY if u not in active_mesh_uuids]
    for u in stale_uuids:
        del _REGISTRY[u]
        if _ACTIVE_VOLUME_UUID == u:
            _ACTIVE_VOLUME_UUID = None

    return stale_uuids


def tag_redraw_all_viewports() -> None:
    """Tag redraw on all 3D Viewport areas across all windows safely."""
    if bpy is None or not hasattr(bpy, "context") or not hasattr(bpy.context, "window_manager"):
        return
    wm = bpy.context.window_manager
    if wm is None or not hasattr(wm, "windows"):
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def on_depsgraph_update(scene, depsgraph) -> None:
    """Depsgraph update handler: trigger guarded UUID deduplication."""
    deduplicate_mesh_uuids(scene, depsgraph)


@persistent
def on_load_post(*args) -> None:
    """File load handler: reset runtime cache across file opens."""
    try:
        from .gpu_preview import cleanup_gpu_preview
        cleanup_gpu_preview()
    except Exception:
        pass
    clear_registry()


@persistent
def on_save_pre(*args) -> None:
    """Pre-save handler: flush dirty runtime grids to Mesh IDProperties."""
    if bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "meshes"):
        return
    from .persistence import serialize_volume

    mesh_by_uuid: Dict[str, Any] = {}
    for mesh in bpy.data.meshes:
        if hasattr(mesh, "voxel_workspace") and mesh.voxel_workspace.uuid:
            mesh_by_uuid[mesh.voxel_workspace.uuid] = mesh

    for vol_uuid, entry in list(_REGISTRY.items()):
        mesh = mesh_by_uuid.get(vol_uuid)
        if mesh is not None and (entry.grid.dirty_bricks or entry.dirty_bricks):
            serialize_volume(mesh, entry.grid, dirty_only=True)
            entry.dirty_bricks.clear()


@persistent
def on_undo_post(*args) -> None:
    """Post-undo handler: invalidate runtime cache, clear hover/GPU state, tag viewports.
    
    Guarded against reentrancy. Performs NO datablock writes and pushes NO undo steps.
    Preserves active editing UUID when matching mesh still exists.
    """
    global _UNDO_GUARD, _ACTIVE_VOLUME_UUID
    if _UNDO_GUARD:
        return
    _UNDO_GUARD = True
    try:
        # Preserve active editing UUID if matching mesh still exists
        if _ACTIVE_VOLUME_UUID:
            mesh_found = False
            if bpy is not None and hasattr(bpy, "data") and hasattr(bpy.data, "meshes"):
                for m in bpy.data.meshes:
                    if hasattr(m, "voxel_workspace") and m.voxel_workspace.uuid == _ACTIVE_VOLUME_UUID:
                        mesh_found = True
                        break
            if not mesh_found:
                _ACTIVE_VOLUME_UUID = None

        _REGISTRY.clear()

        try:
            from .gpu_preview import clear_hover_state
            clear_hover_state()
        except Exception:
            pass

        tag_redraw_all_viewports()
    finally:
        _UNDO_GUARD = False


@persistent
def on_redo_post(*args) -> None:
    """Post-redo handler: invalidate runtime cache, clear hover/GPU state, tag viewports.
    
    Guarded against reentrancy. Performs NO datablock writes and pushes NO undo steps.
    Preserves active editing UUID when matching mesh still exists.
    """
    global _UNDO_GUARD, _ACTIVE_VOLUME_UUID
    if _UNDO_GUARD:
        return
    _UNDO_GUARD = True
    try:
        # Preserve active editing UUID if matching mesh still exists
        if _ACTIVE_VOLUME_UUID:
            mesh_found = False
            if bpy is not None and hasattr(bpy, "data") and hasattr(bpy.data, "meshes"):
                for m in bpy.data.meshes:
                    if hasattr(m, "voxel_workspace") and m.voxel_workspace.uuid == _ACTIVE_VOLUME_UUID:
                        mesh_found = True
                        break
            if not mesh_found:
                _ACTIVE_VOLUME_UUID = None

        _REGISTRY.clear()

        try:
            from .gpu_preview import clear_hover_state
            clear_hover_state()
        except Exception:
            pass

        tag_redraw_all_viewports()
    finally:
        _UNDO_GUARD = False


def register_runtime() -> None:
    """Register lifecycle handlers idempotently."""
    if bpy is None:
        return
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    if on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load_post)
    if on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(on_save_pre)
    if on_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(on_undo_post)
    if on_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(on_redo_post)


def unregister_runtime() -> None:
    """Unregister lifecycle handlers and clean up registry."""
    clear_registry()
    if bpy is None:
        return
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    if on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(on_save_pre)
    if on_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(on_undo_post)
    if on_redo_post in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(on_redo_post)
