"""In-memory runtime cache, active volume management, and lifecycle handling."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid as uuid_lib
import numpy as np

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
    volume_proxy_buffers: Dict[int, Dict[BrickCoord, MeshBuffers]] = field(default_factory=dict)
    dirty_bricks: Set[BrickCoord] = field(default_factory=set)
    voxel_size: float = 1.0
    palette_lut: Optional[np.ndarray] = None


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
    Also forks palette datablocks (material.copy, image.copy, rebind texture node,
    assign slot 0, rename) because mesh.copy copies the IDProperty palette by value
    but the materials list by reference.
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

                # Fork palette datablocks for the duplicated mesh
                props = mesh.voxel_workspace
                if hasattr(props, "palette"):
                    from .material_domains import copy_entry_material_for_mesh
                    for entry in props.palette:
                        if entry.index > 0:
                            copy_entry_material_for_mesh(entry, entry, new_uuid)

                # Rebuild the dense native slot list from the forked entry
                # pointers. Never inject the retired atlas material into slot 0.
                from .material_domains import reconcile_surface_slots, cleanup_legacy_atlas_datablocks
                runtime_entry = get_or_load(mesh)
                if runtime_entry is not None:
                    reconcile_surface_slots(mesh, runtime_entry.grid)
                cleanup_legacy_atlas_datablocks(mesh)

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


def reconcile_all_palette_caches(pack_images: bool = False) -> None:
    """Reconcile native material domain bindings, slots, and GPU preview LUTs for all voxel meshes."""
    if bpy is None or not hasattr(bpy, "data") or not hasattr(bpy.data, "meshes"):
        return
    from .properties import ensure_palette, migrate_native_material_domains
    from .material_domains import cleanup_legacy_atlas_datablocks
    from .volume_proxy import cleanup_stale_proxies
    from .mesh_sync import sync_volume_mesh
    from .gpu_preview import drop_palette_lut

    cleanup_stale_proxies()

    for mesh in bpy.data.meshes:
        if hasattr(mesh, "voxel_workspace") and mesh.voxel_workspace.is_voxel_mesh:
            if len(mesh.voxel_workspace.palette) == 0:
                ensure_palette(mesh)
            migrate_native_material_domains(mesh)
            
            entry = get_or_load(mesh)
            if entry is not None and entry.grid is not None:
                entry.cpu_buffers.clear()
                entry.volume_proxy_buffers.clear()
                sync_volume_mesh(mesh, grid=entry.grid, entry=entry, dirty_only=False, ensure_material=False)

            cleanup_legacy_atlas_datablocks(mesh)

            drop_palette_lut(mesh.voxel_workspace.uuid)

    tag_redraw_all_viewports()


def _reconcile_timer_callback() -> None:
    """Main-thread timer callback for deferred post-undo/post-redo cache reconciliation."""
    reconcile_all_palette_caches(pack_images=False)
    return None


def _schedule_palette_reconcile() -> None:
    """Schedule one cache reconciliation after Blender's handler returns."""
    if bpy is None or not hasattr(bpy.app, "timers"):
        return
    try:
        if not bpy.app.timers.is_registered(_reconcile_timer_callback):
            bpy.app.timers.register(_reconcile_timer_callback, first_interval=0.0)
    except Exception:
        pass


@persistent
def on_load_post(*args) -> None:
    """File load handler: reset runtime cache and reconcile palette datablocks."""
    try:
        from .gpu_preview import cleanup_gpu_preview
        cleanup_gpu_preview()
    except Exception:
        pass
    clear_registry()
    reconcile_all_palette_caches(pack_images=False)


@persistent
def on_save_pre(*args) -> None:
    """Pre-save handler: flush dirty runtime grids to Mesh IDProperties and pack palette images."""
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
    Schedules main-thread cache reconciliation after the handler returns.
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
            from .gpu_preview import clear_hover_state, drop_palette_lut
            clear_hover_state()
            drop_palette_lut()
        except Exception:
            pass

        # Reconcile only after this handler returns; handlers must remain read-only.
        _schedule_palette_reconcile()

        tag_redraw_all_viewports()
    finally:
        _UNDO_GUARD = False


@persistent
def on_redo_post(*args) -> None:
    """Post-redo handler: invalidate runtime cache, clear hover/GPU state, tag viewports.
    
    Guarded against reentrancy. Performs NO datablock writes and pushes NO undo steps.
    Schedules main-thread cache reconciliation after the handler returns.
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
            from .gpu_preview import clear_hover_state, drop_palette_lut
            clear_hover_state()
            drop_palette_lut()
        except Exception:
            pass

        # Reconcile only after this handler returns; handlers must remain read-only.
        _schedule_palette_reconcile()

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
    if hasattr(bpy.app, "timers") and bpy.app.timers.is_registered(_reconcile_timer_callback):
        bpy.app.timers.unregister(_reconcile_timer_callback)
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
