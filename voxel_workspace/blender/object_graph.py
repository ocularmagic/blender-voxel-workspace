"""Canonical hierarchy resolution, object graph navigation, and structure repair."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple
import uuid

try:
    import bpy
    from bpy.types import Mesh, Object
except ImportError:
    bpy = None
    Mesh = Object = object

from ..blender.volume_proxy import (
    PROXY_OBJECT_FLAG,
    PROXY_SOURCE_UUID_FLAG,
    PROXY_PALETTE_INDEX_FLAG,
)


VOXEL_ROOT_FLAG = "is_voxel_root"
VOXEL_INSTANCE_UUID_FLAG = "voxel_instance_uuid"
VOXEL_RENDER_ROLE_FLAG = "voxel_render_role"
VOXEL_ROOT_INSTANCE_UUID_FLAG = "voxel_root_instance_uuid"

VOXEL_FIELD_COLLECTION_NAME = "Voxel Field"


@dataclass
class VoxelResolvedContext:
    root: Any
    surface_object: Any
    mesh: Any
    mesh_uuid: str
    root_instance_uuid: str
    runtime_entry: Any = None


@dataclass
class RepairReport:
    repaired_roots: int = 0
    repaired_surfaces: int = 0
    reparented_children: int = 0
    removed_stale_children: int = 0
    messages: List[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


def is_voxel_root(obj: Any) -> bool:
    """Return True if obj is a canonical Voxel Root Empty object."""
    if obj is None or getattr(obj, "type", None) != 'EMPTY':
        return False
    if hasattr(obj, "get") and obj.get(VOXEL_ROOT_FLAG, False):
        return True
    # Also check custom properties if registered
    props = getattr(obj, "voxel_workspace", None)
    if props is not None and getattr(props, "is_voxel_root", False):
        return True
    return False


def is_surface_render_object(obj: Any) -> bool:
    """Return True if obj is a generated or legacy Surface Mesh render object."""
    if obj is None or getattr(obj, "type", None) != 'MESH':
        return False
    if hasattr(obj, "get") and obj.get(PROXY_OBJECT_FLAG, False):
        return False
    role = obj.get(VOXEL_RENDER_ROLE_FLAG, None) if hasattr(obj, "get") else None
    if role == "SURFACE":
        return True
    # Legacy check
    props = getattr(obj, "voxel_workspace", None)
    if props is not None and getattr(props, "is_voxel_object", False):
        return True
    mesh = getattr(obj, "data", None)
    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        return getattr(mesh.voxel_workspace, "is_voxel_mesh", False)
    return False


def is_volume_render_object(obj: Any) -> bool:
    """Return True if obj is a derived Volume proxy render object."""
    if obj is None or getattr(obj, "type", None) != 'MESH':
        return False
    if hasattr(obj, "get") and obj.get(PROXY_OBJECT_FLAG, False):
        return True
    role = obj.get(VOXEL_RENDER_ROLE_FLAG, None) if hasattr(obj, "get") else None
    return role == "VOLUME"


def _extract_object(obj_or_context: Any) -> Optional[Any]:
    """Helper to unwrap an Object from context or return obj."""
    if obj_or_context is None:
        return None
    if hasattr(obj_or_context, "active_object"):
        return obj_or_context.active_object
    if getattr(obj_or_context, "type", None) in ('EMPTY', 'MESH', 'CURVE', 'ARMATURE', 'CAMERA', 'LIGHT'):
        return obj_or_context
    return None


def resolve_voxel_root(obj_or_context: Any) -> Optional[Any]:
    """Resolve the canonical Voxel Root Empty from an object or context."""
    obj = _extract_object(obj_or_context)
    if obj is None:
        return None

    # 1. If it's already the root
    if is_voxel_root(obj):
        return obj

    # 2. If it's a child (Surface or Volume) whose parent is a root
    if getattr(obj, "parent", None) is not None and is_voxel_root(obj.parent):
        return obj.parent

    # 3. If it's a Surface child object that has a root_instance_uuid tag
    root_uuid = obj.get(VOXEL_ROOT_INSTANCE_UUID_FLAG, "") if hasattr(obj, "get") else ""
    if root_uuid and bpy is not None:
        for scene_obj in bpy.data.objects:
            if is_voxel_root(scene_obj) and scene_obj.get(VOXEL_INSTANCE_UUID_FLAG, "") == root_uuid:
                return scene_obj

    # 4. If it's a legacy Surface mesh object with no parent, find or create root
    if is_surface_render_object(obj):
        return ensure_root_for_surface(obj)

    return None


def resolve_surface_object(obj_or_root: Any) -> Optional[Any]:
    """Resolve the authoritative Surface child object from a root or child object."""
    obj = _extract_object(obj_or_root)
    if obj is None:
        return None

    # If passed a surface object directly
    if is_surface_render_object(obj):
        return obj

    # If passed a root
    if is_voxel_root(obj):
        # First check root pointer property if present
        props = getattr(obj, "voxel_workspace", None)
        if props is not None and hasattr(props, "surface_object"):
            surf = getattr(props, "surface_object", None)
            if surf is not None and is_surface_render_object(surf):
                return surf

        # Check direct children
        for child in getattr(obj, "children", []):
            if is_surface_render_object(child):
                return child

        # Check matching root instance UUID across objects
        root_uuid = obj.get(VOXEL_INSTANCE_UUID_FLAG, "") if hasattr(obj, "get") else ""
        if root_uuid and bpy is not None:
            for scene_obj in bpy.data.objects:
                if scene_obj.get(VOXEL_ROOT_INSTANCE_UUID_FLAG, "") == root_uuid and is_surface_render_object(scene_obj):
                    return scene_obj

    # If passed a volume proxy child
    if is_volume_render_object(obj):
        parent = getattr(obj, "parent", None)
        if parent is not None:
            if is_voxel_root(parent):
                return resolve_surface_object(parent)
            if is_surface_render_object(parent):
                return parent

    return None


def resolve_authoritative_mesh(obj_or_root: Any) -> Optional[Any]:
    """Resolve the authoritative Mesh datablock holding voxel storage and palettes."""
    if obj_or_root is not None and getattr(obj_or_root, "type", None) == 'MESH' and not hasattr(obj_or_root, "data"):
        # Datablock passed directly (e.g. bpy.types.Mesh)
        if hasattr(obj_or_root, "voxel_workspace") and obj_or_root.voxel_workspace.is_voxel_mesh:
            return obj_or_root
    surface_obj = resolve_surface_object(obj_or_root)
    if surface_obj is not None and hasattr(surface_obj, "data"):
        mesh = surface_obj.data
        if mesh is not None and hasattr(mesh, "voxel_workspace") and mesh.voxel_workspace.is_voxel_mesh:
            return mesh
    return None


def resolve_volume_context(context_or_obj: Any) -> Optional[VoxelResolvedContext]:
    """Resolve full VoxelResolvedContext from context or object."""
    obj = _extract_object(context_or_obj)
    if obj is None:
        return None

    root = resolve_voxel_root(obj)
    surface_obj = resolve_surface_object(obj) if root is None else resolve_surface_object(root)

    if surface_obj is None and is_surface_render_object(obj):
        surface_obj = obj

    if surface_obj is None or getattr(surface_obj, "data", None) is None:
        return None

    mesh = surface_obj.data
    if not hasattr(mesh, "voxel_workspace") or not mesh.voxel_workspace.is_voxel_mesh:
        return None

    mesh_uuid = mesh.voxel_workspace.uuid or ""
    
    root_uuid = ""
    if root is not None:
        root_uuid = root.get(VOXEL_INSTANCE_UUID_FLAG, "") if hasattr(root, "get") else ""
        if not root_uuid:
            root_uuid = str(uuid.uuid4())
            root[VOXEL_INSTANCE_UUID_FLAG] = root_uuid
    else:
        root_uuid = surface_obj.get(VOXEL_ROOT_INSTANCE_UUID_FLAG, mesh_uuid) if hasattr(surface_obj, "get") else mesh_uuid

    from .runtime import get_volume
    runtime_entry = get_volume(mesh_uuid) if mesh_uuid else None

    return VoxelResolvedContext(
        root=root,
        surface_object=surface_obj,
        mesh=mesh,
        mesh_uuid=mesh_uuid,
        root_instance_uuid=root_uuid,
        runtime_entry=runtime_entry,
    )


def iter_roots_for_mesh(mesh: Any) -> List[Any]:
    """Find all Voxel Root objects that reference the given Mesh datablock."""
    if bpy is None or mesh is None:
        return []
    roots = []
    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "") if hasattr(mesh, "voxel_workspace") else ""
    for obj in bpy.data.objects:
        if is_voxel_root(obj):
            surf = resolve_surface_object(obj)
            if surf is not None and getattr(surf, "data", None) == mesh:
                roots.append(obj)
            elif surf is not None and mesh_uuid and getattr(surf.data, "voxel_workspace", None):
                if surf.data.voxel_workspace.uuid == mesh_uuid:
                    roots.append(obj)
    return roots


def ensure_root_for_surface(surface_obj: Any) -> Any:
    """Ensure a canonical Voxel Root Empty exists for a surface object, creating one if needed."""
    if bpy is None or surface_obj is None:
        return None

    if surface_obj.parent is not None and is_voxel_root(surface_obj.parent):
        return surface_obj.parent

    # Check if a root already exists with matching instance UUID
    root_uuid = surface_obj.get(VOXEL_ROOT_INSTANCE_UUID_FLAG, "")
    if root_uuid:
        for scene_obj in bpy.data.objects:
            if is_voxel_root(scene_obj) and scene_obj.get(VOXEL_INSTANCE_UUID_FLAG, "") == root_uuid:
                if surface_obj.parent != scene_obj:
                    surface_obj.parent = scene_obj
                    surface_obj.matrix_local.identity()
                return scene_obj

    # Create a new Voxel Root Empty
    root_uuid = str(uuid.uuid4())
    root_name = f"Voxel Root"
    root = bpy.data.objects.new(name=root_name, object_data=None)
    root.empty_display_type = 'PLAIN_AXES'
    root[VOXEL_ROOT_FLAG] = True
    root[VOXEL_INSTANCE_UUID_FLAG] = root_uuid

    if hasattr(root, "voxel_workspace"):
        root.voxel_workspace.is_voxel_root = True
        root.voxel_workspace.voxel_instance_uuid = root_uuid
        root.voxel_workspace.surface_object = surface_obj

    # Put root into the collection of surface_obj
    target_col = None
    if surface_obj.users_collection:
        target_col = surface_obj.users_collection[0]
    elif hasattr(bpy.context, "scene") and bpy.context.scene:
        target_col = bpy.context.scene.collection
    if target_col is not None:
        target_col.objects.link(root)

    # Transfer world matrix from surface_obj to root
    if hasattr(bpy.context, "view_layer") and bpy.context.view_layer is not None:
        bpy.context.view_layer.update()
    root.matrix_world = surface_obj.matrix_world.copy()

    # Parent surface_obj to root with identity local transform
    surface_obj.parent = root
    surface_obj.matrix_local.identity()
    surface_obj[VOXEL_RENDER_ROLE_FLAG] = "SURFACE"
    surface_obj[VOXEL_ROOT_INSTANCE_UUID_FLAG] = root_uuid
    if hasattr(bpy.context, "view_layer") and bpy.context.view_layer is not None:
        bpy.context.view_layer.update()

    return root


def repair_voxel_hierarchy(root_or_surface: Any) -> RepairReport:
    """Check and repair hierarchy invariants for a voxel root or surface object."""
    report = RepairReport()
    if root_or_surface is None or bpy is None:
        return report

    if is_voxel_root(root_or_surface):
        root = root_or_surface
        root_uuid = root.get(VOXEL_INSTANCE_UUID_FLAG, "")
        if not root_uuid:
            root_uuid = str(uuid.uuid4())
            root[VOXEL_INSTANCE_UUID_FLAG] = root_uuid
            report.repaired_roots += 1

        surface_obj = resolve_surface_object(root)
        if surface_obj is not None:
            if surface_obj.parent != root:
                surface_obj.parent = root
                surface_obj.matrix_local.identity()
                report.reparented_children += 1
            surface_obj[VOXEL_RENDER_ROLE_FLAG] = "SURFACE"
            surface_obj[VOXEL_ROOT_INSTANCE_UUID_FLAG] = root_uuid
            report.repaired_surfaces += 1
    elif is_surface_render_object(root_or_surface):
        root = ensure_root_for_surface(root_or_surface)
        report.repaired_surfaces += 1
        if root is not None:
            report.repaired_roots += 1

    return report


def cleanup_stale_voxel_children() -> List[str]:
    """Find and clean up orphaned or stale generated children whose root or mesh no longer exists."""
    if bpy is None:
        return []
    removed = []
    for obj in list(bpy.data.objects):
        if is_surface_render_object(obj):
            if obj.parent is not None and not is_voxel_root(obj.parent) and not obj.parent.get(PROXY_OBJECT_FLAG, False):
                pass
        elif is_volume_render_object(obj):
            # Check if parent exists and is valid
            if obj.parent is None:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed.append(obj.name)
    return removed
