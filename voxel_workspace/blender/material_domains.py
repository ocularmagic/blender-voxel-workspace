"""Native Blender Material lifecycle and domain management for voxel volumes."""
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math
import uuid

try:
    import bpy
    from bpy.types import Material, Mesh
except ImportError:
    bpy = None
    Material = Mesh = object

from ..constants import DEFAULT_PALETTE
from ..core.tagged_grid import VoxelDomain, TaggedVoxelGrid


# Generated material kinds
SURFACE_DEFAULT = "SURFACE_DEFAULT"
VOLUME_DEFAULT = "VOLUME_DEFAULT"


def get_palette(mesh: Any, domain: Union[VoxelDomain, str, int] = VoxelDomain.SURFACE) -> Any:
    """Get the typed palette CollectionProperty for the given mesh and domain."""
    if mesh is None or not hasattr(mesh, "voxel_workspace"):
        return []
    props = mesh.voxel_workspace
    dom_str = domain if isinstance(domain, str) else (
        "VOLUME" if int(domain) == int(VoxelDomain.VOLUME) else "SURFACE"
    )
    if dom_str.upper() == "VOLUME":
        return props.volume_palette if hasattr(props, "volume_palette") else []
    return props.surface_palette if hasattr(props, "surface_palette") else []


def find_entry(mesh: Any, domain: Union[VoxelDomain, str, int], index: int) -> Optional[Any]:
    """Find a palette entry by domain and typed palette index."""
    palette = get_palette(mesh, domain)
    for entry in palette:
        if int(entry.index) == int(index):
            return entry
    return None


def display_rgba_from_entry(entry: Any, palette_type: str = "SURFACE") -> Tuple[float, float, float, float]:
    """Return GPU/brush RGBA from the bound material, else the stored display color."""
    fallback = (0.8, 0.8, 0.8, 1.0)
    if entry is not None:
        try:
            fallback = tuple(float(component) for component in entry.color)
            if len(fallback) < 4:
                fallback = (fallback + (1.0, 1.0, 1.0, 1.0))[:4]
        except Exception:
            pass
    material = getattr(entry, "material", None) if entry is not None else None
    tree = getattr(material, "node_tree", None) if material is not None else None
    if material is None or not getattr(material, "use_nodes", False) or tree is None:
        return fallback
    try:
        if str(palette_type).upper() == "VOLUME":
            for node in tree.nodes:
                if getattr(node, "bl_idname", "") != "ShaderNodeVolumePrincipled":
                    continue
                for key in ("Color", "Scattering Color", "Absorption Color"):
                    if key in node.inputs:
                        value = node.inputs[key].default_value
                        return (float(value[0]), float(value[1]), float(value[2]), 1.0)
                break
        else:
            bsdf = tree.nodes.get("Principled BSDF")
            if bsdf is None:
                for node in tree.nodes:
                    if getattr(node, "bl_idname", "") == "ShaderNodeBsdfPrincipled":
                        bsdf = node
                        break
            if bsdf is not None and "Base Color" in bsdf.inputs:
                value = bsdf.inputs["Base Color"].default_value
                alpha = float(bsdf.inputs["Alpha"].default_value) if "Alpha" in bsdf.inputs else 1.0
                return (float(value[0]), float(value[1]), float(value[2]), alpha)
    except Exception:
        pass
    return fallback


def linear_to_srgb_rgba(rgba):
    """Convert a linear RGBA sequence to sRGB display-encoded RGBA.

    Applies the standard sRGB transfer curve to the RGB channels and leaves
    alpha untouched. GPU overlay colors (hover, edit preview LUT) are written
    to the display outside color management, so material socket values (which
    are linear) must be encoded to match how the rendered mesh / swatches look.
    """
    result = []
    for i, component in enumerate(rgba):
        c = max(0.0, min(1.0, float(component)))
        if i == 3:
            result.append(float(rgba[i]))
        elif c <= 0.0031308:
            result.append(c * 12.92)
        else:
            result.append(1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return tuple(result)


def sync_entry_color_from_material(entry: Any, palette_type: str = "SURFACE") -> None:
    """Keep stored display color aligned with the material without extra UI."""
    if entry is None:
        return
    rgba = display_rgba_from_entry(entry, palette_type)
    try:
        current = tuple(float(component) for component in entry.color)
    except Exception:
        current = ()
    if len(current) != 4 or any(abs(left - right) > 1e-4 for left, right in zip(current, rgba)):
        entry.color = rgba


def create_default_surface_material(
    mesh: Any,
    entry: Any,
    base_color: Optional[Tuple[float, float, float, float]] = None,
) -> Any:
    """Create a new native Blender surface Material with Principled BSDF."""
    if bpy is None:
        return None

    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "unknown") if mesh and hasattr(mesh, "voxel_workspace") else "unknown"
    idx = getattr(entry, "index", 1) if entry else 1
    name = getattr(entry, "name", "") or f"Color {idx}"
    mat_name = f"VoxelSurface_{name}_{idx}"

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Standard Principled setup
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    col = base_color or (tuple(entry.color) if entry else (0.5, 0.5, 0.5, 1.0))
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = float(col[3]) if len(col) > 3 else 1.0
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 1.0  # HEAD compatibility

    # Tag custom metadata
    mat["voxel_workspace_owned"] = True
    mat["voxel_workspace_owner_uuid"] = mesh_uuid
    mat["voxel_workspace_material_uid"] = str(uuid.uuid4())
    mat["voxel_workspace_generated_kind"] = SURFACE_DEFAULT

    return mat


def create_default_volume_material(
    mesh: Any,
    entry: Any,
    volume_color: Optional[Tuple[float, float, float, float]] = None,
    density: float = 5.0,
) -> Any:
    """Create a new native Blender volume Material with Principled Volume."""
    if bpy is None:
        return None

    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "unknown") if mesh and hasattr(mesh, "voxel_workspace") else "unknown"
    idx = getattr(entry, "index", 1) if entry else 1
    name = getattr(entry, "name", "") or f"Mist {idx}"
    mat_name = f"VoxelVolume_{name}_{idx}"

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    p_vol = nodes.new("ShaderNodeVolumePrincipled")
    links.new(p_vol.outputs["Volume"], out.inputs["Volume"])

    col = volume_color or (tuple(entry.color) if entry else (0.8, 0.85, 0.9, 1.0))
    if "Color" in p_vol.inputs:
        p_vol.inputs["Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
    if "Density" in p_vol.inputs:
        p_vol.inputs["Density"].default_value = float(density)

    # Tag custom metadata
    mat["voxel_workspace_owned"] = True
    mat["voxel_workspace_owner_uuid"] = mesh_uuid
    mat["voxel_workspace_material_uid"] = str(uuid.uuid4())
    mat["voxel_workspace_generated_kind"] = VOLUME_DEFAULT

    return mat


def ensure_entry_material(
    mesh: Any,
    entry: Any,
    domain: Optional[Union[VoxelDomain, str, int]] = None,
) -> Any:
    """Ensure a palette entry has a valid native Material bound according to its domain."""
    if bpy is None or entry is None:
        return None

    if domain is not None:
        dom_str = domain if isinstance(domain, str) else (
            "VOLUME" if int(domain) == int(VoxelDomain.VOLUME) else "SURFACE"
        )
    else:
        dom_str = getattr(entry, "material_domain", "SURFACE")

    mat = getattr(entry, "material", None)

    if mat is None:
        if dom_str.upper() == "VOLUME":
            mat = create_default_volume_material(mesh, entry)
        else:
            mat = create_default_surface_material(mesh, entry)
        entry.material = mat
        entry.material_owned = True

    return mat


def copy_entry_material_for_mesh(src_entry: Any, dst_entry: Any, new_mesh_uuid: str) -> None:
    """Copy an entry's material when duplicating or forking a mesh."""
    if bpy is None or src_entry is None or dst_entry is None:
        return

    dst_entry.material_owned = getattr(src_entry, "material_owned", True)
    src_mat = getattr(src_entry, "material", None)

    if src_mat is None:
        dst_entry.material = None
        return

    if getattr(src_entry, "material_owned", True):
        # Fork owned material
        new_mat = src_mat.copy()
        new_mat["voxel_workspace_owned"] = True
        new_mat["voxel_workspace_owner_uuid"] = new_mesh_uuid
        new_mat["voxel_workspace_material_uid"] = str(uuid.uuid4())
        dst_entry.material = new_mat
    else:
        # External shared material: share pointer directly
        dst_entry.material = src_mat


def assign_external_material(entry: Any, material: Any) -> None:
    """Assign an external/shared Blender material to a palette entry."""
    if entry is None:
        return
    entry.material = material
    entry.material_owned = False


def make_entry_material_single_user(
    mesh: Any,
    entry: Any,
    domain: Optional[Union[VoxelDomain, str, int]] = None,
) -> Any:
    """Make an entry's material single-user (owned by this mesh)."""
    if bpy is None or entry is None:
        return None
    curr_mat = getattr(entry, "material", None)
    mesh_uuid = getattr(mesh.voxel_workspace, "uuid", "unknown") if mesh and hasattr(mesh, "voxel_workspace") else "unknown"

    if curr_mat is None:
        return ensure_entry_material(mesh, entry, domain=domain)

    new_mat = curr_mat.copy()
    new_mat["voxel_workspace_owned"] = True
    new_mat["voxel_workspace_owner_uuid"] = mesh_uuid
    new_mat["voxel_workspace_material_uid"] = str(uuid.uuid4())
    entry.material = new_mat
    entry.material_owned = True
    return new_mat


def is_owned_material(material: Any, mesh_uuid: str) -> bool:
    """Check if a material is owned by the specified mesh UUID."""
    if material is None:
        return False
    return (
        material.get("voxel_workspace_owned", False)
        and material.get("voxel_workspace_owner_uuid", "") == mesh_uuid
    )


def remove_owned_material_if_unreferenced(material: Any) -> bool:
    """Remove an owned material if it has no users/references."""
    if bpy is None or material is None:
        return False
    if not material.get("voxel_workspace_owned", False):
        return False

    if material.users == 0:
        bpy.data.materials.remove(material)
        return True
    return False


def cleanup_owned_materials(materials: List[Any]) -> int:
    """Remove unique generated Materials after every Blender user releases them."""
    removed = 0
    seen = set()
    for material in materials:
        if material is None:
            continue
        key = material.as_pointer() if hasattr(material, "as_pointer") else id(material)
        if key in seen:
            continue
        seen.add(key)
        if remove_owned_material_if_unreferenced(material):
            removed += 1
    return removed


def palette_materials(
    mesh: Any,
    domain: Optional[Union[VoxelDomain, str, int]] = None,
) -> List[Any]:
    """Snapshot all non-null Material pointers bound by palette(s)."""
    if mesh is None or not hasattr(mesh, "voxel_workspace"):
        return []
    props = mesh.voxel_workspace
    mats = []
    if domain is not None:
        pal = get_palette(mesh, domain)
        return [e.material for e in pal if getattr(e, "material", None) is not None]
    
    # All palettes
    if hasattr(props, "surface_palette"):
        mats.extend([e.material for e in props.surface_palette if getattr(e, "material", None) is not None])
    if hasattr(props, "volume_palette"):
        mats.extend([e.material for e in props.volume_palette if getattr(e, "material", None) is not None])
    return mats


def cleanup_legacy_atlas_datablocks(mesh: Any) -> None:
    """Delete recognized, unreferenced atlas caches after native slot reconciliation."""
    if bpy is None or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return
    mesh_uuid = mesh.voxel_workspace.uuid
    mat = bpy.data.materials.get(f"VoxelPaletteMaterial_{mesh_uuid}")
    if mat is not None and mat.users == 0:
        bpy.data.materials.remove(mat)
    image = bpy.data.images.get(f"VoxelPalette_{mesh_uuid}")
    if image is not None and image.users == 0:
        bpy.data.images.remove(image)


def copy_palette_entry_binding(src_entry: Any, dst_entry: Any, mesh_uuid: str) -> None:
    """Copy complete entry metadata using owned-copy/external-share semantics."""
    dst_entry.index = int(src_entry.index)
    dst_entry.name = str(src_entry.name)
    dst_entry.color = tuple(float(component) for component in src_entry.color)
    copy_entry_material_for_mesh(src_entry, dst_entry, mesh_uuid)


def set_generated_surface_base_color(entry: Any, color: Optional[Tuple[float, float, float, float]] = None) -> bool:
    """Update base color on a generated Principled BSDF node if present."""
    if entry is None or entry.material is None:
        return False
    mat = entry.material
    if not mat.use_nodes or mat.node_tree is None:
        return False

    col = color or tuple(entry.color)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf and "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
        if "Alpha" in bsdf.inputs and len(col) > 3:
            bsdf.inputs["Alpha"].default_value = float(col[3])
        return True
    return False


def set_generated_volume_color(entry: Any, color: Optional[Tuple[float, float, float, float]] = None) -> bool:
    """Update Color on a recognized Principled Volume node."""
    if entry is None or entry.material is None or not entry.material.use_nodes:
        return False
    col = color or tuple(entry.color)
    for node in entry.material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeVolumePrincipled" and "Color" in node.inputs:
            node.inputs["Color"].default_value = (float(col[0]), float(col[1]), float(col[2]), 1.0)
            return True
    return False


def allocated_entries(
    mesh: Any,
    domain: Union[VoxelDomain, str, int] = VoxelDomain.SURFACE,
) -> List[Any]:
    """Return all allocated non-zero palette entries on a mesh for the specified domain."""
    pal = get_palette(mesh, domain)
    return sorted([e for e in pal if e.index > 0], key=lambda e: e.index)


def used_palette_indices(grid: Any) -> Set[int]:
    """Return the set of all non-zero palette indices used across all bricks in a grid."""
    if grid is None:
        return set()
    used = set()
    import numpy as np
    if hasattr(grid, "iter_used_indices"):
        used.update(grid.iter_used_indices(VoxelDomain.SURFACE))
        used.update(grid.iter_used_indices(VoxelDomain.VOLUME))
        return used

    for brick in grid.bricks.values():
        if np.any(brick):
            unq = np.unique(brick)
            used.update(int(x) for x in unq if x > 0)
    return used


def used_surface_indices(mesh: Any, grid: Any) -> List[int]:
    """Return sorted list of used palette indices for the SURFACE domain."""
    if grid is None:
        return []
    if hasattr(grid, "iter_used_indices"):
        return sorted(list(grid.iter_used_indices(VoxelDomain.SURFACE)))

    used = used_palette_indices(grid)
    if not used or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return sorted(list(used))
    
    props = mesh.voxel_workspace
    if hasattr(props, "palette") and len(props.palette) > 0:
        entry_domain_map = {e.index: getattr(e, "material_domain", "SURFACE") for e in props.palette}
        surf_used = [idx for idx in used if entry_domain_map.get(idx, "SURFACE") == "SURFACE"]
        return sorted(surf_used)

    if hasattr(props, "surface_palette") and len(props.surface_palette) > 0:
        surf_indices = {e.index for e in props.surface_palette if e.index > 0}
        return sorted(list(used & surf_indices))
    
    return sorted(list(used))


def used_volume_indices(mesh: Any, grid: Any) -> List[int]:
    """Return sorted list of used palette indices for the VOLUME domain."""
    if grid is None:
        return []
    if hasattr(grid, "iter_used_indices"):
        return sorted(list(grid.iter_used_indices(VoxelDomain.VOLUME)))

    used = used_palette_indices(grid)
    if not used or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return []
    
    props = mesh.voxel_workspace
    if hasattr(props, "volume_palette") and len(props.volume_palette) > 0:
        vol_indices = {e.index for e in props.volume_palette if e.index > 0}
        # Only treat index as volume if it is in volume_palette and NOT in surface_palette, or if explicitly in legacy palette marked as VOLUME
        if hasattr(props, "surface_palette") and len(props.surface_palette) > 0:
            surf_indices = {e.index for e in props.surface_palette if e.index > 0}
            vol_only = vol_indices - surf_indices
            if (used & vol_only):
                return sorted(list(used & vol_only))
        elif (used & vol_indices):
            return sorted(list(used & vol_indices))

    if hasattr(props, "palette") and len(props.palette) > 0:
        entry_domain_map = {e.index: getattr(e, "material_domain", "SURFACE") for e in props.palette}
        volume_used = [idx for idx in used if entry_domain_map.get(idx, "SURFACE") == "VOLUME"]
        if volume_used:
            return sorted(volume_used)

    return []


def reconcile_surface_slots(mesh: Any, grid: Any) -> Dict[int, int]:
    """Reconcile the primary Mesh material slots for all used SURFACE palette indices.
    
    Returns a dictionary mapping `palette_index -> material_slot_index`.
    """
    if bpy is None or mesh is None:
        return {}

    props = mesh.voxel_workspace if hasattr(mesh, "voxel_workspace") else None
    surface_indices = used_surface_indices(mesh, grid)
    slot_map: Dict[int, int] = {}

    if not surface_indices:
        # No surface voxels used; clear materials
        mesh.materials.clear()
        return slot_map

    # Build entry lookup from surface_palette
    pal = get_palette(mesh, VoxelDomain.SURFACE)
    palette_lookup = {e.index: e for e in pal}

    # Clear and populate slots in deterministic sorted order
    mesh.materials.clear()
    for slot_idx, pal_idx in enumerate(surface_indices):
        entry = palette_lookup.get(pal_idx)
        mat = ensure_entry_material(mesh, entry, domain=VoxelDomain.SURFACE) if entry else None
        if mat is None:
            # Stable shared fallback for corrupt/legacy grids whose index has no
            # palette entry. Never allocate an anonymous Material per sync.
            mat = bpy.data.materials.get("VoxelSurface_Fallback")
            if mat is None:
                mat = bpy.data.materials.new(name="VoxelSurface_Fallback")
                mat.use_nodes = True
        mesh.materials.append(mat)
        slot_map[pal_idx] = slot_idx

    return slot_map


def initialize_surface_entry(
    mesh: Any,
    entry: Any,
    index: int,
    name: str,
    color: Tuple[float, float, float, float],
) -> None:
    """Initialize a Surface palette entry with an owned native surface material."""
    entry.index = index
    entry.name = name
    entry.color = color
    entry.material_owned = True
    entry.material = create_default_surface_material(mesh, entry, base_color=color)


def initialize_volume_entry(
    mesh: Any,
    entry: Any,
    index: int,
    name: str,
    color: Tuple[float, float, float, float],
    density: float = 5.0,
) -> None:
    """Initialize a Volume palette entry with an owned native volume material."""
    entry.index = index
    entry.name = name
    entry.color = color
    entry.material_owned = True
    entry.material = create_default_volume_material(mesh, entry, volume_color=color, density=density)


def initialize_palette_entry(
    mesh: Any,
    entry: Any,
    index: int,
    name: str,
    color: Tuple[float, float, float, float],
    domain: str = "SURFACE",
) -> None:
    """Legacy helper: Initialize a palette entry with given properties and an owned native material."""
    if str(domain).upper() == "VOLUME":
        initialize_volume_entry(mesh, entry, index, name, color)
    else:
        initialize_surface_entry(mesh, entry, index, name, color)
