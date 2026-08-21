"""Palette lookup image and single-material shader setup for Blender 5.x."""
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

from ..constants import DEFAULT_PALETTE
from .properties import ensure_palette

PALETTE_IMAGE_NAME = "VoxelPalette"
PALETTE_MATERIAL_NAME = "VoxelPaletteMaterial"
PALETTE_ATTRIBUTE_NAME = "palette_index"
PALETTE_TEXTURE_NODE_NAME = "VoxelPaletteTextureNode"

# Re-export PALETTE_COLORS for backward compatibility
PALETTE_COLORS = list(DEFAULT_PALETTE)


def _build_palette_pixel_array(mesh: Any) -> np.ndarray:
    """Build a (256, 4) float32 RGBA array from the mesh's palette collection.

    Unallocated rows are transparent black [0.0, 0.0, 0.0, 0.0].
    """
    pix = np.zeros((256, 4), dtype=np.float32)

    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        props = mesh.voxel_workspace
        if len(props.palette) == 0:
            ensure_palette(mesh)
        for entry in props.palette:
            idx = int(entry.index)
            if 0 <= idx < 256:
                pix[idx] = tuple(entry.color)
    else:
        for i, c in enumerate(DEFAULT_PALETTE):
            if i < 256:
                pix[i] = c

    return pix


def get_or_create_palette_image(mesh: Any = None, pack_image: bool = True) -> Any:
    """Get or create the 256x1 Non-Color nearest palette lookup image for a mesh.

    Image is named VoxelPalette_<uuid> if mesh has a UUID, otherwise VoxelPalette.
    Called only at volume creation / material setup; does NOT push undo.
    """
    if bpy is None:
        return None

    target_mesh = getattr(mesh, "data", mesh)
    uuid_str = ""
    if target_mesh is not None and hasattr(target_mesh, "voxel_workspace"):
        uuid_str = target_mesh.voxel_workspace.uuid

    image_name = f"VoxelPalette_{uuid_str}" if uuid_str else PALETTE_IMAGE_NAME

    img = bpy.data.images.get(image_name)
    if img is None or img.size[0] != 256 or img.size[1] != 1:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(
            image_name,
            width=256,
            height=1,
            alpha=True,
            float_buffer=False,
        )

    img.colorspace_settings.name = "Non-Color"
    pix = _build_palette_pixel_array(target_mesh)
    img.pixels.foreach_set(pix.reshape(-1))
    img.update()
    if pack_image:
        img.pack()
    return img


def refresh_palette_image(mesh: Any) -> None:
    """Rewrite pixels and update the palette image for a mesh.

    Does NOT call pack() and does NOT push undo.
    """
    if bpy is None or mesh is None:
        return

    target_mesh = getattr(mesh, "data", mesh)

    # The generated palette node is the only valid binding. Images and names are
    # derived caches, so never guess from unrelated nodes or datablock names.
    img = None
    if target_mesh is not None and hasattr(target_mesh, "materials") and len(target_mesh.materials) > 0:
        mat = target_mesh.materials[0]
        if mat is not None and mat.node_tree is not None:
            if PALETTE_TEXTURE_NODE_NAME in mat.node_tree.nodes:
                node = mat.node_tree.nodes[PALETTE_TEXTURE_NODE_NAME]
                if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                    img = node.image
    if img is None:
        return

    pix = _build_palette_pixel_array(target_mesh)
    img.pixels.foreach_set(pix.reshape(-1))
    img.update()


def _disable_palette_emission(mat: Any) -> None:
    """Unlink palette emission and force Emission Strength to 0."""
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return
    bs = None
    for node in nt.nodes:
        if node.bl_idname == "ShaderNodeBsdfPrincipled":
            bs = node
            break
    if bs is None:
        return
    if "Emission Strength" in bs.inputs:
        bs.inputs["Emission Strength"].default_value = 0.0
    for sock_name in ("Emission Color", "Emission"):
        if sock_name not in bs.inputs:
            continue
        sock = bs.inputs[sock_name]
        for link in list(getattr(sock, "links", [])):
            nt.links.remove(link)


def get_or_create_palette_material(mesh: Any = None, pack_image: bool = True) -> Any:
    """Get or create the VoxelPaletteMaterial_<uuid> shader graph for a mesh (D6).
    
    Samples nearest from 256x1 palette image at (palette_index + 0.5) / 256.
    Binds the texture node to the volume's per-volume palette image.
    """
    if bpy is None:
        return None

    target_mesh = getattr(mesh, "data", mesh)
    uuid_str = ""
    if target_mesh is not None and hasattr(target_mesh, "voxel_workspace"):
        uuid_str = target_mesh.voxel_workspace.uuid

    mat_name = f"VoxelPaletteMaterial_{uuid_str}" if uuid_str else PALETTE_MATERIAL_NAME
    palette_image = get_or_create_palette_image(target_mesh, pack_image=pack_image)

    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        if mat.node_tree is not None and len(mat.node_tree.nodes) > 0:
            if PALETTE_TEXTURE_NODE_NAME in mat.node_tree.nodes:
                node = mat.node_tree.nodes[PALETTE_TEXTURE_NODE_NAME]
                if node.bl_idname == "ShaderNodeTexImage":
                    node.image = palette_image
                    _disable_palette_emission(mat)
                    return mat
            # A malformed/legacy UUID-owned material is rebuilt below. Do not
            # repurpose arbitrary image nodes as the palette texture node.
    else:
        mat = bpy.data.materials.new(mat_name)

    nt = mat.node_tree
    if nt is None:
        return mat

    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = PALETTE_ATTRIBUTE_NAME

    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = 0.5

    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 256.0

    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    comb.inputs["Y"].default_value = 0.5

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.name = PALETTE_TEXTURE_NODE_NAME
    tex.image = palette_image
    tex.interpolation = "Closest"
    tex.extension = "EXTEND"

    # Link math and lookup chain
    nt.links.new(attr.outputs["Fac"], add.inputs[0])
    nt.links.new(add.outputs[0], div.inputs[0])
    nt.links.new(div.outputs[0], comb.inputs["X"])
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])

    if "Emission Strength" in bs.inputs:
        bs.inputs["Emission Strength"].default_value = 0.0

    bs.inputs["Roughness"].default_value = 1.0
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])

    return mat


def ensure_voxel_material(target: Any, pack_image: bool = True) -> Any:
    """Ensure that the given Mesh or Object has its per-volume palette material or native surface slots."""
    if target is None or bpy is None:
        return None

    mesh = getattr(target, "data", target)
    if mesh is None or not hasattr(mesh, "materials"):
        return None

    # If native surface slots already populated on mesh, preserve them
    if len(mesh.materials) > 0:
        return mesh.materials[0]

    mat = get_or_create_palette_material(mesh, pack_image=pack_image)
    if len(mesh.materials) == 0:
        mesh.materials.append(mat)
    else:
        if mesh.materials[0] != mat:
            mesh.materials[0] = mat

    return mat
