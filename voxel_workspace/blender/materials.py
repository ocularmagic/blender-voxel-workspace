"""Palette lookup image and single-material shader setup for Blender 5.x."""
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

PALETTE_IMAGE_NAME = "VoxelPalette"
PALETTE_MATERIAL_NAME = "VoxelPaletteMaterial"
PALETTE_ATTRIBUTE_NAME = "palette_index"

# 256x1 palette lookup image. Indices 1..7 intentionally distinctive (spike003 MVP colors).
PALETTE_COLORS: List[Tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 1.0),      # 0: Empty / Background
    (1.0, 0.03, 0.03, 1.0),    # 1: Red
    (0.03, 1.0, 0.03, 1.0),    # 2: Green
    (0.03, 0.15, 1.0, 1.0),    # 3: Blue
    (1.0, 0.8, 0.03, 1.0),     # 4: Yellow
    (0.8, 0.03, 1.0, 1.0),     # 5: Magenta
    (0.03, 1.0, 1.0, 1.0),     # 6: Cyan
    (1.0, 0.3, 0.03, 1.0),     # 7: Orange
]


def get_or_create_palette_image() -> Any:
    """Get or create the 256x1 Non-Color nearest palette lookup image datablock."""
    if bpy is None:
        return None

    pix = np.zeros((256, 4), dtype=np.float32)
    pix[:, 3] = 1.0
    for i, c in enumerate(PALETTE_COLORS):
        pix[i] = c

    img = bpy.data.images.get(PALETTE_IMAGE_NAME)
    if img is None or img.size[0] != 256 or img.size[1] != 1:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(
            PALETTE_IMAGE_NAME,
            width=256,
            height=1,
            alpha=True,
            float_buffer=False,
        )
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(pix.reshape(-1))
    img.update()
    # Generated image pixel edits are not reliably preserved by a .blend
    # round-trip unless packed. The vertical slice must reopen and render
    # palette colors without an external sidecar image.
    img.pack()
    return img


def get_or_create_palette_material() -> Any:
    """Get or create the single VoxelPaletteMaterial shader graph (D6).
    
    Samples nearest from 256x1 palette image at (palette_index + 0.5) / 256.
    Does NOT assign deprecated Material.use_nodes.
    """
    if bpy is None:
        return None

    palette_image = get_or_create_palette_image()
    if PALETTE_MATERIAL_NAME in bpy.data.materials:
        mat = bpy.data.materials[PALETTE_MATERIAL_NAME]
        if mat.node_tree is not None and len(mat.node_tree.nodes) > 0:
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage":
                    node.image = palette_image
            return mat
    else:
        mat = bpy.data.materials.new(PALETTE_MATERIAL_NAME)

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
    tex.image = palette_image
    tex.interpolation = "Closest"
    tex.extension = "EXTEND"

    # Link math and lookup chain
    nt.links.new(attr.outputs["Fac"], add.inputs[0])
    nt.links.new(add.outputs[0], div.inputs[0])
    nt.links.new(div.outputs[0], comb.inputs["X"])
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])

    # Emission gives engine-independent visible color while retaining Principled BSDF
    if "Emission Color" in bs.inputs:
        nt.links.new(tex.outputs["Color"], bs.inputs["Emission Color"])
        bs.inputs["Emission Strength"].default_value = 1.0
    elif "Emission" in bs.inputs:
        nt.links.new(tex.outputs["Color"], bs.inputs["Emission"])
        bs.inputs["Emission Strength"].default_value = 1.0

    bs.inputs["Roughness"].default_value = 1.0
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])

    return mat


def ensure_voxel_material(target: Any) -> Any:
    """Ensure that the given Mesh or Object has exactly one material slot with VoxelPaletteMaterial."""
    if target is None or bpy is None:
        return None

    mesh = getattr(target, "data", target)
    if mesh is None or not hasattr(mesh, "materials"):
        return None

    mat = get_or_create_palette_material()
    if len(mesh.materials) == 0:
        mesh.materials.append(mat)
    else:
        if mesh.materials[0] != mat:
            mesh.materials[0] = mat

    while len(mesh.materials) > 1:
        mesh.materials.pop(index=len(mesh.materials) - 1)

    return mat
