"""GLB/glTF material extraction for voxel color sampling."""
from typing import Any, List, Optional, Tuple
import numpy as np

from ..voxelization.color_sampling import SampledMaterial


def _srgb_to_linear_rgb(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb, 0.0, 1.0)
    low = c <= 0.04045
    return np.where(low, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def image_to_linear_rgba(image: Any) -> Tuple[Optional[np.ndarray], List[str]]:
    """Read a Blender image into (H, W, 4) linear RGBA float32."""
    warnings: List[str] = []
    if image is None:
        return None, ["Missing image datablock"]
    try:
        width, height = int(image.size[0]), int(image.size[1])
    except Exception:
        return None, [f"Image {getattr(image, 'name', '?')} has no size"]
    if width <= 0 or height <= 0:
        return None, [f"Image {image.name} has empty dimensions"]
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    arr = pixels.reshape((height, width, 4))
    cs = ""
    try:
        cs = str(image.colorspace_settings.name)
    except Exception:
        cs = ""
    out = arr.copy()
    if cs and cs not in ("Linear", "Linear Rec.709", "Non-Color", "Raw"):
        out[..., :3] = _srgb_to_linear_rgb(out[..., :3])
    return out.astype(np.float32), warnings


def extract_material(mat: Any) -> SampledMaterial:
    """Read Principled BSDF base color and an optional linked image texture.

    Does not assign Material.use_nodes. Unsupported procedural graphs fall back
    to the principled (or diffuse) base color and record a warning.
    """
    warnings: List[str] = []
    name = getattr(mat, "name", "") if mat is not None else ""
    fallback = (0.8, 0.8, 0.8, 1.0)
    if mat is None:
        return SampledMaterial(base_color=fallback, name=name, warnings=["Missing material"])

    diffuse = getattr(mat, "diffuse_color", None)
    if diffuse is not None and len(diffuse) >= 3:
        a = float(diffuse[3]) if len(diffuse) > 3 else 1.0
        fallback = (float(diffuse[0]), float(diffuse[1]), float(diffuse[2]), a)

    nt = getattr(mat, "node_tree", None)
    if nt is None:
        warnings.append(f"Material '{name}' has no node tree; using diffuse color")
        return SampledMaterial(base_color=fallback, name=name, warnings=warnings)

    principled = None
    for node in nt.nodes:
        if getattr(node, "bl_idname", "") == "ShaderNodeBsdfPrincipled":
            principled = node
            break
    if principled is None:
        warnings.append(f"Material '{name}' has no Principled BSDF; using diffuse color")
        return SampledMaterial(base_color=fallback, name=name, warnings=warnings)

    sock = principled.inputs.get("Base Color")
    base = fallback
    image = None
    if sock is not None:
        try:
            val = sock.default_value
            if len(val) >= 3:
                a = float(val[3]) if len(val) > 3 else 1.0
                base = (float(val[0]), float(val[1]), float(val[2]), a)
        except Exception:
            pass
        linked = bool(getattr(sock, "is_linked", False))
        if linked:
            try:
                from_node = sock.links[0].from_node
            except Exception:
                from_node = None
            bl_id = getattr(from_node, "bl_idname", "") if from_node is not None else ""
            if bl_id == "ShaderNodeTexImage" and getattr(from_node, "image", None) is not None:
                image, img_warn = image_to_linear_rgba(from_node.image)
                warnings.extend(img_warn)
                if image is None:
                    warnings.append(f"Material '{name}' base-color image could not be read")
            else:
                warnings.append(
                    f"Material '{name}' uses an unsupported procedural base color; using the factor color"
                )
    return SampledMaterial(base_color=base, image=image, name=name, warnings=warnings)
