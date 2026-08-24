"""Render-only procedural voxel edge overlay for Surface materials."""
from typing import Any, Iterable, Optional

try:
    import bpy
except ImportError:
    bpy = None


EDGE_GROUP_NAME = "Voxel Surface Edge Overlay"
EDGE_GROUP_VERSION = 2
EDGE_NODE_NAME = "VoxelSurfaceEdgeOverlay"
EDGE_ENABLED_ATTR = "voxel_surface_edges_enabled"
EDGE_WIDTH_ATTR = "voxel_surface_edge_width"
EDGE_VOXEL_SIZE_ATTR = "voxel_surface_voxel_size"
EDGE_COLOR = (0.18, 0.18, 0.18, 1.0)


def _node(nodes: Any, node_type: str, name: str, x: float, y: float) -> Any:
    node = nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = (x, y)
    return node


def _math(nodes: Any, operation: str, name: str, x: float, y: float) -> Any:
    node = _node(nodes, "ShaderNodeMath", name, x, y)
    node.operation = operation
    return node


def _socket(node: Any, name: str, fallback: Any = None) -> Any:
    socket = node.inputs.get(name)
    if socket is not None and fallback is not None and not socket.is_linked:
        socket.default_value = fallback
    return socket


def ensure_edge_node_group() -> Any:
    """Create or repair the shared color-only procedural edge node group."""
    if bpy is None:
        return None
    group = bpy.data.node_groups.get(EDGE_GROUP_NAME)
    if (
        group is not None
        and group.get("voxel_workspace_surface_edge_group", False)
        and group.get("voxel_workspace_surface_edge_group_version", 0) == EDGE_GROUP_VERSION
    ):
        return group

    if group is None:
        group = bpy.data.node_groups.new(EDGE_GROUP_NAME, "ShaderNodeTree")
    else:
        group.nodes.clear()
        while len(group.interface.items_tree) > 0:
            item = group.interface.items_tree[0]
            group.interface.remove(item)

    group["voxel_workspace_surface_edge_group"] = True
    group["voxel_workspace_surface_edge_group_version"] = EDGE_GROUP_VERSION
    position_socket = group.interface.new_socket(name="Position", in_out="INPUT", socket_type="NodeSocketVector")
    position_socket.default_value = (0.0, 0.0, 0.0)
    normal_socket = group.interface.new_socket(name="Normal", in_out="INPUT", socket_type="NodeSocketVector")
    normal_socket.default_value = (0.0, 0.0, 1.0)
    group.interface.new_socket(name="Original Color", in_out="INPUT", socket_type="NodeSocketColor")
    enabled_socket = group.interface.new_socket(name="Enabled", in_out="INPUT", socket_type="NodeSocketFloat")
    enabled_socket.default_value = 0.0
    width_socket = group.interface.new_socket(name="Edge Width", in_out="INPUT", socket_type="NodeSocketFloat")
    width_socket.default_value = 0.04
    size_socket = group.interface.new_socket(name="Voxel Size", in_out="INPUT", socket_type="NodeSocketFloat")
    size_socket.default_value = 1.0
    color_socket = group.interface.new_socket(name="Edge Color", in_out="INPUT", socket_type="NodeSocketColor")
    color_socket.default_value = EDGE_COLOR
    group.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")

    nodes = group.nodes
    links = group.links
    inp = _node(nodes, "NodeGroupInput", "Edge Inputs", -1100, 0)
    out = _node(nodes, "NodeGroupOutput", "Edge Output", 900, 0)
    pos = _node(nodes, "ShaderNodeSeparateXYZ", "Voxel Position", -900, -160)
    normal = _node(nodes, "ShaderNodeSeparateXYZ", "Face Normal", -900, -520)
    links.new(inp.outputs["Position"], pos.inputs[0])
    links.new(inp.outputs["Normal"], normal.inputs[0])

    distances = []
    for axis, y in zip(("X", "Y", "Z"), (-160, -300, -440)):
        div = _math(nodes, "DIVIDE", f"{axis} Voxel Coordinate", -680, y)
        fract = _math(nodes, "FRACT", f"{axis} Fraction", -500, y)
        inv = _math(nodes, "SUBTRACT", f"{axis} Inverse Fraction", -320, y - 50)
        inv.inputs[0].default_value = 1.0
        minimum = _math(nodes, "MINIMUM", f"{axis} Grid Distance", -120, y)
        position_socket = pos.outputs[axis]
        links.new(position_socket, div.inputs[0])
        links.new(inp.outputs["Voxel Size"], div.inputs[1])
        links.new(div.outputs[0], fract.inputs[0])
        links.new(fract.outputs[0], inv.inputs[1])
        links.new(fract.outputs[0], minimum.inputs[0])
        links.new(inv.outputs[0], minimum.inputs[1])
        distances.append(minimum)

    axis_masks = []
    normal_components = []
    for axis, first, second, y in (
        ("X", 1, 2, -80),
        ("Y", 0, 2, -280),
        ("Z", 0, 1, -480),
    ):
        nearer = _math(nodes, "MINIMUM", f"{axis} Face Nearest Grid", -20, y)
        threshold = _math(nodes, "LESS_THAN", f"{axis} Face Edge", 180, y)
        normal_abs = _math(nodes, "ABSOLUTE", f"{axis} Normal Weight", 180, y - 150)
        weighted = _math(nodes, "MULTIPLY", f"{axis} Face Weight", 380, y)
        links.new(distances[first].outputs[0], nearer.inputs[0])
        links.new(distances[second].outputs[0], nearer.inputs[1])
        links.new(nearer.outputs[0], threshold.inputs[0])
        links.new(inp.outputs["Edge Width"], threshold.inputs[1])
        links.new(normal.outputs[axis], normal_abs.inputs[0])
        links.new(threshold.outputs[0], weighted.inputs[0])
        links.new(normal_abs.outputs[0], weighted.inputs[1])
        axis_masks.append(weighted)
        normal_components.append(normal_abs)

    mask_add_a = _math(nodes, "ADD", "Combine X Y Edge", 560, -120)
    mask_add_b = _math(nodes, "ADD", "Combine XYZ Edge", 700, -120)
    mask_max = _math(nodes, "MINIMUM", "Clamp Edge Mask", 820, -120)
    enabled_mask = _math(nodes, "MULTIPLY", "Render Enabled Mask", 560, 80)
    camera_only = _math(nodes, "MULTIPLY", "Camera Render Only", 740, 80)
    light_path = _node(nodes, "ShaderNodeLightPath", "Final Camera Ray", 360, 220)
    links.new(axis_masks[0].outputs[0], mask_add_a.inputs[0])
    links.new(axis_masks[1].outputs[0], mask_add_a.inputs[1])
    links.new(mask_add_a.outputs[0], mask_add_b.inputs[0])
    links.new(axis_masks[2].outputs[0], mask_add_b.inputs[1])
    mask_max.inputs[1].default_value = 1.0
    links.new(mask_add_b.outputs[0], mask_max.inputs[0])
    links.new(mask_max.outputs[0], enabled_mask.inputs[0])
    links.new(inp.outputs["Enabled"], enabled_mask.inputs[1])
    links.new(enabled_mask.outputs[0], camera_only.inputs[0])
    links.new(light_path.outputs["Is Camera Ray"], camera_only.inputs[1])

    mix = _node(nodes, "ShaderNodeMixRGB", "Grey Surface Edge Mix", 900, 0)
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = 0.0
    links.new(camera_only.outputs[0], mix.inputs[0])
    links.new(inp.outputs["Original Color"], mix.inputs[1])
    links.new(inp.outputs["Edge Color"], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Color"])
    return group


def _find_principled(material: Any) -> Any:
    tree = getattr(material, "node_tree", None)
    if tree is None:
        return None
    node = tree.nodes.get("Principled BSDF")
    if node is not None:
        return node
    return next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)


def ensure_surface_edge_material(
    material: Any,
    enabled_value: float = 0.0,
    width_value: float = 0.04,
    voxel_size_value: float = 1.0,
) -> bool:
    """Install a direct, render-only procedural edge overlay on a Surface material."""
    if bpy is None or material is None:
        return False
    if not getattr(material, "use_nodes", False) or material.node_tree is None:
        return False
    bsdf = _find_principled(material)
    if bsdf is None or "Base Color" not in bsdf.inputs:
        return False

    tree = material.node_tree
    base_socket = bsdf.inputs["Base Color"]
    old_group = tree.nodes.get(EDGE_NODE_NAME)
    old_mix = tree.nodes.get("VoxelSurfaceEdgeMix")

    # Normal reconciliation must be cheap. Once the direct graph exists, only
    # update its three scalar controls; rebuilding dozens of nodes on every
    # voxel sync makes large imported files unnecessarily slow to load/edit.
    required_nodes = (
        "VoxelSurfaceEdgeMix",
        "VoxelSurfaceEdgeGeometry",
        "VoxelSurfaceEdgePosition",
        "VoxelSurfaceEdgeNormal",
        "VoxelSurfaceEdgeLightPath",
        "VoxelSurfaceEdgeWidth",
        "VoxelSurfaceEdgeVoxelSize",
        "VoxelSurfaceEdgeEnabled",
        "VoxelSurfaceEdgeCameraMask",
        "VoxelSurfaceEdgeEnabledMask",
    )
    existing_mix = tree.nodes.get("VoxelSurfaceEdgeMix")
    existing_base_link = next(iter(base_socket.links), None)
    if (
        old_group is None
        and existing_mix is not None
        and existing_mix.bl_idname == "ShaderNodeMixRGB"
        and existing_base_link is not None
        and existing_base_link.from_node == existing_mix
        and all(tree.nodes.get(name) is not None for name in required_nodes)
    ):
        tree.nodes["VoxelSurfaceEdgeWidth"].outputs["Value"].default_value = max(0.001, min(0.45, float(width_value)))
        tree.nodes["VoxelSurfaceEdgeVoxelSize"].outputs["Value"].default_value = max(0.0001, float(voxel_size_value))
        tree.nodes["VoxelSurfaceEdgeEnabled"].outputs["Value"].default_value = 1.0 if enabled_value else 0.0
        material["voxel_workspace_surface_edge_overlay"] = True
        return True

    source_link = next(
        (link for link in list(base_socket.links) if link.from_node not in {old_group, old_mix}),
        None,
    )
    source_socket = source_link.from_socket if source_link is not None else None
    source_default = tuple(base_socket.default_value)
    if old_mix is not None:
        old_source = next(iter(old_mix.inputs[1].links), None)
        if old_source is not None:
            source_socket = old_source.from_socket
        else:
            source_default = tuple(old_mix.inputs[1].default_value)
    elif old_group is not None:
        old_input = old_group.inputs.get("Original Color")
        old_source = next(iter(old_input.links), None) if old_input is not None else None
        if old_source is not None:
            source_socket = old_source.from_socket
        elif old_input is not None:
            source_default = tuple(old_input.default_value)

    generated_names = {
        EDGE_NODE_NAME,
        "VoxelSurfaceEdgeMix",
        "VoxelSurfaceEdgeGeometry",
        "VoxelSurfaceEdgePosition",
        "VoxelSurfaceEdgeNormal",
        "VoxelSurfaceEdgeLightPath",
        "VoxelSurfaceEdgeEnabled",
        "VoxelSurfaceEdgeWidth",
        "VoxelSurfaceEdgeVoxelSize",
        "VoxelSurfaceEdgeCameraMask",
        "VoxelSurfaceEdgeEnabledMask",
    }
    generated_names.update({
        f"VoxelSurfaceEdge{axis}{suffix}"
        for axis in "XYZ"
        for suffix in ("Divide", "Fraction", "Inverse", "Distance", "Nearest", "Threshold", "Normal", "Weight")
    })
    generated_names.update({"VoxelSurfaceEdgeAddXY", "VoxelSurfaceEdgeAddXYZ", "VoxelSurfaceEdgeClamp"})
    for node in list(tree.nodes):
        if node.name in generated_names:
            tree.nodes.remove(node)
    for link in list(base_socket.links):
        tree.links.remove(link)

    def node(node_type: str, name: str, x: float, y: float) -> Any:
        result = tree.nodes.new(node_type)
        result.name = name
        result.label = name
        result.location = (x, y)
        return result

    def math_node(operation: str, name: str, x: float, y: float) -> Any:
        result = node("ShaderNodeMath", name, x, y)
        result.operation = operation
        return result

    geometry = node("ShaderNodeNewGeometry", "VoxelSurfaceEdgeGeometry", -1200, -260)
    position = node("ShaderNodeSeparateXYZ", "VoxelSurfaceEdgePosition", -1000, -100)
    normal = node("ShaderNodeSeparateXYZ", "VoxelSurfaceEdgeNormal", -1000, -520)
    tree.links.new(geometry.outputs["Position"], position.inputs[0])
    tree.links.new(geometry.outputs["Normal"], normal.inputs[0])

    distances = {}
    for axis, y in zip("XYZ", (-100, -260, -420)):
        divide = math_node("DIVIDE", f"VoxelSurfaceEdge{axis}Divide", -800, y)
        fraction = math_node("FRACT", f"VoxelSurfaceEdge{axis}Fraction", -620, y)
        inverse = math_node("SUBTRACT", f"VoxelSurfaceEdge{axis}Inverse", -440, y - 45)
        inverse.inputs[0].default_value = 1.0
        distance = math_node("MINIMUM", f"VoxelSurfaceEdge{axis}Distance", -240, y)
        tree.links.new(position.outputs[axis], divide.inputs[0])
        tree.links.new(fraction.outputs[0], inverse.inputs[1])
        tree.links.new(divide.outputs[0], fraction.inputs[0])
        tree.links.new(fraction.outputs[0], distance.inputs[0])
        tree.links.new(inverse.outputs[0], distance.inputs[1])
        distances[axis] = distance

    width = node("ShaderNodeValue", "VoxelSurfaceEdgeWidth", -800, 160)
    width.outputs["Value"].default_value = max(0.001, min(0.45, float(width_value)))
    size = node("ShaderNodeValue", "VoxelSurfaceEdgeVoxelSize", -800, 300)
    size.outputs["Value"].default_value = max(0.0001, float(voxel_size_value))
    enabled = node("ShaderNodeValue", "VoxelSurfaceEdgeEnabled", -800, 440)
    enabled.outputs["Value"].default_value = 1.0 if enabled_value else 0.0
    for axis in "XYZ":
        divide = tree.nodes[f"VoxelSurfaceEdge{axis}Divide"]
        tree.links.new(size.outputs["Value"], divide.inputs[1])

    weights = []
    for axis, first, second, y in (("X", "Y", "Z", -100), ("Y", "X", "Z", -300), ("Z", "X", "Y", -500)):
        nearest = math_node("MINIMUM", f"VoxelSurfaceEdge{axis}Nearest", 0, y)
        threshold = math_node("LESS_THAN", f"VoxelSurfaceEdge{axis}Threshold", 180, y)
        normal_abs = math_node("ABSOLUTE", f"VoxelSurfaceEdge{axis}Normal", 180, y - 150)
        weight = math_node("MULTIPLY", f"VoxelSurfaceEdge{axis}Weight", 380, y)
        tree.links.new(distances[first].outputs[0], nearest.inputs[0])
        tree.links.new(distances[second].outputs[0], nearest.inputs[1])
        tree.links.new(nearest.outputs[0], threshold.inputs[0])
        tree.links.new(width.outputs["Value"], threshold.inputs[1])
        tree.links.new(normal.outputs[axis], normal_abs.inputs[0])
        tree.links.new(threshold.outputs[0], weight.inputs[0])
        tree.links.new(normal_abs.outputs[0], weight.inputs[1])
        weights.append(weight)

    add_xy = math_node("ADD", "VoxelSurfaceEdgeAddXY", 560, -120)
    add_xyz = math_node("ADD", "VoxelSurfaceEdgeAddXYZ", 700, -120)
    clamp = math_node("MINIMUM", "VoxelSurfaceEdgeClamp", 840, -120)
    tree.links.new(weights[0].outputs[0], add_xy.inputs[0])
    tree.links.new(weights[1].outputs[0], add_xy.inputs[1])
    tree.links.new(add_xy.outputs[0], add_xyz.inputs[0])
    tree.links.new(weights[2].outputs[0], add_xyz.inputs[1])
    clamp.inputs[1].default_value = 1.0
    tree.links.new(add_xyz.outputs[0], clamp.inputs[0])

    light_path = node("ShaderNodeLightPath", "VoxelSurfaceEdgeLightPath", 560, 220)
    camera_mask = math_node("MULTIPLY", "VoxelSurfaceEdgeCameraMask", 760, 100)
    tree.links.new(clamp.outputs[0], camera_mask.inputs[0])
    tree.links.new(light_path.outputs["Is Camera Ray"], camera_mask.inputs[1])
    final_mask = math_node("MULTIPLY", "VoxelSurfaceEdgeEnabledMask", 920, 100)
    tree.links.new(camera_mask.outputs[0], final_mask.inputs[0])
    tree.links.new(enabled.outputs["Value"], final_mask.inputs[1])

    mix = node("ShaderNodeMixRGB", "VoxelSurfaceEdgeMix", 1120, 0)
    mix.blend_type = "MIX"
    tree.links.new(final_mask.outputs[0], mix.inputs[0])
    if source_socket is not None:
        tree.links.new(source_socket, mix.inputs[1])
    else:
        mix.inputs[1].default_value = source_default
    mix.inputs[2].default_value = EDGE_COLOR
    tree.links.new(mix.outputs[0], base_socket)
    material["voxel_workspace_surface_edge_overlay"] = True
    return True


def _surface_objects_for_mesh(mesh: Any) -> Iterable[Any]:
    if bpy is None or mesh is None:
        return []
    from .object_graph import iter_roots_for_mesh, resolve_surface_object
    result = []
    for root in iter_roots_for_mesh(mesh):
        surface = resolve_surface_object(root)
        if surface is not None:
            result.append(surface)
    if not result:
        result = [obj for obj in bpy.data.objects if getattr(obj, "data", None) == mesh and obj.get("voxel_render_role") == "SURFACE"]
    return result


def sync_surface_edge_object(surface_obj: Any) -> None:
    """Copy root settings to per-object shader attributes."""
    if surface_obj is None:
        return
    from .object_graph import resolve_voxel_root
    root = resolve_voxel_root(surface_obj)
    props = getattr(root, "voxel_workspace", None) if root is not None else None
    enabled = bool(getattr(props, "show_rendered_surface_edges", False))
    width = float(getattr(props, "rendered_surface_edge_width", 0.04))
    mesh = getattr(surface_obj, "data", None)
    voxel_size = float(getattr(getattr(mesh, "voxel_workspace", None), "voxel_size", 1.0))
    surface_obj[EDGE_ENABLED_ATTR] = 1.0 if enabled else 0.0
    surface_obj[EDGE_WIDTH_ATTR] = max(0.001, min(0.45, width))
    surface_obj[EDGE_VOXEL_SIZE_ATTR] = max(0.0001, voxel_size)


def sync_surface_edge_materials(mesh: Any) -> None:
    """Ensure Surface palette materials and object shader attributes are current."""
    if bpy is None or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return
    props = mesh.voxel_workspace
    surface_objects = list(_surface_objects_for_mesh(mesh))
    enabled = False
    width = 0.04
    if surface_objects:
        from .object_graph import resolve_voxel_root
        root = resolve_voxel_root(surface_objects[0])
        root_props = getattr(root, "voxel_workspace", None) if root is not None else None
        enabled = bool(getattr(root_props, "show_rendered_surface_edges", False))
        width = float(getattr(root_props, "rendered_surface_edge_width", width))
    voxel_size = float(getattr(props, "voxel_size", 1.0))
    for entry in getattr(props, "surface_palette", []):
        material = getattr(entry, "material", None)
        if material is not None:
            ensure_surface_edge_material(material, enabled, width, voxel_size)
    for surface_obj in surface_objects:
        sync_surface_edge_object(surface_obj)


def sync_surface_edge_settings_from_object(obj: Any) -> None:
    """Update an object's shader attributes after a render setting changes."""
    if obj is None:
        return
    surface_obj = obj
    if getattr(obj, "get", lambda *_: False)("voxel_render_role", "") != "SURFACE":
        from .object_graph import resolve_surface_object
        surface_obj = resolve_surface_object(obj)
    if surface_obj is not None:
        sync_surface_edge_object(surface_obj)
        mesh = getattr(surface_obj, "data", None)
        sync_surface_edge_materials(mesh)
    try:
        from .runtime import tag_redraw_all_viewports
        tag_redraw_all_viewports()
    except Exception:
        pass
