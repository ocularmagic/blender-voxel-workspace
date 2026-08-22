"""Blender custom properties for voxel volumes and objects."""
from typing import Any, Optional, Union
import uuid

try:
    import bpy
    from bpy.props import (
        BoolProperty,
        CollectionProperty,
        EnumProperty,
        FloatProperty,
        FloatVectorProperty,
        IntProperty,
        IntVectorProperty,
        PointerProperty,
        StringProperty,
    )
    from bpy.types import Mesh, Object, PropertyGroup
except ImportError:
    bpy = None
    PropertyGroup = object
    BoolProperty = CollectionProperty = EnumProperty = FloatProperty = FloatVectorProperty = IntProperty = IntVectorProperty = PointerProperty = StringProperty = None
    Mesh = Object = None

from ..constants import DEFAULT_PALETTE


def _palette_items(self, context):
    from ..ui.palette_icons import palette_enum_items
    return palette_enum_items(self, context)


def _palette_choice_changed(self, _context):
    self.active_palette_index = int(self.active_palette_choice)


def _display_changed(_self, _context):
    from .runtime import tag_redraw_all_viewports
    tag_redraw_all_viewports()


def _palette_color_updated(self, _context):
    """Callback when a palette entry's color is modified.
    
    Invalidates only the display-color GPU cache and redraws viewports. Native
    Blender Materials are authoritative for rendered shading.
    """
    from .runtime import _UNDO_GUARD, tag_redraw_all_viewports
    if _UNDO_GUARD or bpy is None:
        return
    # Find owning mesh UUID from id_data if possible.
    mesh = getattr(self, "id_data", None)
    uuid_str = None
    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        uuid_str = getattr(mesh.voxel_workspace, "uuid", None)
    try:
        from .gpu_preview import drop_palette_lut
        drop_palette_lut(uuid_str)
    except Exception:
        pass
    tag_redraw_all_viewports()


class VoxelPaletteEntry(PropertyGroup):
    """A single color entry in a voxel volume's palette."""
    if bpy is not None:
        index: IntProperty(
            name="Index",
            description="Palette index (1-255, 0 reserved for empty)",
            default=0,
            min=0,
            max=255,
        )
        name: StringProperty(
            name="Name",
            description="Display name for this palette entry",
            default="",
        )
        color: FloatVectorProperty(
            name="Color",
            description="Linear RGBA display color for this palette entry",
            size=4,
            subtype='COLOR',
            default=(1.0, 1.0, 1.0, 1.0),
            min=0.0,
            max=1.0,
            update=_palette_color_updated,
        )
        material_domain: EnumProperty(
            name="Material Domain",
            description="Render domain for voxels using this palette entry",
            items=[
                ("SURFACE", "Surface", "Rendered through material slots on primary surface mesh"),
                ("VOLUME", "Volume", "Rendered through a closed proxy object with volume shader"),
            ],
            default="SURFACE",
        )
        material: PointerProperty(
            type=bpy.types.Material,
            name="Material",
            description="Native Blender material bound to this palette entry",
        )
        material_owned: BoolProperty(
            name="Material Owned",
            description="True if this material was generated specifically for this volume and should be forked on copy",
            default=True,
        )


class VoxelMeshProperties(PropertyGroup):
    """Authoritative voxel volume metadata stored on the Mesh datablock."""
    if bpy is not None:
        schema_version: IntProperty(
            name="Schema Version",
            description="Version of the voxel persistence layout",
            default=3,
        )
        palette_schema_version: IntProperty(
            name="Palette Schema Version",
            description="Version of the palette schema",
            default=2,
        )
        palette_index_bits: IntProperty(
            name="Palette Index Bits",
            description="Bit-width of palette indices (default 8)",
            default=8,
        )
        palette_color_space: StringProperty(
            name="Palette Color Space",
            description="Interchange color space (default sRGB)",
            default="sRGB",
        )
        palette: CollectionProperty(
            type=VoxelPaletteEntry,
            name="Palette",
            description="Legacy shared color palette (deprecated, use surface_palette/volume_palette)",
        )
        surface_palette: CollectionProperty(
            type=VoxelPaletteEntry,
            name="Surface Palette",
            description="Per-volume Surface color palette (indices 1-255)",
        )
        volume_palette: CollectionProperty(
            type=VoxelPaletteEntry,
            name="Volume Palette",
            description="Per-volume Volume color palette (indices 1-255)",
        )
        uuid: StringProperty(
            name="Volume UUID",
            description="Authoritative unique identifier for the voxel volume",
            default="",
        )
        is_voxel_mesh: BoolProperty(
            name="Is Voxel Mesh",
            description="True if this mesh is backed by a voxel volume",
            default=False,
        )
        brick_size: IntProperty(
            name="Brick Size",
            description="Edge length of a cubic brick in voxels",
            default=32,
            min=1,
        )
        extent_min: IntVectorProperty(
            name="Extent Min",
            description="Minimum voxel coordinate (inclusive)",
            size=3,
            default=(0, 0, 0),
        )
        extent_max: IntVectorProperty(
            name="Extent Max",
            description="Maximum voxel coordinate (exclusive)",
            size=3,
            default=(32, 32, 32),
        )
        voxel_size: FloatProperty(
            name="Voxel Size",
            description="World-space edge length of a single voxel",
            default=1.0,
            min=0.0001,
        )


class VoxelObjectProperties(PropertyGroup):
    """Interaction and display flags stored on the Object."""
    if bpy is not None:
        is_voxel_object: BoolProperty(
            name="Is Voxel Object",
            description="True if this object is a voxel volume instance",
            default=False,
        )
        is_voxel_root: BoolProperty(
            name="Is Voxel Root",
            description="True if this object is the canonical root Empty for a voxel field",
            default=False,
        )
        voxel_instance_uuid: StringProperty(
            name="Voxel Instance UUID",
            description="Instance UUID of this root or child",
            default="",
        )
        voxel_render_role: StringProperty(
            name="Voxel Render Role",
            description="Render role: SURFACE or VOLUME",
            default="",
        )
        surface_object: PointerProperty(
            type=bpy.types.Object,
            name="Surface Object",
            description="Authoritative Surface mesh child object for this root",
        )
        is_editing: BoolProperty(
            name="Is Editing",
            description="True if this volume is currently being edited in voxel mode",
            default=False,
        )


class VoxelSceneProperties(PropertyGroup):
    """Scene-level voxel interaction properties."""
    if bpy is not None:
        create_size_x: IntProperty(
            name="Size X",
            description="Default X dimension in voxels for new volumes",
            default=16,
            min=1,
            max=512,
        )
        create_size_y: IntProperty(
            name="Size Y",
            description="Default Y dimension in voxels for new volumes",
            default=16,
            min=1,
            max=512,
        )
        create_size_z: IntProperty(
            name="Size Z",
            description="Default Z dimension in voxels for new volumes",
            default=16,
            min=1,
            max=512,
        )
        create_voxel_size: FloatProperty(
            name="Voxel Size",
            description="Default world-space edge length of a single voxel for new volumes",
            default=1.0,
            min=0.0001,
        )
        active_palette_index: IntProperty(
            name="Active Palette Index",
            description="Stored color index used by the voxel brush (legacy alias)",
            default=1,
            min=1,
            max=255,
        )
        active_surface_palette_index: IntProperty(
            name="Active Surface Palette Index",
            description="Active index for Surface editing (1-255)",
            default=1,
            min=1,
            max=255,
        )
        active_volume_palette_index: IntProperty(
            name="Active Volume Palette Index",
            description="Active index for Volume editing (1-255)",
            default=1,
            min=1,
            max=255,
        )
        active_voxel_tool: EnumProperty(
            name="Active Voxel Tool",
            items=[
                ("ADD_SURFACE", "Add Surface", "Place surface voxels"),
                ("ADD_VOLUME", "Add Volume", "Place volume voxels"),
                ("ERASE", "Erase", "Erase voxels"),
            ],
            default="ADD_SURFACE",
        )
        active_palette_tab: EnumProperty(
            name="Active Palette Tab",
            items=[
                ("SURFACE", "Surface", "Surface material palette"),
                ("VOLUME", "Volume", "Volume material palette"),
            ],
            default="SURFACE",
        )
        active_palette_choice: EnumProperty(
            name="Placement Color",
            description="Choose the color for newly placed voxels",
            items=_palette_items,
            default=1,
            update=_palette_choice_changed,
        )
        palette_filter: EnumProperty(
            name="Palette Filter",
            description="Filter visible palette swatches",
            items=[
                ("ALL", "All", "Show all palette colors"),
                ("USED", "Used", "Show only colors used in this volume"),
            ],
            default="ALL",
        )
        active_tool: EnumProperty(
            name="Active Tool",
            items=[
                ("NONE", "None", "No voxel brush is active"),
                ("PLACE", "Place", "Place brush is active"),
                ("ERASE", "Erase", "Erase brush is active"),
            ],
            default="NONE",
        )
        show_voxel_edges: BoolProperty(
            name="Show Voxel Edges",
            description="Draw exposed voxel cell boundaries while editing",
            default=True,
            update=_display_changed,
        )


def migrate_native_material_domains(mesh: Any) -> bool:
    """Migrate a schema-1 voxel mesh to schema-2 native material domain bindings.
    
    Creates owned native surface materials for all palette entries with palette display
    colors, Roughness 1.0, Emission 0, and domain SURFACE. Idempotent.
    """
    if bpy is None or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return False

    props = mesh.voxel_workspace
    if props.palette_schema_version >= 2:
        return False

    from .material_domains import create_default_surface_material
    for entry in props.palette:
        if entry.index > 0:
            if not getattr(entry, "material_domain", None):
                entry.material_domain = "SURFACE"
            if entry.material is None:
                entry.material = create_default_surface_material(mesh, entry)
                entry.material_owned = True

    props.palette_schema_version = 2
    return True


def ensure_palette(mesh: Any) -> None:
    """Ensure surface and volume palette collections contain default entries if empty.
    
    Idempotent and never pushes undo. Old .blend files or new volumes receive
    the defaults without altering voxel indices.
    """
    props = mesh.voxel_workspace
    if not hasattr(props, "surface_palette") or not hasattr(props, "volume_palette"):
        return

    # Populate Surface Palette if empty
    if len(props.surface_palette) == 0:
        from .material_domains import initialize_surface_entry

        # Index 0 placeholder
        item = props.surface_palette.add()
        item.index = 0
        item.name = "Empty"
        item.color = DEFAULT_PALETTE[0]
        item.material_owned = True

        default_names = {
            1: "Neutral Gray",
            2: "Red",
            3: "Green",
            4: "Blue",
            5: "Yellow",
            6: "Magenta",
            7: "Cyan",
            8: "Orange",
        }
        for idx in range(1, len(DEFAULT_PALETTE)):
            item = props.surface_palette.add()
            name = default_names.get(idx, f"Color {idx}")
            color = DEFAULT_PALETTE[idx]
            initialize_surface_entry(mesh, item, index=idx, name=name, color=color)

    # Populate Volume Palette if empty
    if len(props.volume_palette) == 0:
        from .material_domains import initialize_volume_entry

        # Index 0 placeholder
        item = props.volume_palette.add()
        item.index = 0
        item.name = "Empty"
        item.color = DEFAULT_PALETTE[0]
        item.material_owned = True

        # Index 1 default: Mist
        item1 = props.volume_palette.add()
        initialize_volume_entry(mesh, item1, index=1, name="Mist", color=(0.8, 0.85, 0.9, 1.0))

    # Keep legacy palette in sync for migration reads if present
    if hasattr(props, "palette") and len(props.palette) == 0:
        from .material_domains import initialize_palette_entry
        item = props.palette.add()
        item.index = 0
        item.name = "Empty"
        item.color = DEFAULT_PALETTE[0]
        item.material_domain = "SURFACE"
        item.material_owned = True
        for idx in range(1, len(DEFAULT_PALETTE)):
            item = props.palette.add()
            name = default_names.get(idx, f"Color {idx}")
            color = DEFAULT_PALETTE[idx]
            initialize_palette_entry(mesh, item, index=idx, name=name, color=color, domain="SURFACE")


def init_voxel_mesh_properties(
    mesh: "bpy.types.Mesh",
    uuid_str: Optional[str] = None,
    extent_min: tuple[int, int, int] = (0, 0, 0),
    extent_max: tuple[int, int, int] = (32, 32, 32),
    brick_size: int = 32,
    voxel_size: float = 1.0,
    schema_version: int = 3,
) -> str:
    """Initialize voxel metadata on a Blender Mesh ID and return its UUID."""
    if not uuid_str:
        uuid_str = str(uuid.uuid4())
    props = mesh.voxel_workspace
    props.uuid = uuid_str
    props.is_voxel_mesh = True
    props.extent_min = extent_min
    props.extent_max = extent_max
    props.brick_size = brick_size
    props.voxel_size = voxel_size
    props.schema_version = schema_version
    props.palette_schema_version = 2
    props.palette_index_bits = 8
    props.palette_color_space = "sRGB"
    if len(props.palette) == 0:
        ensure_palette(mesh)
    return uuid_str


def init_voxel_object_properties(
    obj: "bpy.types.Object",
    is_root: bool = False,
    instance_uuid: Optional[str] = None,
    render_role: str = "",
    surface_obj: Optional["bpy.types.Object"] = None,
) -> None:
    """Initialize interaction flags on a Blender Object."""
    props = obj.voxel_workspace
    props.is_voxel_object = True
    props.is_editing = False
    props.is_voxel_root = is_root
    if instance_uuid:
        props.voxel_instance_uuid = instance_uuid
        obj["voxel_instance_uuid"] = instance_uuid
    if render_role:
        props.voxel_render_role = render_role
        obj["voxel_render_role"] = render_role
    if surface_obj is not None:
        props.surface_object = surface_obj
    if is_root:
        obj["is_voxel_root"] = True


def get_volume_uuid(target: Union["bpy.types.Mesh", "bpy.types.Object", None]) -> Optional[str]:
    """Extract the volume UUID from a Mesh or Object.
    
    Authoritative identity lives on the Mesh datablock.
    """
    if target is None:
        return None
    if hasattr(target, "data") and target.data is not None:
        mesh = target.data
        if hasattr(mesh, "voxel_workspace"):
            return mesh.voxel_workspace.uuid or None
        return None
    if hasattr(target, "voxel_workspace"):
        return target.voxel_workspace.uuid or None
    return None


PROPERTY_CLASSES = [
    VoxelPaletteEntry,
    VoxelMeshProperties,
    VoxelObjectProperties,
    VoxelSceneProperties,
]


def register_properties() -> None:
    if bpy is None:
        return
    for cls in PROPERTY_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Mesh.voxel_workspace = PointerProperty(type=VoxelMeshProperties)
    bpy.types.Object.voxel_workspace = PointerProperty(type=VoxelObjectProperties)
    bpy.types.Scene.voxel_workspace = PointerProperty(type=VoxelSceneProperties)


def unregister_properties() -> None:
    if bpy is None:
        return
    if hasattr(bpy.types.Mesh, "voxel_workspace"):
        del bpy.types.Mesh.voxel_workspace
    if hasattr(bpy.types.Object, "voxel_workspace"):
        del bpy.types.Object.voxel_workspace
    if hasattr(bpy.types.Scene, "voxel_workspace"):
        del bpy.types.Scene.voxel_workspace
    for cls in reversed(PROPERTY_CLASSES):
        bpy.utils.unregister_class(cls)
