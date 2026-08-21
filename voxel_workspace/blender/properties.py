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
    
    Performs only three cheap operations:
    1. refresh_palette_image(mesh) (updates pixels, NO pack, NO undo_push)
    2. drops the cached preview LUT and stale GPU batches
    3. tag_redraw_all_viewports()
    """
    from .runtime import _UNDO_GUARD, tag_redraw_all_viewports, get_volume
    if _UNDO_GUARD or bpy is None:
        return
    # Find owning mesh from id_data if possible
    mesh = getattr(self, "id_data", None)
    uuid_str = None
    if mesh is not None and hasattr(mesh, "voxel_workspace"):
        from .materials import refresh_palette_image
        refresh_palette_image(mesh)
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
            description="Linear RGBA color for this palette entry",
            size=4,
            subtype='COLOR',
            default=(1.0, 1.0, 1.0, 1.0),
            min=0.0,
            max=1.0,
            update=_palette_color_updated,
        )


class VoxelMeshProperties(PropertyGroup):
    """Authoritative voxel volume metadata stored on the Mesh datablock."""
    if bpy is not None:
        schema_version: IntProperty(
            name="Schema Version",
            description="Version of the voxel persistence layout",
            default=2,
        )
        palette_schema_version: IntProperty(
            name="Palette Schema Version",
            description="Version of the palette schema",
            default=1,
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
            description="Per-volume color palette",
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
        is_editing: BoolProperty(
            name="Is Editing",
            description="True if this volume is currently being edited in voxel mode",
            default=False,
        )


class VoxelSceneProperties(PropertyGroup):
    """Scene-level voxel interaction properties."""
    if bpy is not None:
        active_palette_index: IntProperty(
            name="Active Palette Index",
            description="Stored color index used by the voxel brush",
            default=1,
            min=1,
            max=255,
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


def ensure_palette(mesh: Any) -> None:
    """Ensure the mesh palette collection contains index 0 and default colors 1-8 if empty.
    
    Idempotent and never pushes undo. Old .blend files or new volumes receive
    the defaults without altering voxel indices.
    """
    if mesh is None or not hasattr(mesh, "voxel_workspace"):
        return
    props = mesh.voxel_workspace
    if not hasattr(props, "palette"):
        return
    
    # If palette already has entries, do not re-inject deleted indices
    if len(props.palette) > 0:
        return

    # Check index 0 (reserved empty)
    item = props.palette.add()
    item.index = 0
    item.name = "Empty"
    item.color = DEFAULT_PALETTE[0]

    # Populate default indices 1..8
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
        item = props.palette.add()
        item.index = idx
        item.name = default_names.get(idx, f"Color {idx}")
        item.color = DEFAULT_PALETTE[idx]


def init_voxel_mesh_properties(
    mesh: "bpy.types.Mesh",
    uuid_str: Optional[str] = None,
    extent_min: tuple[int, int, int] = (0, 0, 0),
    extent_max: tuple[int, int, int] = (32, 32, 32),
    brick_size: int = 32,
    voxel_size: float = 1.0,
    schema_version: int = 2,
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
    props.palette_schema_version = 1
    props.palette_index_bits = 8
    props.palette_color_space = "sRGB"
    if len(props.palette) == 0:
        ensure_palette(mesh)
    return uuid_str


def init_voxel_object_properties(obj: "bpy.types.Object") -> None:
    """Initialize interaction flags on a Blender Object."""
    props = obj.voxel_workspace
    props.is_voxel_object = True
    props.is_editing = False


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
