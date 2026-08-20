"""Blender custom properties for voxel volumes and objects."""
from typing import Optional, Union
import uuid

try:
    import bpy
    from bpy.props import (
        BoolProperty,
        FloatProperty,
        IntProperty,
        IntVectorProperty,
        PointerProperty,
        StringProperty,
    )
    from bpy.types import Mesh, Object, PropertyGroup
except ImportError:
    bpy = None
    PropertyGroup = object
    BoolProperty = FloatProperty = IntProperty = IntVectorProperty = PointerProperty = StringProperty = None
    Mesh = Object = None


class VoxelMeshProperties(PropertyGroup):
    """Authoritative voxel volume metadata stored on the Mesh datablock."""
    if bpy is not None:
        schema_version: IntProperty(
            name="Schema Version",
            description="Version of the voxel persistence layout",
            default=1,
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
            name="Palette Index",
            description="Active palette color index for voxel placement",
            default=1,
            min=1,
            max=255,
        )


def init_voxel_mesh_properties(
    mesh: "bpy.types.Mesh",
    uuid_str: Optional[str] = None,
    extent_min: tuple[int, int, int] = (0, 0, 0),
    extent_max: tuple[int, int, int] = (32, 32, 32),
    brick_size: int = 32,
    voxel_size: float = 1.0,
    schema_version: int = 1,
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
