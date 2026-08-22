"""Create Voxel Volume / Field operator."""
from typing import Any, Optional
import uuid

try:
    import bpy
    from bpy.props import FloatProperty, IntProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    IntProperty = FloatProperty = None

from ..core.tagged_grid import TaggedVoxelGrid
from ..blender.properties import (
    init_voxel_mesh_properties,
    init_voxel_object_properties,
)
from ..blender.object_graph import (
    VOXEL_ROOT_FLAG,
    VOXEL_INSTANCE_UUID_FLAG,
    VOXEL_RENDER_ROLE_FLAG,
    VOXEL_ROOT_INSTANCE_UUID_FLAG,
    VOXEL_FIELD_COLLECTION_NAME,
)
from ..blender.persistence import init_volume_storage
from ..blender.mesh_sync import sync_volume_mesh
from ..blender.runtime import register_volume


def ensure_voxel_field_collection(context: Any) -> Any:
    """Find or create the Voxel Field collection in the current scene."""
    if bpy is None:
        return None
    # Check if a collection named Voxel Field already exists
    col = bpy.data.collections.get(VOXEL_FIELD_COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.new(VOXEL_FIELD_COLLECTION_NAME)
        scene = context.scene if context and context.scene else (bpy.data.scenes[0] if bpy.data.scenes else None)
        if scene is not None:
            scene.collection.children.link(col)
    return col


class VOXEL_OT_create_volume(Operator):
    """Create a new bounded voxel field with canonical Voxel Root and Surface child."""
    bl_idname = "voxel.create_volume"
    bl_label = "Create Voxel Volume"
    bl_description = "Create a new bounded voxel field with canonical Voxel Root and Surface child"
    bl_options = {'REGISTER'}

    if bpy is not None:
        size_x: IntProperty(
            name="Size X",
            description="X dimension in voxels",
            default=16,
            min=1,
            max=512,
        )
        size_y: IntProperty(
            name="Size Y",
            description="Y dimension in voxels",
            default=16,
            min=1,
            max=512,
        )
        size_z: IntProperty(
            name="Size Z",
            description="Z dimension in voxels",
            default=16,
            min=1,
            max=512,
        )
        voxel_size: FloatProperty(
            name="Voxel Size",
            description="World-space edge length of a single voxel",
            default=1.0,
            min=0.0001,
        )

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        # Target collection: use existing active collection or Voxel Field container
        target_collection = context.collection
        if target_collection is None:
            target_collection = ensure_voxel_field_collection(context)

        # 1. Create canonical Plain-Axis Empty named "Voxel Root"
        root_instance_uuid = str(uuid.uuid4())
        root_obj = bpy.data.objects.new(name="Voxel Root", object_data=None)
        root_obj.empty_display_type = 'PLAIN_AXES'

        # 2. Create Surface Mesh datablock and child Object named "Voxel Surface"
        mesh = bpy.data.meshes.new(name="Voxel Surface")
        surface_obj = bpy.data.objects.new(name="Voxel Surface", object_data=mesh)

        if target_collection is not None:
            target_collection.objects.link(root_obj)
            target_collection.objects.link(surface_obj)

        # Parenting: surface_obj is direct child of root_obj with identity local transform
        surface_obj.parent = root_obj
        surface_obj.matrix_local.identity()

        # 3. Attach schema/UUID/extent metadata with origin at bottom-plane center
        sx = int(self.size_x)
        sy = int(self.size_y)
        sz = int(self.size_z)
        min_x = -(sx // 2)
        max_x = min_x + sx
        min_y = -(sy // 2)
        max_y = min_y + sy
        min_z = 0
        max_z = sz

        extent_min = (min_x, min_y, min_z)
        extent_max = (max_x, max_y, max_z)
        v_size = float(self.voxel_size)

        mesh_uuid_str = init_voxel_mesh_properties(
            mesh,
            extent_min=extent_min,
            extent_max=extent_max,
            brick_size=32,
            voxel_size=v_size,
        )

        # Tag root metadata
        init_voxel_object_properties(
            root_obj,
            is_root=True,
            instance_uuid=root_instance_uuid,
            surface_obj=surface_obj,
        )
        root_obj[VOXEL_ROOT_FLAG] = True
        root_obj[VOXEL_INSTANCE_UUID_FLAG] = root_instance_uuid

        # Tag surface child metadata
        init_voxel_object_properties(
            surface_obj,
            is_root=False,
            instance_uuid=root_instance_uuid,
            render_role="SURFACE",
        )
        surface_obj[VOXEL_RENDER_ROLE_FLAG] = "SURFACE"
        surface_obj[VOXEL_ROOT_INSTANCE_UUID_FLAG] = root_instance_uuid

        # 4. Create empty grid and initialize storage
        grid = TaggedVoxelGrid(extent_min=extent_min, extent_max_exclusive=extent_max, brick_size=32)
        init_volume_storage(mesh, grid=grid, push_undo=False)

        # 5. Empty native-domain render mesh.
        sync_volume_mesh(mesh, grid=grid, dirty_only=False, ensure_material=False, voxel_size=v_size)

        # 6. Register runtime entry
        register_volume(
            mesh_uuid_str,
            grid=grid,
            voxel_size=v_size,
            brick_size=32,
            extent_min=extent_min,
            extent_max=extent_max,
        )

        # 7. Select and activate root object (generated children unselected)
        if hasattr(context, "view_layer") and context.view_layer is not None:
            for o in context.view_layer.objects.selected:
                o.select_set(False)
            root_obj.select_set(True)
            context.view_layer.objects.active = root_obj

        # 8. Push one explicit undo step for creation
        if hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Create Voxel Volume")
            except Exception:
                pass

        return {'FINISHED'}


OPERATOR_CLASSES = [
    VOXEL_OT_create_volume,
]
