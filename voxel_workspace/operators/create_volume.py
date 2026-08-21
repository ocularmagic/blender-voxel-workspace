"""Create Voxel Volume operator."""
from typing import Any, Optional

try:
    import bpy
    from bpy.props import FloatProperty, IntProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    IntProperty = FloatProperty = None

from ..core.grid import VoxelGrid
from ..blender.properties import (
    init_voxel_mesh_properties,
    init_voxel_object_properties,
)
from ..blender.persistence import init_volume_storage
from ..blender.materials import ensure_voxel_material
from ..blender.mesh_sync import sync_volume_mesh
from ..blender.runtime import register_volume


class VOXEL_OT_create_volume(Operator):
    """Create a new bounded voxel volume object."""
    bl_idname = "voxel.create_volume"
    bl_label = "Create Voxel Volume"
    bl_description = "Create a new bounded voxel volume object"
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

        # 1. Create Mesh and Object datablocks
        mesh = bpy.data.meshes.new(name="VoxelVolume")
        obj = bpy.data.objects.new(name="VoxelVolume", object_data=mesh)

        # Link to target collection
        collection = context.collection
        if collection is None:
            if hasattr(bpy.data, "scenes") and len(bpy.data.scenes) > 0:
                collection = bpy.data.scenes[0].collection
        if collection is not None:
            collection.objects.link(obj)

        # 2. Attach schema/UUID/extent metadata with origin at bottom-plane center
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

        uuid_str = init_voxel_mesh_properties(
            mesh,
            extent_min=extent_min,
            extent_max=extent_max,
            brick_size=32,
            voxel_size=v_size,
        )
        init_voxel_object_properties(obj)

        # 3. Create empty grid and initialize storage
        grid = VoxelGrid(extent_min=extent_min, extent_max_exclusive=extent_max, brick_size=32)
        init_volume_storage(mesh, grid=grid, push_undo=False)

        # 4. Palette material and empty render mesh
        ensure_voxel_material(mesh)
        sync_volume_mesh(mesh, grid=grid, dirty_only=False, ensure_material=True, voxel_size=v_size)

        # 5. Register runtime entry
        register_volume(
            uuid_str,
            grid=grid,
            voxel_size=v_size,
            brick_size=32,
            extent_min=extent_min,
            extent_max=extent_max,
        )

        # 6. Select and activate object
        if hasattr(context, "view_layer") and context.view_layer is not None:
            for o in context.view_layer.objects.selected:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj

        # 7. Push one explicit undo step for creation
        if hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
            try:
                bpy.ops.ed.undo_push(message="Create Voxel Volume")
            except Exception:
                pass

        return {'FINISHED'}


OPERATOR_CLASSES = [
    VOXEL_OT_create_volume,
]
