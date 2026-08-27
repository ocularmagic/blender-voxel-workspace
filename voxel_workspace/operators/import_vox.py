"""Import a MagicaVoxel .vox file as a new Voxel Workspace volume."""
from time import perf_counter
from typing import Any

try:
    import bpy
    from bpy.props import StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    StringProperty = None

from ..importers.vox_import import VoxParseError, parse_vox_file, srgb_bytes_to_linear
from ..blender.object_graph import resolve_volume_context


class VOXEL_OT_import_vox(Operator):
    """Import a MagicaVoxel .vox file as a new voxel volume with its own palette."""

    bl_idname = "voxel.import_vox"
    bl_label = "Import VOX"
    bl_description = "Create a new voxel volume from a MagicaVoxel .vox file (standard format, single model)"
    bl_options = {"REGISTER", "UNDO"}

    if bpy is not None:
        filepath: StringProperty(name="File Path", subtype="FILE_PATH", default="")
        filter_glob: StringProperty(default="*.vox", options={"HIDDEN"})

    def invoke(self, context: Any, event: Any) -> set:
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {"CANCELLED"}
        started = perf_counter()
        try:
            document = parse_vox_file(self.filepath)
        except VoxParseError as exc:
            self.report({"ERROR"}, f"VOX import failed: {exc}")
            return {"CANCELLED"}

        model = document.models[0]
        size_x, size_y, size_z = model.size
        if size_x < 1 or size_y < 1 or size_z < 1:
            self.report({"ERROR"}, "VOX model has zero-sized dimension")
            return {"CANCELLED"}

        # The add-on grid is X/Y-centered at origin with the bottom layer at
        # z=0. .vox coordinates are 0-based with z up, matching that layout
        # once shifted by the same centering offsets create_volume uses.
        created = bpy.ops.voxel.create_volume(
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            voxel_size=1.0,
            push_undo=False,
        )
        if "FINISHED" not in created:
            self.report({"ERROR"}, "Failed to create voxel volume for import")
            return {"CANCELLED"}

        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({"ERROR"}, "Created voxel volume could not be resolved")
            return {"CANCELLED"}
        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        emin = tuple(int(v) for v in props.extent_min)

        # Build the fresh Surface palette straight from the .vox RGBA chunk.
        # Index 0 stays the "Empty" slot; .vox color indices are 1-based and
        # map directly onto palette indices 1..255.
        from ..blender.material_domains import initialize_surface_entry
        from ..constants import DEFAULT_PALETTE

        old_material_names = {material.name for material in bpy.data.materials}
        surface_palette = props.surface_palette
        surface_palette.clear()
        empty = surface_palette.add()
        empty.index = 0
        empty.name = "Empty"
        empty.color = DEFAULT_PALETTE[0]
        empty.material_owned = True

        used_indices = set(document.used_color_indices())
        for vox_index in sorted(used_indices):
            if vox_index > 255:
                continue
            raw = document.palette[vox_index - 1]
            if raw[3] == 0:
                # Fully transparent color: treat the referencing voxels as empty.
                continue
            entry = surface_palette.add()
            initialize_surface_entry(
                mesh,
                entry,
                index=vox_index,
                name=f"Vox {vox_index}",
                color=srgb_bytes_to_linear(raw),
            )

        # Populate the grid, mapping transparent palette colors to empty.
        from ..core.tagged_grid import VoxelDomain

        grid = None
        from ..blender.runtime import get_or_load
        entry_runtime = get_or_load(mesh)
        if entry_runtime is not None:
            grid = entry_runtime.grid

        transparent_indices = {
            vox_index
            for vox_index in used_indices
            if vox_index <= 255 and document.palette[vox_index - 1][3] == 0
        }
        written = 0
        for (vx, vy, vz), color_index in model.voxels.items():
            if color_index in transparent_indices:
                continue
            if not (1 <= color_index <= 255):
                continue
            coord = (emin[0] + vx, emin[1] + vy, emin[2] + vz)
            if grid is not None:
                grid.set_cell(coord, VoxelDomain.SURFACE, color_index)
            written += 1

        # Commit: serialize, sync mesh/GPU, refresh previews, one undo step.
        from ..blender.persistence import serialize_volume
        from ..blender.mesh_sync import sync_volume_mesh
        from ..blender.gpu_preview import drop_palette_lut, update_volume_gpu_preview
        from ..blender.surface_edges import sync_surface_edge_materials

        voxel_size = float(props.voxel_size)
        if entry_runtime is not None:
            serialize_volume(mesh, entry_runtime.grid, dirty_only=False)
            drop_palette_lut(str(props.uuid))
            entry_runtime.palette_lut = None
            entry_runtime.cpu_buffers.clear()
            entry_runtime.volume_proxy_buffers.clear()
            entry_runtime.gpu_batches.clear()
            entry_runtime.gpu_edge_batches.clear()
            entry_runtime.dirty_bricks.clear()
        sync_volume_mesh(mesh, grid=grid, entry=entry_runtime, dirty_only=False, ensure_material=True, voxel_size=voxel_size)
        update_volume_gpu_preview(entry_runtime, dirty_only=False)
        sync_surface_edge_materials(mesh)
        try:
            bpy.ops.wm.preview_ensure()
        except Exception:
            pass

        # Select the created root.
        if context.view_layer is not None:
            for other in context.view_layer.objects.selected:
                other.select_set(False)
            target = v_ctx.root if v_ctx.root is not None else v_ctx.surface_object
            target.select_set(True)
            context.view_layer.objects.active = target

        try:
            bpy.ops.ed.undo_push(message="Import VOX")
        except Exception:
            pass

        # Release any orphaned materials from the replaced default palette.
        from ..blender.material_domains import cleanup_owned_materials
        try:
            stale = [
                material
                for material in bpy.data.materials
                if material.name not in old_material_names and material.users == 0
            ]
            cleanup_owned_materials(stale)
        except Exception:
            pass

        elapsed = perf_counter() - started
        palette_count = len(used_indices - transparent_indices)
        self.report(
            {"INFO"},
            f"Imported {written} voxels ({size_x}×{size_y}×{size_z}), {palette_count} colors in {elapsed:.2f}s",
        )
        return {"FINISHED"}


VOX_IMPORT_OPERATOR_CLASSES = [
    VOXEL_OT_import_vox,
]
