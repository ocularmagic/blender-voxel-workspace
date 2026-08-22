"""Import a GLB/glTF file into the active Voxel Workspace volume."""
from pathlib import Path
from time import perf_counter
from typing import Any, List, Tuple

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    BoolProperty = EnumProperty = FloatProperty = IntProperty = StringProperty = None

from ..blender.runtime import get_or_load, get_volume
from ..blender.object_graph import resolve_volume_context, resolve_authoritative_mesh, resolve_surface_object
from ..constants import DEFAULT_PALETTE
from ..importers.glb_scene import LARGE_CELL_COUNT, stage_glb
from ..operators.palette import get_used_palette_counts
from ..voxelization.voxelize import voxelize_fitted_mesh


def _is_voxel_object(obj: Any) -> bool:
    mesh = resolve_authoritative_mesh(obj)
    return bool(
        mesh is not None
        and hasattr(mesh, "voxel_workspace")
        and mesh.voxel_workspace.is_voxel_mesh
    )


def replace_volume_palette(mesh: Any, palette: List[Tuple[float, float, float, float]]) -> List[Any]:
    """Replace the mesh palette collection with owned default surface materials. Index 0 remains empty."""
    from ..blender.material_domains import initialize_palette_entry, palette_materials
    props = mesh.voxel_workspace
    old_materials = palette_materials(mesh)
    props.palette.clear()
    empty = props.palette.add()
    empty.index = 0
    empty.name = "Empty"
    empty.color = DEFAULT_PALETTE[0]
    empty.material_domain = "SURFACE"
    empty.material_owned = True
    for idx, color in enumerate(palette):
        if idx == 0:
            continue
        if idx > 255:
            break
        item = props.palette.add()
        initialize_palette_entry(
            mesh,
            item,
            index=idx,
            name=f"Imported {idx}",
            color=tuple(float(c) for c in color),
            domain="SURFACE",
        )
    return old_materials


def _commit_import(obj: Any, result: Any, undo_message: str) -> None:
    mesh = obj.data
    uuid_str = mesh.voxel_workspace.uuid
    entry = get_or_load(mesh)
    if entry is None:
        from ..blender.runtime import register_volume

        entry = register_volume(
            uuid_str,
            grid=result.grid,
            voxel_size=float(mesh.voxel_workspace.voxel_size),
            brick_size=int(mesh.voxel_workspace.brick_size),
            extent_min=tuple(mesh.voxel_workspace.extent_min),
            extent_max=tuple(mesh.voxel_workspace.extent_max),
        )
    else:
        entry.grid = result.grid
        entry.cpu_buffers.clear()
        entry.volume_proxy_buffers.clear()
        entry.gpu_batches.clear()
        entry.gpu_edge_batches.clear()
        entry.dirty_bricks.clear()
        entry.palette_lut = None

    old_materials = replace_volume_palette(mesh, result.palette)

    from ..blender.persistence import serialize_volume
    from ..blender.mesh_sync import sync_volume_mesh
    from ..blender.gpu_preview import drop_palette_lut, update_volume_gpu_preview

    serialize_volume(mesh, entry.grid, dirty_only=False)
    drop_palette_lut(uuid_str)
    v_size = float(mesh.voxel_workspace.voxel_size)
    sync_volume_mesh(mesh, grid=entry.grid, entry=entry, dirty_only=False, ensure_material=True, voxel_size=v_size)
    update_volume_gpu_preview(entry, dirty_only=False)
    from ..blender.material_domains import cleanup_owned_materials
    cleanup_owned_materials(old_materials)

    if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass


class VOXEL_OT_import_glb(Operator):
    """Voxelize a GLB/glTF file into the selected empty Voxel Workspace volume."""

    bl_idname = "voxel.import_glb"
    bl_label = "Import GLB into Volume"
    bl_description = "Fit and voxelize a GLB/glTF file into the selected voxel volume"
    bl_options = {"REGISTER"}

    filename_ext = ".glb"

    if bpy is not None:
        filepath: StringProperty(
            name="File Path",
            description="GLB or glTF file to import",
            subtype="FILE_PATH",
            default="",
        )
        filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})
        padding: IntProperty(
            name="Padding",
            description="Empty voxel border kept around the fitted model",
            default=1,
            min=0,
            max=32,
        )
        occupancy: EnumProperty(
            name="Occupancy",
            description="Solid fill (closed meshes) or surface shell",
            items=[
                ("SOLID", "Solid", "Fill the interior of closed meshes"),
                ("SHELL", "Surface Shell", "Occupy only voxels near the source surface"),
            ],
            default="SOLID",
        )
        palette_size: IntProperty(
            name="Palette Size",
            description="Maximum generated palette colors (index 0 stays empty)",
            default=64,
            min=1,
            max=255,
        )
        alpha_cutoff: FloatProperty(
            name="Alpha Cutoff",
            description="Surface samples below this alpha are treated as empty",
            default=0.1,
            min=0.0,
            max=1.0,
        )
        keep_source: BoolProperty(
            name="Keep Source Objects",
            description="Keep imported GLB objects hidden in a staging collection",
            default=False,
        )
        clear_and_replace: BoolProperty(
            name="Clear and Replace Volume",
            description="Allow import to replace voxels already in this volume",
            default=False,
        )

    def invoke(self, context: Any, event: Any) -> set:
        if not _is_voxel_object(context):
            self.report({"WARNING"}, "Select a voxel volume first")
            return {"CANCELLED"}
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context: Any) -> None:
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        if v_ctx is not None and v_ctx.mesh is not None:
            props = v_ctx.mesh.voxel_workspace
            emin = tuple(props.extent_min)
            emax = tuple(props.extent_max)
            dim = (emax[0] - emin[0], emax[1] - emin[1], emax[2] - emin[2])
            layout.label(text=f"Target: {dim[0]} × {dim[1]} × {dim[2]}")
            counts = get_used_palette_counts(v_ctx.mesh)
            occupied = int(sum(counts.values()))
            if occupied:
                layout.label(text=f"Volume has {occupied} voxels — enable Clear and Replace", icon="ERROR")
                layout.prop(self, "clear_and_replace")
            else:
                layout.label(text="Volume is empty", icon="INFO")
            cells = dim[0] * dim[1] * dim[2]
            if cells > LARGE_CELL_COUNT:
                layout.label(text=f"Large target ({cells} cells) may be slow", icon="ERROR")
        layout.prop(self, "padding")
        layout.prop(self, "occupancy")
        layout.prop(self, "palette_size")
        layout.prop(self, "alpha_cutoff")
        layout.prop(self, "keep_source")

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {"CANCELLED"}
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({"WARNING"}, "Select a voxel volume first")
            return {"CANCELLED"}
        obj = v_ctx.surface_object
        path = Path(self.filepath)
        if not path.is_file():
            self.report({"ERROR"}, f"File not found: {self.filepath}")
            return {"CANCELLED"}
        suffix = path.suffix.lower()
        if suffix not in (".glb", ".gltf"):
            self.report({"ERROR"}, "Choose a .glb or .gltf file")
            return {"CANCELLED"}

        mesh = v_ctx.mesh
        counts = get_used_palette_counts(mesh)
        occupied = int(sum(counts.values()))
        if occupied and not bool(self.clear_and_replace):
            self.report(
                {"ERROR"},
                "Volume is not empty. Enable Clear and Replace Volume to overwrite it.",
            )
            return {"CANCELLED"}

        props = mesh.voxel_workspace
        emin = tuple(int(v) for v in props.extent_min)
        emax = tuple(int(v) for v in props.extent_max)
        cells = (emax[0] - emin[0]) * (emax[1] - emin[1]) * (emax[2] - emin[2])
        if cells > LARGE_CELL_COUNT:
            self.report(
                {"WARNING"},
                f"Target volume has {cells} cells; conversion may take a while",
            )

        started = perf_counter()
        try:
            source = stage_glb(
                bpy,
                context,
                str(path),
                obj,
                padding=int(self.padding),
                keep_source=bool(self.keep_source),
            )
            result = voxelize_fitted_mesh(
                source.triangles,
                extent_min=emin,
                extent_max_exclusive=emax,
                occupancy_mode=str(self.occupancy),
                mesh_closed=source.mesh_closed,
                uvs=source.uvs,
                material_indices=source.material_indices,
                materials=source.materials,
                palette_size=int(self.palette_size),
                alpha_cutoff=float(self.alpha_cutoff),
                brick_size=int(props.brick_size),
                warnings=list(source.warnings),
                occupy_min=tuple(int(round(v)) for v in source.fit.usable_min),
                occupy_max=tuple(int(round(v)) for v in source.fit.usable_max),
            )
        except Exception as exc:
            self.report({"ERROR"}, f"GLB import failed: {exc}")
            return {"CANCELLED"}

        try:
            _commit_import(obj, result, "Import GLB into Volume")
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to commit imported volume: {exc}")
            return {"CANCELLED"}

        elapsed = perf_counter() - started
        bricks = len(result.grid.bricks)
        fit = source.fit
        msg = (
            f"Imported {result.occupied_count} voxels, {bricks} bricks, "
            f"{result.palette_color_count} colors, scale {fit.scale:.4g} in {elapsed:.2f}s"
        )
        self.report({"INFO"}, msg)
        for warning in result.warnings:
            self.report({"WARNING"}, warning)
        if context.view_layer is not None:
            for other in context.view_layer.objects.selected:
                other.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


IMPORT_GLB_OPERATOR_CLASSES = [VOXEL_OT_import_glb]
