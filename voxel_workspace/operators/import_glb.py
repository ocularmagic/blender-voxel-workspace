"""Analyze and import GLB/glTF geometry into existing or generated voxel volumes."""
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    BoolProperty = EnumProperty = FloatProperty = IntProperty = StringProperty = None

from ..blender.runtime import get_or_load
from ..blender.object_graph import resolve_volume_context, resolve_authoritative_mesh
from ..constants import DEFAULT_PALETTE
from ..importers.glb_analysis import GLBAnalysis, analyze_glb_file
from ..importers.glb_scene import LARGE_CELL_COUNT, stage_glb
from ..operators.palette import get_used_palette_counts
from ..voxelization.resolution_analysis import volume_dimensions
from ..voxelization.voxelize import voxelize_fitted_mesh


_ANALYSIS_BY_SCENE: Dict[int, GLBAnalysis] = {}


def _scene_key(scene: Any) -> int:
    return int(scene.as_pointer()) if scene is not None and hasattr(scene, "as_pointer") else id(scene)


def get_glb_analysis(scene: Any) -> Optional[GLBAnalysis]:
    return _ANALYSIS_BY_SCENE.get(_scene_key(scene))


def _set_glb_analysis(scene: Any, analysis: GLBAnalysis) -> None:
    _ANALYSIS_BY_SCENE[_scene_key(scene)] = analysis


def _is_voxel_object(context: Any) -> bool:
    v_ctx = resolve_volume_context(context)
    return bool(v_ctx is not None and v_ctx.mesh is not None)


def replace_volume_palette(mesh: Any, palette: List[Tuple[float, float, float, float]]) -> List[Any]:
    """Replace only the authoritative Surface Palette with imported defaults."""
    from ..blender.material_domains import initialize_surface_entry, palette_materials

    props = mesh.voxel_workspace
    old_materials = palette_materials(mesh, domain="SURFACE")
    props.surface_palette.clear()
    empty = props.surface_palette.add()
    empty.index = 0
    empty.name = "Empty"
    empty.color = DEFAULT_PALETTE[0]
    empty.material_owned = True
    for idx, color in enumerate(palette):
        if idx == 0:
            continue
        if idx > 255:
            break
        item = props.surface_palette.add()
        initialize_surface_entry(
            mesh,
            item,
            index=idx,
            name=f"Imported {idx}",
            color=tuple(float(c) for c in color),
        )
    return old_materials


def _as_surface_tagged_grid(grid: Any) -> Any:
    """Convert voxelizer scalar output to the schema-3 tagged Surface authority."""
    from ..core.tagged_grid import TaggedBrick, TaggedVoxelGrid, VoxelDomain

    if isinstance(grid, TaggedVoxelGrid):
        return grid
    tagged = TaggedVoxelGrid(
        extent_min=grid.extent_min,
        extent_max_exclusive=grid.extent_max_exclusive,
        brick_size=grid.brick_size,
    )
    for coord, indices in grid.bricks.items():
        brick = TaggedBrick(grid.brick_size)
        brick.indices = indices.copy()
        brick.domains = np.where(indices > 0, int(VoxelDomain.SURFACE), 0).astype(np.uint8)
        tagged.bricks[coord] = brick
    tagged.dirty_bricks.update(tagged.bricks)
    return tagged


def _commit_import(obj: Any, result: Any, undo_message: str) -> None:
    mesh = obj.data
    uuid_str = mesh.voxel_workspace.uuid
    tagged_grid = _as_surface_tagged_grid(result.grid)
    entry = get_or_load(mesh)
    if entry is None:
        from ..blender.runtime import register_volume

        entry = register_volume(
            uuid_str,
            grid=tagged_grid,
            voxel_size=float(mesh.voxel_workspace.voxel_size),
            brick_size=int(mesh.voxel_workspace.brick_size),
            extent_min=tuple(mesh.voxel_workspace.extent_min),
            extent_max=tuple(mesh.voxel_workspace.extent_max),
        )
    else:
        entry.grid = tagged_grid
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
    voxel_size = float(mesh.voxel_workspace.voxel_size)
    sync_volume_mesh(mesh, grid=entry.grid, entry=entry, dirty_only=False, ensure_material=True, voxel_size=voxel_size)
    update_volume_gpu_preview(entry, dirty_only=False)
    from ..blender.material_domains import cleanup_owned_materials

    cleanup_owned_materials(old_materials)
    if bpy is not None and hasattr(bpy.ops, "ed") and hasattr(bpy.ops.ed, "undo_push"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass


def _quality_maximum(analysis: GLBAnalysis, quality: str, custom: int) -> int:
    rec = analysis.recommendations
    return {
        "DRAFT": rec.draft,
        "BALANCED": rec.balanced,
        "FINE": rec.fine,
        "CUSTOM": max(1, int(custom)),
    }.get(str(quality), rec.balanced)


def _palette_maximum(analysis: Optional[GLBAnalysis], quality: str, custom: int) -> int:
    if analysis is None or str(quality) == "CUSTOM":
        return max(1, min(255, int(custom)))
    rec = analysis.palette_recommendations
    return {
        "DRAFT": rec.draft,
        "BALANCED": rec.balanced,
        "FINE": rec.fine,
    }.get(str(quality), rec.balanced)


def _preserving_voxel_size(source_dims: Tuple[float, float, float], dims: Tuple[int, int, int], padding: int) -> float:
    usable = [int(v) - 2 * int(padding) for v in dims]
    if any(v <= 0 for v in usable):
        raise ValueError("Padding leaves no usable volume interior")
    ratios = [float(source_dims[i]) / usable[i] for i in range(3) if float(source_dims[i]) > 1e-12]
    if not ratios:
        raise ValueError("Source geometry has degenerate dimensions")
    return max(ratios)


def _cleanup_created_volume(root: Any, mesh: Any, uuid_str: str, old_material_names: set[str]) -> None:
    from ..blender.runtime import unregister_volume

    unregister_volume(uuid_str)
    objects = list(getattr(root, "children_recursive", [])) + ([root] if root is not None else [])
    for obj in objects:
        if obj is not None and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.name in bpy.data.meshes and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.name not in old_material_names and material.users == 0:
            bpy.data.materials.remove(material)


class VOXEL_OT_analyze_glb(Operator):
    """Inspect GLB meshes and estimate aspect-aware exterior detail resolutions."""

    bl_idname = "voxel.analyze_glb"
    bl_label = "Analyze GLB"
    bl_description = "Choose a GLB/glTF and estimate Draft, Balanced, and Fine exterior detail sizes"
    bl_options = {"REGISTER"}

    if bpy is not None:
        filepath: StringProperty(subtype="FILE_PATH", default="")
        filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})

    def invoke(self, context: Any, event: Any) -> set:
        props = context.scene.voxel_workspace
        if props.glb_filepath:
            self.filepath = props.glb_filepath
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set:
        path = Path(self.filepath)
        if not path.is_file() or path.suffix.lower() not in (".glb", ".gltf"):
            self.report({"ERROR"}, "Choose an existing .glb or .gltf file")
            return {"CANCELLED"}
        started = perf_counter()
        try:
            analysis = analyze_glb_file(bpy, context, str(path))
        except Exception as exc:
            self.report({"ERROR"}, f"GLB analysis failed: {exc}")
            return {"CANCELLED"}
        _set_glb_analysis(context.scene, analysis)
        props = context.scene.voxel_workspace
        props.glb_filepath = str(path)
        props.glb_target_mode = "MAX_AXIS"
        props.glb_quality = "BALANCED"
        props.glb_custom_max_axis = int(analysis.recommendations.balanced)
        props.glb_palette_quality = "BALANCED"
        props.glb_palette_size = int(analysis.palette_recommendations.balanced)
        primary = next((item.name for item in analysis.objects if item.primary), analysis.objects[0].name)
        self.report(
            {"INFO"},
            f"Analyzed {len(analysis.objects)} meshes; primary {primary}; Balanced {analysis.recommendations.balanced} ({perf_counter() - started:.2f}s)",
        )
        return {"FINISHED"}


class VOXEL_OT_toggle_glb_object(Operator):
    """Include or exclude one analyzed mesh object."""

    bl_idname = "voxel.toggle_glb_object"
    bl_label = "Toggle GLB Mesh"
    bl_options = {"INTERNAL"}

    if bpy is not None:
        object_index: IntProperty(default=0, min=0)

    def execute(self, context: Any) -> set:
        analysis = get_glb_analysis(context.scene)
        if analysis is None or self.object_index >= len(analysis.objects):
            return {"CANCELLED"}
        row = analysis.objects[self.object_index]
        row.included = not row.included
        try:
            analysis.refresh_recommendations()
        except ValueError:
            row.included = True
            self.report({"WARNING"}, "At least one mesh object must remain selected")
            return {"CANCELLED"}
        context.scene.voxel_workspace.glb_custom_max_axis = int(analysis.recommendations.balanced)
        context.scene.voxel_workspace.glb_palette_size = int(analysis.palette_recommendations.balanced)
        return {"FINISHED"}


class VOXEL_OT_import_glb(Operator):
    """Voxelize a GLB/glTF into an existing or newly generated Voxel Workspace volume."""

    bl_idname = "voxel.import_glb"
    bl_label = "Import GLB"
    bl_description = "Create or select a voxel volume, then voxelize analyzed GLB geometry"
    bl_options = {"REGISTER"}

    filename_ext = ".glb"

    if bpy is not None:
        filepath: StringProperty(name="File Path", subtype="FILE_PATH", default="")
        filter_glob: StringProperty(default="*.glb;*.gltf", options={"HIDDEN"})
        target_mode: EnumProperty(
            name="Target",
            items=[
                ("EXISTING", "Selected Volume", "Import into the selected voxel volume"),
                ("MAX_AXIS", "Maximum Axis", "Create an aspect-matched volume"),
                ("PANEL_DIMENSIONS", "Panel Dimensions", "Create using the panel X, Y, Z values"),
            ],
            default="EXISTING",
        )
        quality: EnumProperty(
            name="Exterior Detail",
            items=[
                ("DRAFT", "Draft", "Lower-cost exterior silhouette"),
                ("BALANCED", "Balanced", "Estimated detail/cost knee"),
                ("FINE", "Fine", "More thin-feature and recess detail"),
                ("CUSTOM", "Custom", "Use Maximum Axis"),
            ],
            default="BALANCED",
        )
        maximum_axis: IntProperty(name="Maximum Axis", default=128, min=1, max=512)
        padding: IntProperty(name="Padding", default=1, min=0, max=32)
        override_voxel_size: BoolProperty(name="Override Voxel Size", default=False)
        voxel_size: FloatProperty(name="Voxel Size", default=1.0, min=0.0001)
        occupancy: EnumProperty(
            name="Occupancy",
            items=[("SOLID", "Solid", "Fill closed interiors"), ("SHELL", "Surface Shell", "Exterior shell only")],
            default="SOLID",
        )
        palette_quality: EnumProperty(
            name="Palette Detail",
            items=[
                ("DRAFT", "Draft", "Simplified dominant color families"),
                ("BALANCED", "Balanced", "Estimated perceptual color/cost knee"),
                ("FINE", "Fine", "More subtle color variation"),
                ("CUSTOM", "Custom", "Use Maximum Colors"),
            ],
            default="CUSTOM",
        )
        palette_size: IntProperty(name="Maximum Colors", default=64, min=1, max=255)
        alpha_cutoff: FloatProperty(name="Alpha Cutoff", default=0.1, min=0.0, max=1.0)
        clear_and_replace: BoolProperty(name="Clear and Replace Volume", default=False)

    def invoke(self, context: Any, event: Any) -> set:
        if not self.filepath:
            analysis = get_glb_analysis(context.scene)
            if analysis is not None:
                self.filepath = analysis.filepath
        if self.filepath:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.prop(self, "target_mode")
        if self.target_mode != "EXISTING":
            if self.target_mode == "MAX_AXIS":
                layout.prop(self, "quality")
                if self.quality == "CUSTOM":
                    layout.prop(self, "maximum_axis")
            layout.prop(self, "override_voxel_size")
            if self.override_voxel_size:
                layout.prop(self, "voxel_size")
        layout.prop(self, "padding")
        layout.prop(self, "occupancy")
        layout.prop(self, "palette_quality")
        if self.palette_quality == "CUSTOM":
            layout.prop(self, "palette_size")
        layout.prop(self, "alpha_cutoff")

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {"CANCELLED"}
        path = Path(self.filepath)
        if not path.is_file() or path.suffix.lower() not in (".glb", ".gltf"):
            self.report({"ERROR"}, "Choose an existing .glb or .gltf file")
            return {"CANCELLED"}

        analysis = get_glb_analysis(context.scene)
        if analysis is not None and Path(analysis.filepath) != path:
            analysis = None
        needs_analysis = self.target_mode != "EXISTING"
        if analysis is None and needs_analysis:
            try:
                analysis = analyze_glb_file(bpy, context, str(path))
                _set_glb_analysis(context.scene, analysis)
            except Exception as exc:
                self.report({"ERROR"}, f"GLB analysis failed: {exc}")
                return {"CANCELLED"}

        created_root = None
        created_mesh = None
        created_uuid = ""
        old_material_names = {material.name for material in bpy.data.materials}
        if self.target_mode == "EXISTING":
            v_ctx = resolve_volume_context(context)
            if v_ctx is None or v_ctx.mesh is None:
                self.report({"WARNING"}, "Select a voxel volume or choose a create target")
                return {"CANCELLED"}
            counts = get_used_palette_counts(v_ctx.mesh)
            occupied = int(sum(counts.values()))
            if occupied and not bool(self.clear_and_replace):
                self.report({"ERROR"}, "Volume is not empty. Enable Clear and Replace Volume to overwrite it.")
                return {"CANCELLED"}
        else:
            assert analysis is not None
            props = context.scene.voxel_workspace
            if self.target_mode == "PANEL_DIMENSIONS":
                dims = (int(props.create_size_x), int(props.create_size_y), int(props.create_size_z))
            else:
                maximum = _quality_maximum(analysis, self.quality, self.maximum_axis)
                try:
                    dims = volume_dimensions(analysis.dimensions, maximum, int(self.padding))
                except ValueError as exc:
                    self.report({"ERROR"}, str(exc))
                    return {"CANCELLED"}
            try:
                target_voxel_size = (
                    float(self.voxel_size)
                    if bool(self.override_voxel_size)
                    else _preserving_voxel_size(analysis.dimensions, dims, int(self.padding))
                )
                created = bpy.ops.voxel.create_volume(
                    size_x=dims[0], size_y=dims[1], size_z=dims[2],
                    voxel_size=target_voxel_size, push_undo=False,
                )
                if "FINISHED" not in created:
                    raise RuntimeError(f"volume creation failed: {created}")
                v_ctx = resolve_volume_context(context)
                if v_ctx is None or v_ctx.mesh is None:
                    raise RuntimeError("created voxel volume could not be resolved")
                created_root = v_ctx.root
                created_mesh = v_ctx.mesh
                created_uuid = str(created_mesh.voxel_workspace.uuid)
            except Exception as exc:
                self.report({"ERROR"}, f"Failed to create import volume: {exc}")
                return {"CANCELLED"}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        emin = tuple(int(v) for v in props.extent_min)
        emax = tuple(int(v) for v in props.extent_max)
        cells = (emax[0] - emin[0]) * (emax[1] - emin[1]) * (emax[2] - emin[2])
        if cells > LARGE_CELL_COUNT:
            self.report({"WARNING"}, f"Target volume has {cells} cells; conversion may take a while")

        started = perf_counter()
        try:
            included_names = analysis.included_names if analysis is not None else None
            palette_limit = _palette_maximum(analysis, self.palette_quality, self.palette_size)
            source = stage_glb(
                bpy, context, str(path), v_ctx.surface_object,
                padding=int(self.padding),
                included_object_names=included_names,
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
                palette_size=palette_limit,
                alpha_cutoff=float(self.alpha_cutoff),
                brick_size=int(props.brick_size),
                warnings=list(source.warnings),
                occupy_min=tuple(int(round(v)) for v in source.fit.usable_min),
                occupy_max=tuple(int(round(v)) for v in source.fit.usable_max),
            )
            _commit_import(v_ctx.surface_object, result, "Create and Import GLB" if created_root else "Import GLB into Volume")
        except Exception as exc:
            if created_root is not None:
                _cleanup_created_volume(created_root, created_mesh, created_uuid, old_material_names)
            self.report({"ERROR"}, f"GLB import failed: {exc}")
            return {"CANCELLED"}

        try:
            bpy.ops.wm.preview_ensure()
        except Exception:
            pass
        elapsed = perf_counter() - started
        dims = (emax[0] - emin[0], emax[1] - emin[1], emax[2] - emin[2])
        self.report(
            {"INFO"},
            f"Imported {result.occupied_count} voxels into {dims[0]} × {dims[1]} × {dims[2]}, {result.palette_color_count} colors in {elapsed:.2f}s",
        )
        for warning in result.warnings:
            self.report({"WARNING"}, warning)
        if context.view_layer is not None:
            for other in context.view_layer.objects.selected:
                other.select_set(False)
            target = created_root if created_root is not None else v_ctx.root
            target = target if target is not None else v_ctx.surface_object
            target.select_set(True)
            context.view_layer.objects.active = target
        return {"FINISHED"}


IMPORT_GLB_OPERATOR_CLASSES = [
    VOXEL_OT_analyze_glb,
    VOXEL_OT_toggle_glb_object,
    VOXEL_OT_import_glb,
]
