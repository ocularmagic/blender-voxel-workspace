"""Export exact visible Surface voxel faces as vertex-colour OBJ."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

try:
    import bpy
    from mathutils import Matrix
    from bpy.props import BoolProperty, StringProperty
    from bpy.types import Operator
    from bpy_extras.io_utils import ExportHelper
except ImportError:
    bpy = None
    Matrix = None
    Operator = object
    ExportHelper = object
    BoolProperty = StringProperty = None

from ..blender.material_domains import display_rgba_from_entry, find_entry
from ..blender.object_graph import resolve_volume_context
from ..blender.runtime import get_or_load
from ..core.tagged_grid import VoxelDomain
from ..geometry.voxel_lined_export import (
    OBJ_Y_UP_CONVERSION,
    build_voxel_lined_mesh,
    write_vertex_color_obj,
)


class VOXEL_OT_export_obj(Operator, ExportHelper):
    """Export exact per-visible-voxel Surface geometry with RGB vertex colors."""

    bl_idname = "voxel.export_obj"
    bl_label = "Export Exact Voxel-Lined OBJ"
    bl_description = (
        "Export visible Surface voxel faces with inset palette-colored centers "
        "and grey perimeter strips as an OBJ with vertex colors"
    )
    bl_options = {"REGISTER"}
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={"HIDDEN"})
    check_existing = False
    overwrite: BoolProperty(
        name="Overwrite",
        description="Skip overwrite confirmation (used by the confirmation operator)",
        default=False,
        options={"HIDDEN"},
    )

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({"WARNING"}, "Select a voxel volume first")
            return {"CANCELLED"}
        name = v_ctx.root.name if v_ctx.root is not None else "VoxelSurface"
        if not self.filepath:
            self.filepath = str(Path(tempfile.gettempdir()) / f"{name}.obj")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({"WARNING"}, "Select a voxel volume first")
            return {"CANCELLED"}
        target = Path(self.filepath).with_suffix(".obj")
        if target.exists() and not self.overwrite:
            bpy.ops.voxel.export_obj_confirm(filepath=str(target))
            return {"FINISHED"}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        entry = get_or_load(mesh)
        if entry is None or entry.grid is None:
            self.report({"ERROR"}, "No voxel grid available to export")
            return {"CANCELLED"}
        grid = entry.grid
        if not grid.iter_used_indices(VoxelDomain.SURFACE):
            self.report({"WARNING"}, "Voxel volume has no Surface cells")
            return {"CANCELLED"}
        root_props = getattr(v_ctx.root, "voxel_workspace", None)
        edge_width = float(getattr(root_props, "rendered_surface_edge_width", 0.01))
        edge_color = tuple(getattr(root_props, "rendered_surface_edge_color", (0.0, 0.0, 0.0, 1.0)))
        voxel_size = float(getattr(props, "voxel_size", 1.0))

        def color_for_index(index: int):
            palette_entry = find_entry(mesh, VoxelDomain.SURFACE, int(index))
            if palette_entry is None:
                return (0.8, 0.8, 0.8)
            return display_rgba_from_entry(palette_entry, "SURFACE")

        try:
            root_matrix = v_ctx.root.matrix_world if v_ctx.root is not None else None
            if root_matrix is not None:
                # Preserve the root placement, then convert Blender Z-up to
                # the Y-up convention expected by common OBJ consumers.
                transform_matrix = Matrix(OBJ_Y_UP_CONVERSION) @ root_matrix
                transform = tuple(tuple(float(v) for v in row) for row in transform_matrix)
            else:
                transform = OBJ_Y_UP_CONVERSION
            lined = build_voxel_lined_mesh(
                grid,
                color_for_index,
                voxel_size=voxel_size,
                edge_width=edge_width,
                edge_color=edge_color[:3],
                transform=transform,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                write_vertex_color_obj(str(temp_path), lined, edge_width, transform_space="OBJ_Y_UP_WORLD")
                os.replace(temp_path, target)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"OBJ export failed: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Exported {len(lined.faces)} triangles to {target}")
        return {"FINISHED"}


class VOXEL_OT_export_obj_confirm(Operator):
    """Confirm replacement of an existing vertex-colour OBJ."""

    bl_idname = "voxel.export_obj_confirm"
    bl_label = "Overwrite existing voxel-lined OBJ?"
    bl_description = "The existing OBJ will be replaced"
    bl_options = {"REGISTER", "INTERNAL"}

    if bpy is not None:
        filepath: StringProperty(subtype="FILE_PATH", default="")

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Any) -> None:
        self.layout.label(text="Overwrite existing voxel-lined OBJ?")
        self.layout.label(text="The existing file will be replaced.", icon="ERROR")

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        bpy.ops.voxel.export_obj(filepath=self.filepath, overwrite=True)
        return {"FINISHED"}


EXPORT_OBJ_OPERATOR_CLASSES = [VOXEL_OT_export_obj, VOXEL_OT_export_obj_confirm]
