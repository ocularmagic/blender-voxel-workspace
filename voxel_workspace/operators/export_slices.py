"""Export the voxel volume as one PNG per Z-layer, bottom-to-top."""
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    import bpy
    from bpy.props import BoolProperty, StringProperty
    from bpy.types import Operator
    from bpy_extras.io_utils import ExportHelper
except ImportError:
    bpy = None
    Operator = object
    ExportHelper = object
    BoolProperty = StringProperty = None

from ..blender.object_graph import resolve_volume_context
from ..blender.runtime import get_or_load, tag_redraw_all_viewports
from ..core.tagged_grid import VoxelDomain


def _extract_base(filepath: str) -> Tuple[Path, str]:
    """Resolve the output directory and base stem from a chosen file path.

    Handles both a plain base name ("VoxelRoot.png") and an already-suffixed
    layer file ("VoxelRoot_000.png") so re-selecting an existing layer resolves
    back to the correct base ("VoxelRoot").
    """
    p = Path(filepath)
    stem = p.stem
    m = re.search(r"_\d+$", stem)
    if m:
        stem = stem[: m.start()]
    return p.parent, stem


class VOXEL_OT_export_slices(Operator, ExportHelper):
    """Export the voxel volume as PNG slices, one per Z layer, bottom-to-top."""

    bl_idname = "voxel.export_slices"
    bl_label = "Export Voxel Slices"
    bl_description = (
        "Export one PNG per Z layer (bottom-to-top). Each PNG is one pixel per "
        "voxel at that level; surface voxels are their display color, empty and "
        "volume voxels are transparent. Files are named <base>_NNN.png."
    )
    bl_options = {"REGISTER"}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})
    # Disable ExportHelper's single-file auto-check: this export writes derived
    # files, so overwrite confirmation is handled via a dedicated confirm step.
    check_existing = False

    overwrite: BoolProperty(
        name="Overwrite",
        description="Skip the overwrite confirmation (set by the confirm operator)",
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
        mesh = v_ctx.mesh
        name_hint = v_ctx.root.name if v_ctx.root else (v_ctx.surface_object.name if v_ctx.surface_object else "Volume")
        base = "".join(c for c in name_hint if c.isalnum() or c in ("_", "-")) or "slices"
        if not self.filepath:
            self.filepath = str(Path(tempfile.gettempdir()) / f"{base}.png")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context: Any) -> None:
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            layout.label(text="No active voxel volume", icon="ERROR")
            return
        props = v_ctx.mesh.voxel_workspace
        emin = tuple(props.extent_min)
        emax = tuple(props.extent_max)
        dim = (emax[0] - emin[0], emax[1] - emin[1], emax[2] - emin[2])
        layout.label(text=f"Slices: {dim[2]} images of {dim[0]} × {dim[1]} px", icon="INFO")
        layout.label(text="Surface voxels use their display color.")
        layout.label(text="Empty and volume voxels are transparent.")

    def _color_for_cell(self, mesh: Any, cell: Any, pal_type: str) -> Optional[Tuple[float, float, float, float]]:
        """Return the display RGBA for a surface cell, else None for transparent."""
        if cell is None or cell.index <= 0:
            return None
        if cell.domain != VoxelDomain.SURFACE:
            return None
        from ..blender.material_domains import find_entry, display_rgba_from_entry
        entry = find_entry(mesh, VoxelDomain.SURFACE, int(cell.index))
        if entry is None:
            return None
        return display_rgba_from_entry(entry, pal_type)

    def _target_paths(self, out_dir: Path, base: str, dim_z: int) -> List[Path]:
        """Compute the full list of derived layer file paths."""
        pad = max(3, len(str(dim_z - 1)))
        return [out_dir / f"{base}_{z:0{pad}d}.png" for z in range(dim_z)]

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({"WARNING"}, "Select a voxel volume first")
            return {"CANCELLED"}

        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        entry = get_or_load(mesh)
        if entry is None or entry.grid is None:
            self.report({"ERROR"}, "No voxel grid available to export")
            return {"CANCELLED"}
        grid = entry.grid

        emin = tuple(int(c) for c in props.extent_min)
        emax = tuple(int(c) for c in props.extent_max)
        dim_x = emax[0] - emin[0]
        dim_y = emax[1] - emin[1]
        dim_z = emax[2] - emin[2]
        if dim_x <= 0 or dim_y <= 0 or dim_z <= 0:
            self.report({"ERROR"}, "Voxel volume has invalid dimensions")
            return {"CANCELLED"}

        out_dir, base = _extract_base(self.filepath)
        targets = self._target_paths(out_dir, base, dim_z)

        # Single overwrite confirmation against the actual layer files.
        existing = [p for p in targets if p.exists()]
        if existing and not self.overwrite:
            bpy.ops.voxel.export_slices_confirm(
                "INVOKE_DEFAULT",
                filepath=self.filepath,
            )
            return {"FINISHED"}

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            pad = max(3, len(str(dim_z - 1)))
            for z in range(dim_z):
                world_z = emin[2] + z
                img = bpy.data.images.new(
                    f"_voxel_slice_{z:0{pad}d}",
                    width=dim_x,
                    height=dim_y,
                    alpha=True,
                )
                pixels = [0.0] * (dim_x * dim_y * 4)
                for y in range(dim_y):
                    world_y = emin[1] + y
                    for x in range(dim_x):
                        world_x = emin[0] + x
                        cell = grid.get_cell((world_x, world_y, world_z))
                        rgba = self._color_for_cell(mesh, cell, "SURFACE")
                        off = (y * dim_x + x) * 4
                        if rgba is not None:
                            pixels[off + 0] = rgba[0]
                            pixels[off + 1] = rgba[1]
                            pixels[off + 2] = rgba[2]
                            pixels[off + 3] = rgba[3]
                        else:
                            pixels[off + 0] = 0.0
                            pixels[off + 1] = 0.0
                            pixels[off + 2] = 0.0
                            pixels[off + 3] = 0.0
                img.pixels = pixels
                img.file_format = "PNG"
                img.filepath_raw = str(targets[z])
                img.save()
                bpy.data.images.remove(img)
                written += 1
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Export failed: {exc}")
            return {"CANCELLED"}

        tag_redraw_all_viewports()
        self.report({"INFO"}, f"Exported {written} slice PNG(s) to {out_dir}")
        return {"FINISHED"}


class VOXEL_OT_export_slices_confirm(Operator):
    """Confirm overwriting existing voxel slice files."""

    bl_idname = "voxel.export_slices_confirm"
    bl_label = "Overwrite existing voxel slices?"
    bl_description = "Existing voxel slice files will be overwritten"
    bl_options = {"REGISTER", "INTERNAL"}

    if bpy is not None:
        filepath: StringProperty(
            name="File Path",
            description="Export file path to continue with",
            subtype="FILE_PATH",
            default="",
        )

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Any) -> None:
        layout = self.layout
        layout.label(text="Overwrite existing voxel slices?")
        layout.label(text="Files with the same base name will be replaced.", icon="ERROR")

    def execute(self, context: Any) -> set:
        if bpy is None:
            return {"CANCELLED"}
        bpy.ops.voxel.export_slices(
            filepath=self.filepath,
            overwrite=True,
        )
        return {"FINISHED"}


EXPORT_OPERATOR_CLASSES = [VOXEL_OT_export_slices, VOXEL_OT_export_slices_confirm]
