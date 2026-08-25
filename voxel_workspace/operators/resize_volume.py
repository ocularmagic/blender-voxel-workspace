"""Resize Voxel Volume operator: grow/shrink an existing volume's dimensions."""
from typing import Any, Optional, Tuple

try:
    import bpy
    from bpy.props import EnumProperty, IntProperty, StringProperty
    from bpy.types import Operator
except ImportError:
    bpy = None
    Operator = object
    EnumProperty = IntProperty = StringProperty = None

import numpy as np

from ..blender.object_graph import resolve_volume_context

MAX_DIM = 512

ANCHOR_ITEMS = [
    ("CENTER", "Center", "Keep this axis centered (like creation)"),
    ("MIN", "Min", "Pin the low face; move the high face"),
    ("MAX", "Max", "Pin the high face; move the low face"),
]


def _occupied_bounds(grid: Any) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return (min_coord, max_coord_inclusive) over occupied cells, or None if empty."""
    from ..core.coords import join_coord

    bs = int(grid.brick_size)
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    for bcoord, brick in grid.bricks.items():
        nz = np.argwhere(brick.indices > 0)
        if len(nz) == 0:
            continue
        gc = np.stack([join_coord(bcoord, (int(c[0]), int(c[1]), int(c[2])), bs) for c in nz])
        mn = gc.min(axis=0)
        mx = gc.max(axis=0)
        lo = mn if lo is None else np.minimum(lo, mn)
        hi = mx if hi is None else np.maximum(hi, mx)
    if lo is None:
        return None
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def _axis_new_range(
    axis: int,
    new_size: int,
    anchor: str,
    old_min: int,
    old_max: int,
) -> Tuple[int, int]:
    """Compute the new (min, max_exclusive) range for one axis given an anchor."""
    if anchor == "MIN":
        return old_min, old_min + new_size
    if anchor == "MAX":
        return old_max - new_size, old_max
    # CENTER: preserve the current integer midpoint
    center = (old_min + old_max) // 2
    new_min = center - new_size // 2
    return new_min, new_min + new_size


def compute_new_extents(
    extent_min: Tuple[int, int, int],
    extent_max: Tuple[int, int, int],
    sizes: Tuple[int, int, int],
    anchors: Tuple[str, str, str],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return the new (extent_min, extent_max_exclusive) for the requested sizes/anchors."""
    new_min = []
    new_max = []
    for axis in range(3):
        nmin, nmax = _axis_new_range(axis, int(sizes[axis]), anchors[axis], extent_min[axis], extent_max[axis])
        new_min.append(nmin)
        new_max.append(nmax)
    return tuple(new_min), tuple(new_max)


def validate_no_orphaned_voxels(
    grid: Any,
    new_min: Tuple[int, int, int],
    new_max: Tuple[int, int, int],
) -> Optional[str]:
    """Return an error message if any occupied cell would fall outside the new extent."""
    bounds = _occupied_bounds(grid)
    if bounds is None:
        return None
    occ_lo, occ_hi = bounds
    problems = []
    for i, name in enumerate(("X", "Y", "Z")):
        if occ_lo[i] < new_min[i] or occ_hi[i] >= new_max[i]:
            cur = new_max[i] - new_min[i]
            need = max(cur, occ_hi[i] - new_min[i] + 1, new_max[i] - occ_lo[i])
            problems.append(f"{name} must be at least {need}")
    if problems:
        return (
            "Cannot shrink below existing voxels: "
            + "; ".join(problems)
            + ". Voxels would be lost."
        )
    return None


class VOXEL_OT_resize_volume(Operator):
    """Resize the active voxel volume; cannot shrink below placed voxels."""

    bl_idname = "voxel.resize_volume"
    bl_label = "Apply Resize"
    bl_description = (
        "Resize the active voxel volume. Existing voxels keep their positions; "
        "shrinking below any placed voxel is rejected"
    )
    bl_options = {'REGISTER'}

    if bpy is not None:
        target_uuid: StringProperty(
            name="Target UUID",
            description="UUID of the volume to resize",
            default="",
        )
        size_x: IntProperty(name="Size X", description="New X dimension in voxels", default=16, min=1, max=MAX_DIM)
        size_y: IntProperty(name="Size Y", description="New Y dimension in voxels", default=16, min=1, max=MAX_DIM)
        size_z: IntProperty(name="Size Z", description="New Z dimension in voxels", default=16, min=1, max=MAX_DIM)
        anchor_x: EnumProperty(name="X Anchor", description="Where X grows/shrinks from", items=ANCHOR_ITEMS, default="CENTER")
        anchor_y: EnumProperty(name="Y Anchor", description="Where Y grows/shrinks from", items=ANCHOR_ITEMS, default="CENTER")
        anchor_z: EnumProperty(name="Z Anchor", description="Where Z grows/shrinks from", items=ANCHOR_ITEMS, default="MIN")

    @classmethod
    def poll(cls, context: Any) -> bool:
        if context is None:
            return False
        v_ctx = resolve_volume_context(context)
        return v_ctx is not None and v_ctx.mesh is not None

    def invoke(self, context: Any, event: Any) -> set:
        if bpy is None:
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: Any) -> set:
        layout = self.layout
        v_ctx = resolve_volume_context(context)
        if v_ctx is not None and v_ctx.mesh is not None:
            props = v_ctx.mesh.voxel_workspace
            emin = tuple(props.extent_min)
            emax = tuple(props.extent_max)
            layout.label(text=f"Resize {emax[0]-emin[0]}×{emax[1]-emin[1]}×{emax[2]-emin[2]} "
                              f"to {int(self.size_x)}×{int(self.size_y)}×{int(self.size_z)}?")
            layout.label(text="Existing voxels keep their positions.", icon="INFO")
            layout.separator()
        col = layout.column(align=True)
        row_x = col.row(align=True)
        row_x.prop(self, "size_x", text="X")
        row_x.prop(self, "anchor_x", text="")
        row_y = col.row(align=True)
        row_y.prop(self, "size_y", text="Y")
        row_y.prop(self, "anchor_y", text="")
        row_z = col.row(align=True)
        row_z.prop(self, "size_z", text="Z")
        row_z.prop(self, "anchor_z", text="")

    def execute(self, context: Any) -> set:
        if bpy is None or context is None:
            return {'CANCELLED'}

        v_ctx = resolve_volume_context(context)
        if v_ctx is None or v_ctx.mesh is None:
            self.report({'ERROR'}, "Active object is not a valid voxel volume")
            return {'CANCELLED'}
        mesh = v_ctx.mesh
        props = mesh.voxel_workspace

        from ..blender.runtime import get_or_load
        entry = get_or_load(mesh)
        if entry is None or entry.grid is None:
            self.report({'ERROR'}, "No voxel grid available for this volume")
            return {'CANCELLED'}
        grid = entry.grid

        emin = tuple(int(c) for c in props.extent_min)
        emax = tuple(int(c) for c in props.extent_max)

        sizes = (int(self.size_x), int(self.size_y), int(self.size_z))
        anchors = (str(self.anchor_x), str(self.anchor_y), str(self.anchor_z))
        new_min, new_max = compute_new_extents(emin, emax, sizes, anchors)

        err = validate_no_orphaned_voxels(grid, new_min, new_max)
        if err is not None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        # Update authoritative metadata first
        props.extent_min = new_min
        props.extent_max = new_max
        grid.extent_min = new_min
        grid.extent_max_exclusive = new_max

        # Prune bricks that now lie entirely outside the shrunk extent
        bs = int(grid.brick_size)
        from ..core.coords import split_coord
        for bcoord in list(grid.bricks.keys()):
            bmin = (bcoord[0] * bs, bcoord[1] * bs, bcoord[2] * bs)
            bmax = (bmin[0] + bs, bmin[1] + bs, bmin[2] + bs)
            if (
                bmax[0] <= new_min[0] or bmin[0] >= new_max[0]
                or bmax[1] <= new_min[1] or bmin[1] >= new_max[1]
                or bmax[2] <= new_min[2] or bmin[2] >= new_max[2]
            ):
                del grid.bricks[bcoord]
                grid.dirty_bricks.discard(bcoord)

        # Persist full state and rebuild the render mesh + proxies
        from ..blender.persistence import serialize_volume
        serialize_volume(mesh, grid, dirty_only=False)

        from ..blender.mesh_sync import sync_volume_mesh
        entry.cpu_buffers.clear()
        sync_volume_mesh(mesh, grid=grid, entry=entry, dirty_only=False,
                         ensure_material=False, voxel_size=float(entry.voxel_size))

        try:
            bpy.ops.ed.undo_push(message="Resize Voxel Volume")
        except Exception:
            pass

        self.report({'INFO'},
                    f"Volume resized to {sizes[0]}×{sizes[1]}×{sizes[2]}")
        return {'FINISHED'}


RESIZE_OPERATOR_CLASSES = [VOXEL_OT_resize_volume]
