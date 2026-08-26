"""Blender integration test for the Stretch/Squash Interior (scale) feature."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tests.blender.bootstrap import setup_test_environment

setup_test_environment()


def run_test():
    import bpy
    import voxel_workspace
    from voxel_workspace.core.scale_volume import (
        compute_scale_writes,
        validate_scale_extent,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.use_global_undo = True

    print("--- Phase 1: Registration ---")
    voxel_workspace.register()
    assert hasattr(bpy.ops.voxel, "scale_voxels"), "scale_voxels must register"
    assert hasattr(bpy.ops.voxel, "accept_scale_voxels")
    assert hasattr(bpy.ops.voxel, "cancel_scale_voxels")

    # SCALE enum value must exist on the real scene property.
    props = bpy.context.scene.voxel_workspace
    items = dict((k, n) for k, n, _ in
                 [(i.identifier, i.name, i.description)
                  for i in props.bl_rna.properties["active_tool"].enum_items])
    assert "SCALE" in items, f"SCALE missing from active_tool: {items}"

    # Calling without a volume must not start anything harmful.
    try:
        res = bpy.ops.voxel.scale_voxels()
        assert res in ({'CANCELLED'}, {'PASS_THROUGH'}, {'RUNNING_MODAL'}), \
            f"unexpected result without a volume: {res}"
        if 'RUNNING_MODAL' in res:
            bpy.ops.voxel.cancel_scale_voxels()
    except RuntimeError:
        pass  # poll rejection raises in headless call

    print("--- Phase 2: Create volume and paint voxels ---")
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    root = bpy.context.active_object
    root.name = "VoxelRoot"
    try:
        bpy.ops.voxel.create_volume(size_x=4, size_y=4, size_z=2)
    except Exception as exc:  # pragma: no cover - depends on selection semantics
        raise AssertionError(f"create_volume failed: {exc}")

    from voxel_workspace.blender.object_graph import resolve_volume_context
    from voxel_workspace.blender.runtime import get_or_load

    vctx = resolve_volume_context(bpy.context)
    assert vctx is not None and vctx.mesh is not None, "volume context must resolve"
    mesh = vctx.mesh
    entry = get_or_load(mesh)
    grid = entry.grid

    emin = tuple(int(c) for c in mesh.voxel_workspace.extent_min)
    emax = tuple(int(c) for c in mesh.voxel_workspace.extent_max)
    print(f"extent {emin}..{emax}")

    # Paint a solid 4x4 slab on the bottom layer using set_cell.
    x0, y0, z0 = emin
    for dx in range(4):
        for dy in range(4):
            grid.set_cell((x0 + dx, y0 + dy, z0), 1, 3)

    print("--- Phase 3: scale math on the live grid ---")
    old_extent = (emin, emax)
    new_extent = ((emin[0], emin[1], emin[2]),
                  (emax[0] * 2, emax[1], emax[2]))
    writes = compute_scale_writes(grid, old_extent, new_extent, axis=0)
    xs = sorted({k[0][0] for k in writes})
    # Full-extent slab must fill the doubled X span contiguously.
    assert len(xs) == (new_extent[1][0] - new_extent[0][0]), xs
    assert xs == list(range(xs[0], xs[-1] + 1)), "stretch must be solid"
    doms = {dom for (_, dom) in writes}
    assert doms == {1}, doms
    vals = {v for v in writes.values()}
    assert vals == {3}, vals

    # Squash to one voxel along X collapses but keeps values non-empty.
    tiny = ((emin[0], emin[1], emin[2]), (emin[0] + 1, emax[1], emax[2]))
    squash = compute_scale_writes(grid, old_extent, tiny, axis=0)
    assert len(squash) == 4, len(squash)  # Y spans 4; X collapses to 1; one painted Z layer
    assert all(v != 0 for v in squash.values())

    # validate_scale_extent bounds.
    assert validate_scale_extent((0, 0, 0), (513, 1, 1)) is not None
    assert validate_scale_extent((0, 0, 0), (0, 1, 1)) is not None
    assert validate_scale_extent((-2, -2, 0), (2, 2, 4)) is None

    print("--- Phase 4: brush refuses to overwrite boundary voxels ---")
    from voxel_workspace.core.line import compute_brush_target
    from voxel_workspace.core.grid import VoxelGrid as PlainGrid

    # Small fully-populated flat grid: every voxel is at an extent boundary.
    pg = PlainGrid((0, 0, 0), (3, 3, 1))
    for x in range(3):
        for y in range(3):
            pg.set((x, y, 0), 2)
    # Ray straight down onto the top face of a boundary voxel: add target
    # would be (cell + normal) = outside Z extent. It must NOT clamp back
    # onto the boundary voxel; it must be refused entirely.
    target, hover, normal = compute_brush_target(pg, (1.5, 1.5, 10.0),
                                                 (0.0, 0.0, -1.0),
                                                 mode="ADD_SURFACE")
    assert target is None, f"add past full-to-edge root must refuse, got {target}"
    # Erase still works on the same hit.
    target_e, hover_e, normal_e = compute_brush_target(pg, (1.5, 1.5, 10.0),
                                                       (0.0, 0.0, -1.0),
                                                       mode="ERASE")
    assert target_e == (1, 1, 0), target_e
    # A ray hitting a boundary voxel from OUTSIDE must also refuse (its
    # outer wall has no free neighbour); previously this clamped and
    # overwrote the boundary voxel.
    target_w, _, _ = compute_brush_target(pg, (-5.0, 1.5, 0.5),
                                          (1.0, 0.0, 0.0),
                                          mode="ADD_SURFACE")
    assert target_w is None, f"outer-wall add must refuse, got {target_w}"
    # But an EMPTY grid still offers the viewer-facing wall for drawing.
    eg = PlainGrid((0, 0, 0), (3, 3, 1))
    target_empty, _, _ = compute_brush_target(eg, (-5.0, 1.5, 0.5),
                                              (1.0, 0.0, 0.0),
                                              mode="ADD_SURFACE")
    assert target_empty == (2, 1, 0), target_empty  # EXIT wall (far side), +X ray

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_test()
