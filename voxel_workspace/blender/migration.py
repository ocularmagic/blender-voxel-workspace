"""Validation-first migration from shared-palette scalar volumes to schema-3 tagged fields."""
from dataclasses import dataclass, field
from typing import Any, List
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

from ..constants import DEFAULT_PALETTE
from ..core.grid import VoxelGrid
from ..core.tagged_grid import TaggedBrick, TaggedVoxelGrid, VoxelDomain


@dataclass
class MigrationReport:
    migrated: bool = False
    mesh_uuid: str = ""
    occupied_cells: int = 0
    roots_created_or_repaired: int = 0
    messages: List[str] = field(default_factory=list)


def _copy_entry(src: Any, dst: Any) -> None:
    dst.index = int(src.index)
    dst.name = str(src.name)
    dst.color = tuple(float(component) for component in src.color)
    dst.material = src.material
    dst.material_owned = bool(src.material_owned)


def migrate_mesh_to_schema3(mesh: Any) -> MigrationReport:
    """Migrate one authoritative Mesh exactly once, preserving indices and Materials."""
    report = MigrationReport(mesh_uuid=getattr(getattr(mesh, "voxel_workspace", None), "uuid", ""))
    if bpy is None or mesh is None or not hasattr(mesh, "voxel_workspace"):
        return report
    props = mesh.voxel_workspace
    if int(props.schema_version) >= 3:
        return report

    legacy_entries = {}
    for entry in props.palette:
        if int(entry.index) <= 0:
            continue
        domain = str(getattr(entry, "material_domain", "SURFACE")).upper()
        if domain not in {"SURFACE", "VOLUME"}:
            raise ValueError(f"Legacy palette index {entry.index} has invalid domain {domain!r}")
        legacy_entries[int(entry.index)] = {
            "entry": entry,
            "domain": VoxelDomain.VOLUME if domain == "VOLUME" else VoxelDomain.SURFACE,
        }

    from .persistence import deserialize_volume, serialize_volume
    scalar = deserialize_volume(
        mesh,
        grid=VoxelGrid(
            extent_min=tuple(props.extent_min),
            extent_max_exclusive=tuple(props.extent_max),
            brick_size=int(props.brick_size),
        ),
    )
    tagged = TaggedVoxelGrid(
        extent_min=scalar.extent_min,
        extent_max_exclusive=scalar.extent_max_exclusive,
        brick_size=scalar.brick_size,
    )

    for coord, indices in scalar.bricks.items():
        used = {int(value) for value in np.unique(indices) if int(value) > 0}
        missing = sorted(used - set(legacy_entries))
        if missing:
            raise ValueError(f"Legacy brick {coord} references missing palette indices {missing}")
        brick = TaggedBrick(tagged.brick_size)
        brick.indices = indices.copy()
        brick.domains = np.zeros_like(indices, dtype=np.uint8)
        for index in used:
            brick.domains[indices == index] = int(legacy_entries[index]["domain"])
        tagged.bricks[coord] = brick
        report.occupied_cells += int(np.count_nonzero(indices))
    tagged.validate()

    # Build independent typed palettes by pointer identity; do not regenerate graphs.
    props.surface_palette.clear()
    props.volume_palette.clear()
    for collection in (props.surface_palette, props.volume_palette):
        empty = collection.add()
        empty.index = 0
        empty.name = "Empty"
        empty.color = DEFAULT_PALETTE[0]
        empty.material_owned = True

    from .material_domains import ensure_entry_material
    for index in sorted(legacy_entries):
        info = legacy_entries[index]
        collection = props.volume_palette if info["domain"] == VoxelDomain.VOLUME else props.surface_palette
        dst = collection.add()
        _copy_entry(info["entry"], dst)
        ensure_entry_material(mesh, dst, info["domain"])

    # Persist only after the complete in-memory representation validates.
    tagged.dirty_bricks.update(tagged.bricks)
    serialize_volume(mesh, tagged, dirty_only=False)
    props.palette_schema_version = 3

    from .object_graph import ensure_root_for_surface, repair_voxel_hierarchy
    from .volume_proxy import PROXY_OBJECT_FLAG
    for obj in list(bpy.data.objects):
        if getattr(obj, "type", None) == 'MESH' and getattr(obj, "data", None) == mesh and not obj.get(PROXY_OBJECT_FLAG, False):
            root = ensure_root_for_surface(obj)
            if root is not None:
                repair_voxel_hierarchy(root)
                report.roots_created_or_repaired += 1

    report.migrated = True
    return report
