"""Voxel brick persistence, IDProperty array serialization, and Blender memfile undo bridge.

Authoritative schema (D2 / D3):
- Metadata is stored on the Mesh datablock via PointerProperty `voxel_workspace` (VoxelMeshProperties):
    - uuid (str)
    - is_voxel_mesh (bool)
    - brick_size (int, default 32)
    - extent_min (int[3])
    - extent_max (int[3])
    - voxel_size (float)
    - schema_version (int, default 1)
- Voxel bricks are stored as deterministic top-level IDProperty arrays on the Mesh datablock:
    - 'vox_brick_<bx>_<by>_<bz>': IDPropertyArray of signed 32-bit ints (4 raw bytes per int)
    - 'vox_brick_<bx>_<by>_<bz>_len': integer original byte length (brick_size^3)
- Empty / zero bricks are pruned from IDProperties to keep memfile state sparse and lightweight.
"""
from array import array
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

from ..constants import BRICK_SIZE, BrickCoord
from ..core.grid import VoxelGrid


SCHEMA_VERSION = 1
BRICK_KEY_PATTERN = re.compile(r"^vox_brick_(-?\d+)_(-?\d+)_(-?\d+)$")


def pack_bytes_to_i32(data: bytes) -> Tuple[List[int], int]:
    """Pack arbitrary raw bytes into a list of signed 32-bit integers.
    
    Returns (list_of_i32, original_byte_len).
    """
    byte_len = len(data)
    if byte_len == 0:
        return [], 0
    pad = (-byte_len) % 4
    a = array('i')
    a.frombytes(data + b'\x00' * pad)
    return a.tolist(), byte_len


def unpack_i32_to_bytes(values: Union[Sequence[int], Any], byte_len: int) -> bytes:
    """Unpack a sequence/IDPropertyArray of signed 32-bit integers back to raw bytes."""
    if byte_len == 0:
        return b""
    if hasattr(values, "to_list"):
        vals = values.to_list()
    elif isinstance(values, list):
        vals = values
    else:
        vals = list(values)
    a = array('i', vals)
    return a.tobytes()[:byte_len]


def pack_brick(brick: np.ndarray) -> Tuple[List[int], int]:
    """Pack a 3D numpy uint8 array (C-contiguous) into signed 32-bit ints."""
    if not isinstance(brick, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(brick)}")
    data = np.ascontiguousarray(brick, dtype=np.uint8).tobytes()
    return pack_bytes_to_i32(data)


def unpack_brick(
    values: Union[Sequence[int], Any],
    byte_len: int,
    brick_size: int = BRICK_SIZE,
) -> np.ndarray:
    """Unpack signed 32-bit ints into a 3D (brick_size, brick_size, brick_size) uint8 numpy array."""
    raw_bytes = unpack_i32_to_bytes(values, byte_len)
    expected_size = brick_size ** 3
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"Byte length {len(raw_bytes)} does not match expected brick volume {expected_size} ({brick_size}^3)"
        )
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((brick_size, brick_size, brick_size)).copy()


def brick_coord_to_key(coord: BrickCoord) -> str:
    """Convert a brick coordinate tuple (bx, by, bz) to custom property key."""
    return f"vox_brick_{coord[0]}_{coord[1]}_{coord[2]}"


def key_to_brick_coord(key: str) -> Optional[BrickCoord]:
    """Parse custom property key to brick coordinate tuple, or None if not a brick key."""
    m = BRICK_KEY_PATTERN.match(key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def serialize_volume(
    mesh: Any,
    grid: VoxelGrid,
    dirty_only: bool = True,
) -> Set[BrickCoord]:
    """Serialize voxel bricks to Mesh custom IDProperties.
    
    If dirty_only=True, only bricks in grid.dirty_bricks are updated/pruned.
    If dirty_only=False, all bricks are serialized and any non-existent bricks pruned.
    Returns the set of brick coordinates processed.
    """
    if hasattr(mesh, "voxel_workspace"):
        mesh.voxel_workspace.schema_version = SCHEMA_VERSION
        mesh.voxel_workspace.brick_size = grid.brick_size
        mesh.voxel_workspace.extent_min = grid.extent_min
        mesh.voxel_workspace.extent_max = grid.extent_max_exclusive

    if dirty_only:
        target_coords: Set[BrickCoord] = set(grid.dirty_bricks)
    else:
        existing_mesh_coords = {
            key_to_brick_coord(k)
            for k in list(mesh.keys())
            if key_to_brick_coord(k) is not None
        }
        target_coords = set(grid.bricks.keys()) | existing_mesh_coords

    processed: Set[BrickCoord] = set()

    for coord in target_coords:
        if coord is None:
            continue
        key = brick_coord_to_key(coord)
        key_len = key + "_len"
        brick = grid.bricks.get(coord)

        if brick is not None and np.any(brick):
            vals, byte_len = pack_brick(brick)
            mesh[key] = vals
            mesh[key_len] = byte_len
            processed.add(coord)
        else:
            # Prune empty or deleted brick from IDProperties
            if key in mesh:
                del mesh[key]
            if key_len in mesh:
                del mesh[key_len]
            processed.add(coord)

    grid.dirty_bricks.clear()
    return processed


def deserialize_volume(
    mesh: Any,
    grid: Optional[VoxelGrid] = None,
) -> VoxelGrid:
    """Deserialize voxel bricks from Mesh custom IDProperties into a VoxelGrid.
    
    Reads metadata from mesh.voxel_workspace and restores all vox_brick_* arrays.
    """
    if hasattr(mesh, "voxel_workspace"):
        brick_size = int(mesh.voxel_workspace.brick_size)
        extent_min = tuple(mesh.voxel_workspace.extent_min)
        extent_max = tuple(mesh.voxel_workspace.extent_max)
    else:
        brick_size = BRICK_SIZE
        extent_min = (0, 0, 0)
        extent_max = (32, 32, 32)

    if grid is None:
        grid = VoxelGrid(
            extent_min=extent_min,
            extent_max_exclusive=extent_max,
            brick_size=brick_size,
        )
    else:
        grid.extent_min = extent_min
        grid.extent_max_exclusive = extent_max
        grid.brick_size = brick_size
        grid.bricks.clear()
        grid.dirty_bricks.clear()

    # Discover and sort all brick keys for stable deterministic load
    brick_entries: List[Tuple[BrickCoord, str]] = []
    for k in list(mesh.keys()):
        coord = key_to_brick_coord(k)
        if coord is not None:
            brick_entries.append((coord, k))

    brick_entries.sort(key=lambda item: item[0])

    for coord, key in brick_entries:
        key_len = key + "_len"
        vals = mesh[key]
        byte_len = int(mesh[key_len]) if key_len in mesh else (brick_size ** 3)
        brick = unpack_brick(vals, byte_len, brick_size=brick_size)
        if np.any(brick):
            grid.bricks[coord] = brick

    grid.dirty_bricks.clear()
    return grid


def commit_volume_state(
    target: Any,
    grid: Optional[VoxelGrid] = None,
    undo_message: str = "Voxel Edit",
    push_undo: bool = True,
    sync_mesh: bool = False,
    mesh_sync_callback: Optional[Any] = None,
) -> bool:
    """Commit volume changes to Blender Mesh IDProperties and push Blender undo.
    
    Exposes a clean interface for committed strokes (Task 11) and structures a hook
    for mesh synchronization (Task 8).
    """
    if target is None:
        return False

    mesh = getattr(target, "data", target)
    if mesh is None or not hasattr(mesh, "voxel_workspace"):
        return False

    vol_uuid = mesh.voxel_workspace.uuid
    if grid is None:
        from .runtime import get_volume
        entry = get_volume(vol_uuid) if vol_uuid else None
        if entry is not None:
            grid = entry.grid
        else:
            grid = deserialize_volume(mesh)

    dirty_coords = set(grid.dirty_bricks)
    serialize_volume(mesh, grid, dirty_only=True)

    if sync_mesh:
        if mesh_sync_callback is not None:
            mesh_sync_callback(mesh, grid)
        else:
            from .mesh_sync import sync_volume_mesh
            sync_volume_mesh(mesh, grid=grid, dirty_only=True, dirty_bricks=dirty_coords)

    if push_undo and bpy is not None and hasattr(bpy, "ops") and hasattr(bpy.ops, "ed"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass

    return True


def init_volume_storage(
    target: Any,
    grid: Optional[VoxelGrid] = None,
    push_undo: bool = True,
    undo_message: str = "Create Voxel Volume",
) -> str:
    """Initialize IDProperty storage for a newly created volume and push undo step."""
    mesh = getattr(target, "data", target)
    if mesh is None or not hasattr(mesh, "voxel_workspace"):
        raise ValueError("Target must be a Mesh or Object with voxel_workspace")

    if grid is None:
        brick_size = int(mesh.voxel_workspace.brick_size)
        extent_min = tuple(mesh.voxel_workspace.extent_min)
        extent_max = tuple(mesh.voxel_workspace.extent_max)
        grid = VoxelGrid(extent_min=extent_min, extent_max_exclusive=extent_max, brick_size=brick_size)

    serialize_volume(mesh, grid, dirty_only=False)

    if push_undo and bpy is not None and hasattr(bpy, "ops") and hasattr(bpy.ops, "ed"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass

    return mesh.voxel_workspace.uuid
