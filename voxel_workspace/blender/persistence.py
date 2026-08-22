"""Voxel brick persistence, IDProperty array serialization, and Blender memfile undo bridge.

Authoritative schema 3:
- Metadata is stored on the Mesh datablock via PointerProperty `voxel_workspace` (VoxelMeshProperties):
    - uuid (str)
    - is_voxel_mesh (bool)
    - brick_size (int, default 32)
    - extent_min (int[3])
    - extent_max (int[3])
    - voxel_size (float)
    - schema_version (int, default 3)
- Voxel bricks are stored as deterministic top-level IDProperty arrays on the Mesh datablock:
    - 'vox_brick_<bx>_<by>_<bz>': IDPropertyArray of signed 32-bit ints (uint8 indices packed into i32)
    - 'vox_brick_<bx>_<by>_<bz>_len': integer original index byte length (brick_size^3)
    - 'vox_domain_<bx>_<by>_<bz>': IDPropertyArray of signed 32-bit ints (1-bit-per-cell packed domain mask)
    - 'vox_domain_<bx>_<by>_<bz>_len': integer original domain mask byte length (brick_size^3 // 8)
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
from ..core.tagged_grid import TaggedVoxelGrid, VoxelDomain, TaggedBrick


SCHEMA_VERSION = 3
BRICK_KEY_PATTERN = re.compile(r"^vox_brick_(-?\d+)_(-?\d+)_(-?\d+)$")
DOMAIN_KEY_PATTERN = re.compile(r"^vox_domain_(-?\d+)_(-?\d+)_(-?\d+)$")


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


def pack_domain_mask(domains: np.ndarray, indices: np.ndarray) -> bytes:
    """Pack domain flags into a 1-bit-per-cell mask (4096 bytes for 32^3).
    
    Convention:
    - 0 = SURFACE (VoxelDomain.SURFACE)
    - 1 = VOLUME (VoxelDomain.VOLUME)
    Index 0 cells ignore the bit and canonicalize to EMPTY on decode.
    """
    flat_domains = np.ascontiguousarray(domains, dtype=np.uint8).ravel()
    flat_indices = np.ascontiguousarray(indices, dtype=np.uint8).ravel()
    
    # bit 0 for Surface, 1 for Volume
    bits = (flat_domains == int(VoxelDomain.VOLUME)).astype(np.uint8)
    packed = np.packbits(bits, bitorder='little').tobytes()
    return packed


def unpack_domain_mask(packed_bytes: bytes, indices: np.ndarray) -> np.ndarray:
    """Unpack 1-bit-per-cell mask back to domain array (0=EMPTY, 1=SURFACE, 2=VOLUME)."""
    flat_indices = np.ascontiguousarray(indices, dtype=np.uint8).ravel()
    total_cells = len(flat_indices)
    expected_bytes = (total_cells + 7) // 8
    if len(packed_bytes) != expected_bytes:
        raise ValueError(
            f"Domain mask byte length {len(packed_bytes)} does not match expected {expected_bytes} for {total_cells} cells"
        )
    unpacked_bits = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8), bitorder='little')[:total_cells]
    
    # 0 -> SURFACE (1), 1 -> VOLUME (2)
    domains = np.where(unpacked_bits == 1, int(VoxelDomain.VOLUME), int(VoxelDomain.SURFACE)).astype(np.uint8)
    # Where index is 0, domain is canonicalized to EMPTY (0)
    domains[flat_indices == 0] = int(VoxelDomain.EMPTY)
    return domains.reshape(indices.shape)


def brick_coord_to_key(coord: BrickCoord) -> str:
    """Convert a brick coordinate tuple (bx, by, bz) to custom property index key."""
    return f"vox_brick_{coord[0]}_{coord[1]}_{coord[2]}"


def brick_coord_to_domain_key(coord: BrickCoord) -> str:
    """Convert a brick coordinate tuple (bx, by, bz) to custom property domain mask key."""
    return f"vox_domain_{coord[0]}_{coord[1]}_{coord[2]}"


def key_to_brick_coord(key: str) -> Optional[BrickCoord]:
    """Parse custom property key to brick coordinate tuple, or None if not a brick key."""
    m = BRICK_KEY_PATTERN.match(key)
    if not m:
        m = DOMAIN_KEY_PATTERN.match(key)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def serialize_volume(
    mesh: Any,
    grid: Union[TaggedVoxelGrid, VoxelGrid],
    dirty_only: bool = True,
) -> Set[BrickCoord]:
    """Serialize voxel bricks and domain masks to Mesh custom IDProperties.
    
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
        key_idx = brick_coord_to_key(coord)
        key_idx_len = key_idx + "_len"
        key_dom = brick_coord_to_domain_key(coord)
        key_dom_len = key_dom + "_len"

        brick_obj = grid.bricks.get(coord)

        if brick_obj is not None:
            if isinstance(brick_obj, TaggedBrick):
                indices = brick_obj.indices
                domains = brick_obj.domains
            elif isinstance(brick_obj, np.ndarray):
                indices = brick_obj
                domains = np.where(indices > 0, int(VoxelDomain.SURFACE), int(VoxelDomain.EMPTY)).astype(np.uint8)
            else:
                indices = getattr(brick_obj, "indices", None)
                domains = getattr(brick_obj, "domains", None)

            if indices is not None and np.any(indices):
                # 1. Pack index channel
                idx_vals, idx_len = pack_brick(indices)
                # 2. Pack domain channel
                dom_bytes = pack_domain_mask(domains, indices)
                dom_vals, dom_len = pack_bytes_to_i32(dom_bytes)

                # Write IDProperties
                mesh[key_idx] = idx_vals
                mesh[key_idx_len] = idx_len
                mesh[key_dom] = dom_vals
                mesh[key_dom_len] = dom_len
                processed.add(coord)
            else:
                # Prune empty or zero brick
                for k in (key_idx, key_idx_len, key_dom, key_dom_len):
                    if k in mesh:
                        del mesh[k]
                processed.add(coord)
        else:
            # Prune non-existent brick
            for k in (key_idx, key_idx_len, key_dom, key_dom_len):
                if k in mesh:
                    del mesh[k]
            processed.add(coord)

    grid.dirty_bricks.clear()
    return processed


def deserialize_volume(
    mesh: Any,
    grid: Optional[Union[TaggedVoxelGrid, VoxelGrid]] = None,
) -> Union[TaggedVoxelGrid, VoxelGrid]:
    """Deserialize voxel bricks from Mesh custom IDProperties into a TaggedVoxelGrid (or VoxelGrid).
    
    Reads metadata from mesh.voxel_workspace and restores both index and domain channels.
    Supports Schema 3 (tagged index + domain channels) and handles Schema 1 & 2 legacy data.
    """
    if hasattr(mesh, "voxel_workspace"):
        props = mesh.voxel_workspace
        schema_v = int(getattr(props, "schema_version", 3))
        brick_size = int(props.brick_size)
        extent_min = tuple(props.extent_min)
        extent_max = tuple(props.extent_max)
    else:
        schema_v = 3
        brick_size = BRICK_SIZE
        extent_min = (0, 0, 0)
        extent_max = (32, 32, 32)

    return_scalar = isinstance(grid, VoxelGrid)

    if grid is None:
        target_grid = TaggedVoxelGrid(
            extent_min=extent_min,
            extent_max_exclusive=extent_max,
            brick_size=brick_size,
        )
    else:
        target_grid = grid
        target_grid.extent_min = extent_min
        target_grid.extent_max_exclusive = extent_max
        target_grid.brick_size = brick_size
        target_grid.bricks.clear()
        target_grid.dirty_bricks.clear()

    # Discover and sort all brick index keys
    brick_entries: List[Tuple[BrickCoord, str]] = []
    for k in list(mesh.keys()):
        if k.startswith("vox_brick_") and not k.endswith("_len"):
            coord = key_to_brick_coord(k)
            if coord is not None:
                brick_entries.append((coord, k))

    brick_entries.sort(key=lambda item: item[0])

    for coord, key in brick_entries:
        key_len = key + "_len"
        vals = mesh[key]
        byte_len = int(mesh[key_len]) if key_len in mesh else (brick_size ** 3)
        indices = unpack_brick(vals, byte_len, brick_size=brick_size)

        if not np.any(indices):
            continue

        dom_key = brick_coord_to_domain_key(coord)
        dom_key_len = dom_key + "_len"

        if dom_key in mesh:
            dom_vals = mesh[dom_key]
            expected_dom_bytes = (brick_size ** 3 + 7) // 8
            dom_byte_len = int(mesh[dom_key_len]) if dom_key_len in mesh else expected_dom_bytes
            dom_raw = unpack_i32_to_bytes(dom_vals, dom_byte_len)
            domains = unpack_domain_mask(dom_raw, indices)
        else:
            # Schema 1/2 fallback: all occupied cells are SURFACE
            if schema_v >= 3:
                raise ValueError(
                    f"Corrupt schema-3 volume: brick {coord} is occupied but missing domain mask '{dom_key}'"
                )
            domains = np.where(indices > 0, int(VoxelDomain.SURFACE), int(VoxelDomain.EMPTY)).astype(np.uint8)

        if isinstance(target_grid, TaggedVoxelGrid):
            tagged_brick = TaggedBrick(brick_size)
            tagged_brick.indices = indices
            tagged_brick.domains = domains
            target_grid.bricks[coord] = tagged_brick
        elif isinstance(target_grid, VoxelGrid):
            target_grid.bricks[coord] = indices

    target_grid.dirty_bricks.clear()
    return target_grid


def commit_volume_state(
    target: Any,
    grid: Optional[Union[TaggedVoxelGrid, VoxelGrid]] = None,
    undo_message: str = "Voxel Edit",
    push_undo: bool = True,
    sync_mesh: bool = False,
    mesh_sync_callback: Optional[Any] = None,
) -> bool:
    """Commit volume changes to Blender Mesh IDProperties and push Blender undo."""
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
    grid: Optional[Union[TaggedVoxelGrid, VoxelGrid]] = None,
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
        grid = TaggedVoxelGrid(extent_min=extent_min, extent_max_exclusive=extent_max, brick_size=brick_size)

    serialize_volume(mesh, grid, dirty_only=False)

    if push_undo and bpy is not None and hasattr(bpy, "ops") and hasattr(bpy.ops, "ed"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass

    return mesh.voxel_workspace.uuid
