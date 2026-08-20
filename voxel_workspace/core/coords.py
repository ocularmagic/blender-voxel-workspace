from typing import Tuple
from voxel_workspace.constants import BRICK_SIZE, BrickCoord, VoxelCoord


def split_coord(coord: VoxelCoord, brick_size: int = BRICK_SIZE) -> Tuple[BrickCoord, VoxelCoord]:
    """Split a global voxel coordinate into (brick_coord, local_coord).
    
    Uses floor division and modulo so negative coordinates work correctly.
    """
    bx, lx = divmod(coord[0], brick_size)
    by, ly = divmod(coord[1], brick_size)
    bz, lz = divmod(coord[2], brick_size)
    return (bx, by, bz), (lx, ly, lz)


def join_coord(brick_coord: BrickCoord, local_coord: VoxelCoord, brick_size: int = BRICK_SIZE) -> VoxelCoord:
    """Combine a brick coordinate and local offset into a global voxel coordinate."""
    return (
        brick_coord[0] * brick_size + local_coord[0],
        brick_coord[1] * brick_size + local_coord[1],
        brick_coord[2] * brick_size + local_coord[2],
    )
