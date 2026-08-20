from typing import Dict, Set, Tuple, Optional
import numpy as np

from ..constants import BRICK_SIZE, EMPTY, BrickCoord, VoxelCoord
from .coords import split_coord


class VoxelGrid:
    def __init__(
        self,
        extent_min: VoxelCoord = (0, 0, 0),
        extent_max_exclusive: VoxelCoord = (32, 32, 32),
        brick_size: int = BRICK_SIZE,
    ) -> None:
        self.extent_min: VoxelCoord = extent_min
        self.extent_max_exclusive: VoxelCoord = extent_max_exclusive
        self.brick_size: int = brick_size
        self.bricks: Dict[BrickCoord, np.ndarray] = {}
        self.dirty_bricks: Set[BrickCoord] = set()

    def in_extent(self, coord: VoxelCoord) -> bool:
        """Check if a voxel coordinate is within the volume extents."""
        return (
            self.extent_min[0] <= coord[0] < self.extent_max_exclusive[0]
            and self.extent_min[1] <= coord[1] < self.extent_max_exclusive[1]
            and self.extent_min[2] <= coord[2] < self.extent_max_exclusive[2]
        )

    def get(self, coord: VoxelCoord) -> int:
        """Get the palette index at a coordinate, returning 0 (EMPTY) for unallocated or out-of-bounds."""
        if not self.in_extent(coord):
            return EMPTY
        brick_coord, local_coord = split_coord(coord, self.brick_size)
        brick = self.bricks.get(brick_coord)
        if brick is None:
            return EMPTY
        return int(brick[local_coord[0], local_coord[1], local_coord[2]])

    def set(self, coord: VoxelCoord, palette_index: int) -> None:
        """Set the palette index at a coordinate."""
        if not (0 <= palette_index <= 255):
            raise ValueError(f"Palette index {palette_index} outside valid range 0..255")
        if not self.in_extent(coord):
            raise ValueError(f"Coordinate {coord} outside extent [{self.extent_min}, {self.extent_max_exclusive})")

        brick_coord, local_coord = split_coord(coord, self.brick_size)
        brick = self.bricks.get(brick_coord)

        if brick is None:
            if palette_index == EMPTY:
                return
            brick = np.zeros((self.brick_size, self.brick_size, self.brick_size), dtype=np.uint8)
            self.bricks[brick_coord] = brick

        old_val = brick[local_coord[0], local_coord[1], local_coord[2]]
        if old_val != palette_index:
            brick[local_coord[0], local_coord[1], local_coord[2]] = palette_index
            self.dirty_bricks.add(brick_coord)

            # Prune empty brick if set to EMPTY and no occupied cells remain
            if palette_index == EMPTY and not np.any(brick):
                del self.bricks[brick_coord]

    def read_apron(self, brick_coord: BrickCoord) -> np.ndarray:
        """Return a (34, 34, 34) array for the given brick, including 1-voxel padding from neighbors.
        
        Zero is returned for empty voxels and voxels outside the volume extent.
        """
        s = self.brick_size
        apron = np.zeros((s + 2, s + 2, s + 2), dtype=np.uint8)
        bx, by, bz = brick_coord

        slice_map = {
            -1: (slice(s - 1, s), slice(0, 1)),
            0: (slice(0, s), slice(1, s + 1)),
            1: (slice(0, 1), slice(s + 1, s + 2)),
        }

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    n_coord = (bx + dx, by + dy, bz + dz)
                    n_brick = self.bricks.get(n_coord)
                    if n_brick is None:
                        continue
                    b_sx, a_sx = slice_map[dx]
                    b_sy, a_sy = slice_map[dy]
                    b_sz, a_sz = slice_map[dz]
                    apron[a_sx, a_sy, a_sz] = n_brick[b_sx, b_sy, b_sz]

        # Mask out any cells outside the volume extent
        # Apron index a corresponds to global coord = b * s + (a - 1)
        gx_start = bx * s - 1
        gy_start = by * s - 1
        gz_start = bz * s - 1

        if gx_start < self.extent_min[0]:
            apron[0, :, :] = 0
        if gx_start + s + 1 >= self.extent_max_exclusive[0]:
            apron[s + 1, :, :] = 0

        if gy_start < self.extent_min[1]:
            apron[:, 0, :] = 0
        if gy_start + s + 1 >= self.extent_max_exclusive[1]:
            apron[:, s + 1, :] = 0

        if gz_start < self.extent_min[2]:
            apron[:, :, 0] = 0
        if gz_start + s + 1 >= self.extent_max_exclusive[2]:
            apron[:, :, s + 1] = 0

        return apron

    def sorted_brick_coords(self) -> list[BrickCoord]:
        """Return brick coordinates sorted in stable (Z, Y, X) or standard tuple order."""
        return sorted(self.bricks.keys())


