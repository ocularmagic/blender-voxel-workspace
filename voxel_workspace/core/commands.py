from dataclasses import dataclass
from typing import Dict, List, Set

from ..constants import BRICK_SIZE, BrickCoord, VoxelCoord
from .coords import split_coord
from .grid import VoxelGrid


@dataclass
class CellDelta:
    coord: VoxelCoord
    before: int
    after: int


class VoxelStroke:
    def __init__(self, brick_size: int = BRICK_SIZE) -> None:
        self.brick_size: int = brick_size
        self._deltas: Dict[VoxelCoord, CellDelta] = {}

    @property
    def deltas(self) -> List[CellDelta]:
        return list(self._deltas.values())

    def record(self, grid: VoxelGrid, coord: VoxelCoord, new_value: int) -> None:
        """Record a touch on a voxel coordinate.
        
        Captures the first 'before' value from the grid on first touch,
        and updates 'after' on subsequent touches.
        """
        if coord in self._deltas:
            self._deltas[coord].after = new_value
        else:
            before_val = grid.get(coord)
            self._deltas[coord] = CellDelta(coord=coord, before=before_val, after=new_value)

    touch = record

    def apply(self, grid: VoxelGrid) -> None:
        """Apply the stroke's final 'after' values to the grid."""
        for delta in self._deltas.values():
            grid.set(delta.coord, delta.after)

    def revert(self, grid: VoxelGrid) -> None:
        """Revert the grid back to the stroke's original 'before' values."""
        for delta in self._deltas.values():
            grid.set(delta.coord, delta.before)

    def changed_bricks(self) -> Set[BrickCoord]:
        """Report impacted bricks plus face-neighbor bricks when a changed cell lies on a brick boundary."""
        result: Set[BrickCoord] = set()
        for delta in self._deltas.values():
            if delta.before == delta.after:
                continue
            brick_coord, local_coord = split_coord(delta.coord, self.brick_size)
            result.add(brick_coord)
            bx, by, bz = brick_coord
            lx, ly, lz = local_coord

            if lx == 0:
                result.add((bx - 1, by, bz))
            elif lx == self.brick_size - 1:
                result.add((bx + 1, by, bz))

            if ly == 0:
                result.add((bx, by - 1, bz))
            elif ly == self.brick_size - 1:
                result.add((bx, by + 1, bz))

            if lz == 0:
                result.add((bx, by, bz - 1))
            elif lz == self.brick_size - 1:
                result.add((bx, by, bz + 1))

        return result


