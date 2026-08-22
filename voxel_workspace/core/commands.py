from dataclasses import dataclass
from typing import Dict, List, Set, Union

from ..constants import BRICK_SIZE, BrickCoord, VoxelCoord
from .coords import split_coord
from .grid import VoxelGrid
from .tagged_grid import TaggedVoxelGrid, VoxelCell, VoxelDomain, CELL_EMPTY


def apply_brush_value(
    grid: TaggedVoxelGrid,
    coord: VoxelCoord,
    mode: str,
    palette_index: int,
) -> VoxelCell:
    """Apply one explicit tagged brush mode and return the canonical new cell."""
    normalized = str(mode).upper()
    if normalized == "ADD_SURFACE":
        cell = VoxelCell(VoxelDomain.SURFACE, palette_index)
    elif normalized == "ADD_VOLUME":
        cell = VoxelCell(VoxelDomain.VOLUME, palette_index)
    elif normalized == "ERASE":
        cell = CELL_EMPTY
    else:
        raise ValueError(f"Unknown brush mode: {mode}")
    grid.set_cell(coord, cell.domain, cell.index)
    return cell


@dataclass
class CellDelta:
    coord: VoxelCoord
    before: Union[int, VoxelCell]
    after: Union[int, VoxelCell]


class VoxelStroke:
    def __init__(self, brick_size: int = BRICK_SIZE) -> None:
        self.brick_size: int = brick_size
        self._deltas: Dict[VoxelCoord, CellDelta] = {}

    @property
    def deltas(self) -> List[CellDelta]:
        return list(self._deltas.values())

    def record(self, grid: Union[VoxelGrid, TaggedVoxelGrid], coord: VoxelCoord, new_value: Union[int, VoxelCell]) -> None:
        """Record a touch on a voxel coordinate.
        
        Captures the first 'before' value from the grid on first touch,
        and updates 'after' on subsequent touches.
        """
        if coord in self._deltas:
            self._deltas[coord].after = new_value
        else:
            if isinstance(grid, TaggedVoxelGrid):
                before_val = grid.get_cell(coord)
            else:
                before_val = grid.get(coord)
            self._deltas[coord] = CellDelta(coord=coord, before=before_val, after=new_value)

    touch = record

    def apply(self, grid: Union[VoxelGrid, TaggedVoxelGrid]) -> None:
        """Apply the stroke's final 'after' values to the grid."""
        for delta in self._deltas.values():
            if isinstance(grid, TaggedVoxelGrid):
                if isinstance(delta.after, VoxelCell):
                    grid.set_cell(delta.coord, delta.after.domain, delta.after.index)
                else:
                    grid.set_surface(delta.coord, int(delta.after))
            else:
                if isinstance(delta.after, VoxelCell):
                    grid.set(delta.coord, delta.after.index)
                else:
                    grid.set(delta.coord, int(delta.after))

    def revert(self, grid: Union[VoxelGrid, TaggedVoxelGrid]) -> None:
        """Revert the grid back to the stroke's original 'before' values."""
        for delta in self._deltas.values():
            if isinstance(grid, TaggedVoxelGrid):
                if isinstance(delta.before, VoxelCell):
                    grid.set_cell(delta.coord, delta.before.domain, delta.before.index)
                else:
                    grid.set_surface(delta.coord, int(delta.before))
            else:
                if isinstance(delta.before, VoxelCell):
                    grid.set(delta.coord, delta.before.index)
                else:
                    grid.set(delta.coord, int(delta.before))

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
