from __future__ import annotations

import enum
from typing import Dict, Iterator, List, Optional, Set, Tuple
import numpy as np

from ..constants import BRICK_SIZE, BrickCoord, VoxelCoord
from .coords import split_coord


class VoxelDomain(enum.IntEnum):
    EMPTY = 0
    SURFACE = 1
    VOLUME = 2


class VoxelCell:
    __slots__ = ("domain", "index")

    def __init__(self, domain: VoxelDomain | int, index: int) -> None:
        d = VoxelDomain(domain)
        idx = int(index)
        if not (0 <= idx <= 255):
            raise ValueError(f"Palette index {idx} outside valid range 0..255")
        if idx == 0 and d != VoxelDomain.EMPTY:
            raise ValueError(f"Index 0 requires domain EMPTY, got {d.name}")
        if idx > 0 and d == VoxelDomain.EMPTY:
            raise ValueError(f"Non-zero index {idx} requires non-EMPTY domain")
        self.domain: VoxelDomain = d
        self.index: int = idx

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VoxelCell):
            return self.domain == other.domain and self.index == other.index
        if isinstance(other, tuple) and len(other) == 2:
            return self.domain == other[0] and self.index == other[1]
        return False

    def __hash__(self) -> int:
        return hash((self.domain, self.index))

    def __repr__(self) -> str:
        return f"VoxelCell({self.domain.name}, {self.index})"

    def __iter__(self):
        yield self.domain
        yield self.index


CELL_EMPTY = VoxelCell(VoxelDomain.EMPTY, 0)


class TaggedBrick:
    __slots__ = ("indices", "domains", "brick_size")

    def __init__(self, brick_size: int = BRICK_SIZE) -> None:
        self.brick_size: int = brick_size
        self.indices: np.ndarray = np.zeros((brick_size, brick_size, brick_size), dtype=np.uint8)
        self.domains: np.ndarray = np.zeros((brick_size, brick_size, brick_size), dtype=np.uint8)

    def is_empty(self) -> bool:
        return not np.any(self.indices)

    def copy(self) -> TaggedBrick:
        b = TaggedBrick(self.brick_size)
        b.indices = self.indices.copy()
        b.domains = self.domains.copy()
        return b

    def __getitem__(self, item):
        return self.indices[item]

    def __setitem__(self, item, value):
        self.indices[item] = value
        if isinstance(value, np.ndarray):
            self.domains[item] = np.where(value > 0, int(VoxelDomain.SURFACE), int(VoxelDomain.EMPTY))
        elif int(value) == 0:
            self.domains[item] = int(VoxelDomain.EMPTY)
        else:
            self.domains[item] = int(VoxelDomain.SURFACE)

    @property
    def shape(self):
        return self.indices.shape

    @property
    def dtype(self):
        return self.indices.dtype

    def __array__(self, dtype=None):
        return np.asarray(self.indices, dtype=dtype)


class TaggedVoxelGrid:
    def __init__(
        self,
        extent_min: VoxelCoord = (0, 0, 0),
        extent_max_exclusive: VoxelCoord = (32, 32, 32),
        brick_size: int = BRICK_SIZE,
    ) -> None:
        self.extent_min: VoxelCoord = extent_min
        self.extent_max_exclusive: VoxelCoord = extent_max_exclusive
        self.brick_size: int = brick_size
        self.bricks: Dict[BrickCoord, TaggedBrick] = {}
        self.dirty_bricks: Set[BrickCoord] = set()

    def in_extent(self, coord: VoxelCoord) -> bool:
        return (
            self.extent_min[0] <= coord[0] < self.extent_max_exclusive[0]
            and self.extent_min[1] <= coord[1] < self.extent_max_exclusive[1]
            and self.extent_min[2] <= coord[2] < self.extent_max_exclusive[2]
        )

    def get_cell(self, coord: VoxelCoord) -> VoxelCell:
        if not self.in_extent(coord):
            return CELL_EMPTY
        brick_coord, local_coord = split_coord(coord, self.brick_size)
        brick = self.bricks.get(brick_coord)
        if brick is None:
            return CELL_EMPTY
        lx, ly, lz = local_coord
        idx = int(brick.indices[lx, ly, lz])
        dom = int(brick.domains[lx, ly, lz])
        if idx == 0:
            return CELL_EMPTY
        return VoxelCell(dom, idx)

    def get(self, coord: VoxelCoord) -> int:
        """Compatibility getter returning palette index, or 0 if empty / out-of-bounds."""
        return self.get_index(coord)

    def set(self, coord: VoxelCoord, palette_index: int) -> None:
        """Compatibility setter: sets Surface if palette_index > 0, else erases."""
        idx = int(palette_index)
        if idx == 0:
            self.erase(coord)
        else:
            self.set_surface(coord, idx)

    def get_index(self, coord: VoxelCoord) -> int:
        return self.get_cell(coord).index

    def get_domain(self, coord: VoxelCoord) -> VoxelDomain:
        return self.get_cell(coord).domain

    def set_cell(self, coord: VoxelCoord, domain: VoxelDomain | int, index: int) -> None:
        dom = VoxelDomain(domain)
        idx = int(index)
        if not (0 <= idx <= 255):
            raise ValueError(f"Palette index {idx} outside valid range 0..255")
        if not self.in_extent(coord):
            raise ValueError(f"Coordinate {coord} outside extent [{self.extent_min}, {self.extent_max_exclusive})")
        if idx == 0 and dom != VoxelDomain.EMPTY:
            raise ValueError(f"Index 0 requires domain EMPTY, got {dom.name}")
        if idx > 0 and dom == VoxelDomain.EMPTY:
            raise ValueError(f"Non-zero index {idx} requires non-EMPTY domain")

        brick_coord, local_coord = split_coord(coord, self.brick_size)
        brick = self.bricks.get(brick_coord)

        if brick is None:
            if idx == 0:
                return
            brick = TaggedBrick(self.brick_size)
            self.bricks[brick_coord] = brick

        lx, ly, lz = local_coord
        old_idx = int(brick.indices[lx, ly, lz])
        old_dom = int(brick.domains[lx, ly, lz])

        if old_idx != idx or old_dom != int(dom):
            brick.indices[lx, ly, lz] = idx
            brick.domains[lx, ly, lz] = int(dom) if idx != 0 else 0
            self.dirty_bricks.add(brick_coord)

            if idx == 0 and brick.is_empty():
                del self.bricks[brick_coord]

    def set_surface(self, coord: VoxelCoord, index: int) -> None:
        if not (1 <= index <= 255):
            raise ValueError(f"Surface palette index must be in 1..255, got {index}")
        self.set_cell(coord, VoxelDomain.SURFACE, index)

    def set_volume(self, coord: VoxelCoord, index: int) -> None:
        if not (1 <= index <= 255):
            raise ValueError(f"Volume palette index must be in 1..255, got {index}")
        self.set_cell(coord, VoxelDomain.VOLUME, index)

    def erase(self, coord: VoxelCoord) -> None:
        if not self.in_extent(coord):
            return
        self.set_cell(coord, VoxelDomain.EMPTY, 0)

    def read_apron(self, brick_coord: BrickCoord) -> np.ndarray:
        """Compatibility method for read_index_apron without filters."""
        return self.read_index_apron(brick_coord)

    def read_index_apron(
        self,
        brick_coord: BrickCoord,
        domain_filter: Optional[VoxelDomain | int] = None,
        only_index: Optional[int] = None,
    ) -> np.ndarray:
        s = self.brick_size
        apron = np.zeros((s + 2, s + 2, s + 2), dtype=np.uint8)
        bx, by, bz = brick_coord

        slice_map = {
            -1: (slice(s - 1, s), slice(0, 1)),
            0: (slice(0, s), slice(1, s + 1)),
            1: (slice(0, 1), slice(s + 1, s + 2)),
        }

        target_domain = int(domain_filter) if domain_filter is not None else None

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

                    b_idx = n_brick.indices[b_sx, b_sy, b_sz]
                    b_dom = n_brick.domains[b_sx, b_sy, b_sz]

                    mask = b_idx > 0
                    if target_domain is not None:
                        mask = mask & (b_dom == target_domain)
                    if only_index is not None:
                        mask = mask & (b_idx == only_index)

                    projected = np.where(mask, b_idx, 0).astype(np.uint8)
                    apron[a_sx, a_sy, a_sz] = projected

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

    def iter_used_indices(self, domain: VoxelDomain | int) -> Set[int]:
        d = int(domain)
        used: Set[int] = set()
        for brick in self.bricks.values():
            mask = (brick.domains == d) & (brick.indices > 0)
            if np.any(mask):
                vals = np.unique(brick.indices[mask])
                used.update(int(v) for v in vals)
        return used

    def count_indices(self, domain: VoxelDomain | int) -> Dict[int, int]:
        d = int(domain)
        counts: Dict[int, int] = {}
        for brick in self.bricks.values():
            mask = (brick.domains == d) & (brick.indices > 0)
            if np.any(mask):
                vals, freqs = np.unique(brick.indices[mask], return_counts=True)
                for v, c in zip(vals, freqs):
                    counts[int(v)] = counts.get(int(v), 0) + int(c)
        return counts

    def remap_indices(self, domain: VoxelDomain | int, mapping: Dict[int, int]) -> None:
        d = int(domain)
        for brick_coord, brick in list(self.bricks.items()):
            mask = (brick.domains == d) & (brick.indices > 0)
            if not np.any(mask):
                continue

            changed = False
            coords = np.argwhere(mask)
            for lx, ly, lz in coords:
                old_val = int(brick.indices[lx, ly, lz])
                if old_val in mapping:
                    new_val = int(mapping[old_val])
                    if not (0 <= new_val <= 255):
                        raise ValueError(f"Remap target {new_val} out of range 0..255")
                    if new_val != old_val:
                        brick.indices[lx, ly, lz] = new_val
                        if new_val == 0:
                            brick.domains[lx, ly, lz] = 0
                        changed = True

            if changed:
                self.dirty_bricks.add(brick_coord)
                if brick.is_empty():
                    del self.bricks[brick_coord]

    def validate(self) -> None:
        for b_coord, brick in self.bricks.items():
            if brick.indices.shape != (self.brick_size, self.brick_size, self.brick_size):
                raise ValueError(f"Brick {b_coord} invalid indices shape: {brick.indices.shape}")
            if brick.domains.shape != (self.brick_size, self.brick_size, self.brick_size):
                raise ValueError(f"Brick {b_coord} invalid domains shape: {brick.domains.shape}")

            zero_idx_non_zero_dom = (brick.indices == 0) & (brick.domains != 0)
            if np.any(zero_idx_non_zero_dom):
                raise ValueError(f"Brick {b_coord} has index 0 with non-zero domain")

            non_zero_idx_zero_dom = (brick.indices > 0) & (brick.domains == 0)
            if np.any(non_zero_idx_zero_dom):
                raise ValueError(f"Brick {b_coord} has non-zero index with EMPTY domain")

            invalid_dom = (brick.domains != 0) & (brick.domains != 1) & (brick.domains != 2)
            if np.any(invalid_dom):
                raise ValueError(f"Brick {b_coord} has invalid domain values")

            if brick.is_empty():
                raise ValueError(f"Brick {b_coord} is completely empty and should have been pruned")

    def sorted_brick_coords(self) -> List[BrickCoord]:
        return sorted(self.bricks.keys())
