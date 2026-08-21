"""Material domain filtering for voxel meshing (core and apron)."""
from typing import Collection, Optional, Tuple
import numpy as np


def filter_brick_and_apron(
    core: np.ndarray,
    apron: np.ndarray,
    exclude_indices: Optional[Collection[int]] = None,
    only_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter voxel indices in core and apron arrays for domain-specific meshing.
    
    Parameters:
        core: (S, S, S) uint8 core brick array.
        apron: (S+2, S+2, S+2) uint8 1-voxel padded neighborhood array.
        exclude_indices: Indices to mask to zero (e.g. VOLUME indices during surface meshing).
        only_index: Single index to isolate (all other values masked to zero during proxy meshing).
        
    Returns:
        (filtered_core, filtered_apron)
    """
    if exclude_indices is not None and only_index is not None:
        raise ValueError("exclude_indices and only_index are mutually exclusive")

    if exclude_indices is None and only_index is None:
        # No filtering required; return as-is
        return core, apron

    filtered_core = np.copy(core)
    filtered_apron = np.copy(apron)

    if only_index is not None:
        # Proxy mode: keep only voxels equal to only_index
        filtered_core[core != only_index] = 0
        filtered_apron[apron != only_index] = 0
    elif exclude_indices:
        # Surface mode: mask out excluded indices
        for idx in exclude_indices:
            filtered_core[core == idx] = 0
            filtered_apron[apron == idx] = 0

    return filtered_core, filtered_apron
