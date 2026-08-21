"""Voxel operators."""
from typing import List, Type
from .create_volume import (
    VOXEL_OT_create_volume,
)
from .edit_session import (
    VOXEL_OT_start_place,
    VOXEL_OT_start_erase,
    VOXEL_OT_stop_editing,
    EDIT_SESSION_OPERATOR_CLASSES,
)
from .brush import (
    VOXEL_OT_brush,
    BRUSH_OPERATOR_CLASSES,
)
from .palette import (
    VOXEL_OT_select_palette_color,
    VOXEL_OT_add_palette_color,
    VOXEL_OT_duplicate_palette_color,
    VOXEL_OT_remove_palette_color,
    VOXEL_OT_eyedropper,
    VOXEL_OT_compact_palette,
    VOXEL_OT_save_palette_preset,
    VOXEL_OT_load_palette_preset,
    PALETTE_OPERATOR_CLASSES,
    remap_volume_palette_indices,
    get_used_palette_counts,
)

OPERATOR_CLASSES: List[Type] = [
    VOXEL_OT_create_volume,
    *EDIT_SESSION_OPERATOR_CLASSES,
    *BRUSH_OPERATOR_CLASSES,
    *PALETTE_OPERATOR_CLASSES,
]

__all__ = [
    "VOXEL_OT_create_volume",
    "VOXEL_OT_start_place",
    "VOXEL_OT_start_erase",
    "VOXEL_OT_stop_editing",
    "VOXEL_OT_brush",
    "VOXEL_OT_select_palette_color",
    "VOXEL_OT_add_palette_color",
    "VOXEL_OT_duplicate_palette_color",
    "VOXEL_OT_remove_palette_color",
    "VOXEL_OT_eyedropper",
    "VOXEL_OT_compact_palette",
    "VOXEL_OT_save_palette_preset",
    "VOXEL_OT_load_palette_preset",
    "remap_volume_palette_indices",
    "get_used_palette_counts",
    "OPERATOR_CLASSES",
]
