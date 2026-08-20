"""Voxel operators."""
from typing import List, Type
from voxel_workspace.operators.create_volume import (
    VOXEL_OT_create_volume,
)
from voxel_workspace.operators.edit_session import (
    VOXEL_OT_start_place,
    VOXEL_OT_start_erase,
    VOXEL_OT_stop_editing,
    EDIT_SESSION_OPERATOR_CLASSES,
)
from voxel_workspace.operators.brush import (
    VOXEL_OT_brush,
    BRUSH_OPERATOR_CLASSES,
)

OPERATOR_CLASSES: List[Type] = [
    VOXEL_OT_create_volume,
    *EDIT_SESSION_OPERATOR_CLASSES,
    *BRUSH_OPERATOR_CLASSES,
]

__all__ = [
    "VOXEL_OT_create_volume",
    "VOXEL_OT_start_place",
    "VOXEL_OT_start_erase",
    "VOXEL_OT_stop_editing",
    "VOXEL_OT_brush",
    "OPERATOR_CLASSES",
]
