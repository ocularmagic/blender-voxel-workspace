"""UI panels and workspace integration."""
from typing import List, Type
from .panels import (
    VOXEL_PT_main_panel,
    PANEL_CLASSES,
)
from .palette_icons import register_palette_icons, unregister_palette_icons

__all__ = [
    "VOXEL_PT_main_panel",
    "PANEL_CLASSES",
    "register_palette_icons",
    "unregister_palette_icons",
]
