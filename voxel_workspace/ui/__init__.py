"""UI panels and workspace integration."""
from typing import List, Type
from .panels import (
    VOXEL_PT_palette_panel,
    VOXEL_PT_main_panel,

    PANEL_CLASSES,
    draw_typed_palette,
    draw_voxel_tool_header,
)
from .palette_icons import register_palette_icons, unregister_palette_icons
from .gizmos import VOXEL_GGT_workspace_tools, GIZMO_CLASSES
from .workspace import (
    register_voxel_workspace,
    unregister_voxel_workspace,
)

__all__ = [
    "VOXEL_PT_palette_panel",
    "VOXEL_PT_main_panel",

    "PANEL_CLASSES",
    "draw_typed_palette",
    "draw_voxel_tool_header",
    "register_palette_icons",
    "unregister_palette_icons",
    "register_voxel_workspace",
    "unregister_voxel_workspace",
    "VOXEL_GGT_workspace_tools",
    "GIZMO_CLASSES",
]
