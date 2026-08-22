"""UI panels and workspace integration."""
from typing import List, Type
from .panels import (
    VOXEL_PT_palette_panel,
    VOXEL_PT_main_panel,
    VOXEL_PT_workspace_settings,
    PANEL_CLASSES,
    draw_typed_palette,
    draw_voxel_tool_header,
)
from .palette_icons import register_palette_icons, unregister_palette_icons
from .workspace import (
    register_voxel_workspace,
    schedule_voxel_workspace_registration,
    unregister_voxel_workspace,
)

__all__ = [
    "VOXEL_PT_palette_panel",
    "VOXEL_PT_main_panel",
    "VOXEL_PT_workspace_settings",
    "PANEL_CLASSES",
    "draw_typed_palette",
    "draw_voxel_tool_header",
    "register_palette_icons",
    "unregister_palette_icons",
    "register_voxel_workspace",
    "schedule_voxel_workspace_registration",
    "unregister_voxel_workspace",
]
