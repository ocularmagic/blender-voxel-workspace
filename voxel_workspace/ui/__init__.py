"""UI panels and workspace integration."""
from .panels import (
    VOXEL_PT_palette_panel,
    VOXEL_PT_main_panel,
    PANEL_CLASSES,
    draw_typed_palette,
)
from .shelf import (
    VOXEL_AST_workspace,
    SHELF_CLASSES,
    ensure_tool_assets,
)
from .palette_icons import register_palette_icons, unregister_palette_icons
from .gizmos import GIZMO_CLASSES
from .workspace import (
    register_voxel_workspace,
    unregister_voxel_workspace,
)

__all__ = [
    "VOXEL_PT_palette_panel",
    "VOXEL_PT_main_panel",
    "VOXEL_AST_workspace",
    "PANEL_CLASSES",
    "SHELF_CLASSES",
    "draw_typed_palette",
    "ensure_tool_assets",
    "register_palette_icons",
    "unregister_palette_icons",
    "register_voxel_workspace",
    "unregister_voxel_workspace",
    "GIZMO_CLASSES",
]
