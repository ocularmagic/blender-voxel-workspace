"""Creation and configuration of the custom Voxel Workspace layout.

Blender does not allow arbitrary Python panels in the TOOLS or ASSET_SHELF
regions.  The workspace therefore uses supported UI surfaces: a VIEW_3D
sidebar flipped to the left for the palette, a tool header flipped to the
bottom for brush buttons, and the existing right Properties editor for volume
settings.
"""
from typing import Any, Optional

try:
    import bpy
    from bpy.types import WorkSpace
except ImportError:
    bpy = None
    WorkSpace = object

WORKSPACE_NAME = "Voxel Workspace"
_WORKSPACE_UUID = "voxel-workspace-layout-v2"
_header_registered = False
_timer_registered = False


def _find_workspace() -> Optional["WorkSpace"]:
    if bpy is None:
        return None
    return bpy.data.workspaces.get(WORKSPACE_NAME)


def _window() -> Any:
    if bpy is None:
        return None
    window = getattr(bpy.context, "window", None)
    if window is not None:
        return window
    manager = getattr(bpy.context, "window_manager", None)
    return manager.windows[0] if manager is not None and manager.windows else None


def _flip_region(window: Any, area: Any, region_type: str, alignment: str) -> bool:
    region = next((candidate for candidate in area.regions if candidate.type == region_type), None)
    if region is None:
        return False
    if region.alignment == alignment:
        return True
    screen = window.screen
    space = area.spaces.active
    try:
        with bpy.context.temp_override(
            window=window,
            screen=screen,
            area=area,
            region=region,
            space_data=space,
        ):
            result = bpy.ops.screen.region_flip()
        return result == {'FINISHED'} and region.alignment == alignment
    except Exception:
        return False


def _configure_workspace(workspace: "WorkSpace", window: Any) -> bool:
    """Configure supported Blender regions for the Voxel Workspace."""
    if window.workspace != workspace or all(window.screen != screen for screen in workspace.screens):
        return False
    screen = window.screen
    try:
        view_areas = [area for area in screen.areas if area.type == 'VIEW_3D']
        if not view_areas:
            return False
        view = max(view_areas, key=lambda area: area.width * area.height)
        space = view.spaces.active
        space.show_region_ui = True
        space.show_region_toolbar = False
        space.show_region_tool_header = True
        if hasattr(space, "show_region_asset_shelf"):
            try:
                space.show_region_asset_shelf = False
            except Exception:
                pass
        palette_left = _flip_region(window, view, 'UI', 'LEFT')
        tools_bottom = _flip_region(window, view, 'TOOL_HEADER', 'BOTTOM')
        ui_region = next((region for region in view.regions if region.type == 'UI'), None)
        if ui_region is not None:
            try:
                ui_region.active_panel_category = "Voxel Palette"
            except Exception:
                pass

        properties_areas = [area for area in screen.areas if area.type == 'PROPERTIES']
        if properties_areas:
            # Layout-derived workspaces already place this editor at the right.
            properties_areas[0].spaces.active.context = 'SCENE'
        workspace["voxel_workspace_layout"] = _WORKSPACE_UUID
        return palette_left and tools_bottom and bool(properties_areas)
    except Exception:
        return False


def _workspace_is_configured(workspace: "WorkSpace") -> bool:
    for screen in workspace.screens:
        has_settings = any(
            area.type == 'PROPERTIES' and area.spaces.active.context == 'SCENE'
            for area in screen.areas
        )
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            space = area.spaces.active
            palette_left = bool(space.show_region_ui) and any(
                region.type == 'UI' and region.alignment == 'LEFT' for region in area.regions
            )
            palette_active = any(
                region.type == 'UI' and region.active_panel_category == 'Voxel Palette'
                for region in area.regions
            )
            tools_bottom = bool(space.show_region_tool_header) and any(
                region.type == 'TOOL_HEADER' and region.alignment == 'BOTTOM'
                for region in area.regions
            )
            if palette_left and palette_active and tools_bottom and has_settings:
                return True
    return False


def _register_tool_header() -> None:
    global _header_registered
    if bpy is None or _header_registered:
        return
    from .panels import draw_voxel_tool_header
    bpy.types.VIEW3D_HT_tool_header.prepend(draw_voxel_tool_header)
    _header_registered = True


def _unregister_tool_header() -> None:
    global _header_registered
    if bpy is None or not _header_registered:
        return
    from .panels import draw_voxel_tool_header
    try:
        bpy.types.VIEW3D_HT_tool_header.remove(draw_voxel_tool_header)
    except Exception:
        pass
    _header_registered = False


def register_voxel_workspace() -> Optional["WorkSpace"]:
    """Create/reuse and configure the top-level Voxel Workspace tab."""
    if bpy is None:
        return None
    _register_tool_header()
    window = _window()
    if window is None:
        return None

    workspace = _find_workspace()
    if workspace is not None and _workspace_is_configured(workspace):
        return workspace
    if workspace is None:
        before = set(bpy.data.workspaces.keys())
        with bpy.context.temp_override(window=window, screen=window.screen):
            result = bpy.ops.workspace.duplicate()
        if result != {'FINISHED'}:
            return None
        created = list(set(bpy.data.workspaces.keys()) - before)
        if not created:
            return None
        workspace = bpy.data.workspaces[created[0]]
        workspace.name = WORKSPACE_NAME

    # Workspace/screen reassignment is applied by Blender on the next UI tick.
    # Keep the new workspace active so the deferred timer can configure its own
    # regions rather than accidentally modifying the previous workspace.
    if window.workspace != workspace:
        window.workspace = workspace
    _configure_workspace(workspace, window)
    return workspace


def _deferred_register() -> Optional[float]:
    global _timer_registered
    workspace = register_voxel_workspace()
    if workspace is None or not _workspace_is_configured(workspace):
        return 0.5
    _timer_registered = False
    return None


def schedule_voxel_workspace_registration() -> None:
    """Defer layout operators until Blender has a live window/screen context."""
    global _timer_registered
    if bpy is None or _timer_registered:
        return
    bpy.app.timers.register(_deferred_register, first_interval=0.1, persistent=False)
    _timer_registered = True


def unregister_voxel_workspace() -> None:
    """Remove callbacks but retain the user's workspace datablock/layout."""
    global _timer_registered
    if bpy is not None and _timer_registered and bpy.app.timers.is_registered(_deferred_register):
        bpy.app.timers.unregister(_deferred_register)
    _timer_registered = False
    _unregister_tool_header()
