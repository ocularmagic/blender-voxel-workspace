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
    from bpy.app.handlers import persistent
except ImportError:
    bpy = None
    WorkSpace = object
    persistent = lambda function: function

WORKSPACE_NAME = "Voxel Workspace"
_WORKSPACE_UUID = "voxel-workspace-layout-v3"
_timer_registered = False


@persistent
def _voxel_workspace_load_post(_unused: Any) -> None:
    """Recreate the file-local workspace after New/Open replaces datablocks."""
    global _timer_registered
    _timer_registered = False
    schedule_voxel_workspace_registration()


def _register_load_handler() -> None:
    if bpy is None:
        return
    for handler in list(bpy.app.handlers.load_post):
        if (
            handler is not _voxel_workspace_load_post
            and getattr(handler, "__module__", "") == __name__
            and getattr(handler, "__name__", "") == "_voxel_workspace_load_post"
        ):
            bpy.app.handlers.load_post.remove(handler)
    if _voxel_workspace_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_voxel_workspace_load_post)


def _unregister_load_handler() -> None:
    if bpy is not None and _voxel_workspace_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_voxel_workspace_load_post)


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
        palette_areas = [area for area in screen.areas if area.type == 'TEXT_EDITOR' and area.x < 400]
        if not view_areas:
            return False
        if len(view_areas) == 1 and not palette_areas:
            view = view_areas[0]
            with bpy.context.temp_override(window=window, screen=screen, area=view):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.13)
            return False  # New area dimensions/regions settle on the next UI tick.

        if not palette_areas:
            left = min(view_areas, key=lambda area: area.x)
            left.type = 'TEXT_EDITOR'
            return False  # Editor regions are recreated on the next UI tick.
        else:
            left = min(palette_areas, key=lambda area: area.x)
        main = max((area for area in screen.areas if area.type == 'VIEW_3D'), key=lambda area: area.width * area.height)

        left_space = left.spaces.active
        left_space.show_region_ui = True
        if hasattr(left_space, "show_region_toolbar"):
            left_space.show_region_toolbar = False
        if hasattr(left_space, "show_region_header"):
            left_space.show_region_header = False
        if hasattr(left_space, "show_region_tool_header"):
            left_space.show_region_tool_header = False
        palette_left = _flip_region(window, left, 'UI', 'LEFT')
        left_ui = next((region for region in left.regions if region.type == 'UI'), None)
        if left_ui is not None:
            left_ui.active_panel_category = "Voxel Palette"

        main_space = main.spaces.active
        main_space.show_region_ui = True
        main_space.show_region_toolbar = True
        main_space.show_region_header = True
        sidebar_right = _flip_region(window, main, 'UI', 'RIGHT')
        main_ui = next((region for region in main.regions if region.type == 'UI'), None)
        if main_ui is not None and main_ui.active_panel_category == 'Voxel Palette':
            main_ui.active_panel_category = 'Item'

        # Restore the Layout workspace's normal right-side Properties context.
        properties_areas = [area for area in screen.areas if area.type == 'PROPERTIES']
        if properties_areas:
            properties_areas[0].spaces.active.context = 'OBJECT'
        workspace["voxel_workspace_layout"] = _WORKSPACE_UUID
        return palette_left and sidebar_right
    except Exception:
        return False


def _workspace_is_configured(workspace: "WorkSpace") -> bool:
    for screen in workspace.screens:
        views = [area for area in screen.areas if area.type == 'VIEW_3D']
        palettes = [area for area in screen.areas if area.type == 'TEXT_EDITOR' and area.x < 400]
        if not views or not palettes:
            continue
        left = min(palettes, key=lambda area: area.x)
        main = max(views, key=lambda area: area.width * area.height)
        palette_left = bool(left.spaces.active.show_region_ui) and any(
            region.type == 'UI' and region.alignment == 'LEFT'
            and region.active_panel_category == 'Voxel Palette'
            for region in left.regions
        )
        sidebar_right = bool(main.spaces.active.show_region_ui) and any(
            region.type == 'UI' and region.alignment == 'RIGHT' for region in main.regions
        )
        if palette_left and sidebar_right:
            return True
    return False


def register_voxel_workspace() -> Optional["WorkSpace"]:
    """Create/reuse and configure the top-level Voxel Workspace tab."""
    if bpy is None:
        return None
    _register_load_handler()
    window = _window()
    if window is None:
        return None

    workspace = _find_workspace()
    if workspace is not None and _workspace_is_configured(workspace):
        return workspace
    if workspace is None:
        template = bpy.data.workspaces.get('Layout')
        if template is not None and window.workspace != template:
            window.workspace = template
            return None  # Duplicate the predictable Layout screen next UI tick.
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
    _unregister_load_handler()
