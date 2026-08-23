"""Creation and configuration of the custom Voxel Workspace layout.

Blender 5.1 VIEW_3D has one UI region (the N-panel) and cannot host a second
native category-tab strip without splitting a separate editor area. The
workspace therefore:

* Keeps the 3D View N-panel on the RIGHT (Item / Tool / View / Voxel).
* Adds Voxel Palette as another native UI category on that same strip.
* Pins brush actions to the bottom Asset Shelf (Sculpting-style thumbnails).
* Leaves native Toolbar tools (Select / Cursor / …) on the left.

Never assign ``window.workspace`` from register/install/load_post. Doing that
from inside ``wm_event_do_notifiers`` crashes Blender 5.2 (null compositor
texture in ``ED_workspace_change``). Create the tab in place and configure
regions only after the user is already on Voxel Workspace.
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
_WORKSPACE_UUID = "voxel-workspace-layout-v7"
_timer_registered = False
_msgbus_owner = object()
_pending_restore_name: Optional[str] = None
_created_this_pass = False


@persistent
def _voxel_workspace_load_post(_unused: Any) -> None:
    """Recreate the file-local workspace after New/Open replaces datablocks."""
    global _timer_registered, _pending_restore_name
    _timer_registered = False
    _pending_restore_name = None
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


def _on_active_workspace_changed(*_args: Any) -> None:
    schedule_voxel_workspace_registration()


def _subscribe_workspace_activation() -> None:
    if bpy is None:
        return
    try:
        bpy.msgbus.clear_by_owner(_msgbus_owner)
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.Window, "workspace"),
            owner=_msgbus_owner,
            args=(),
            notify=_on_active_workspace_changed,
        )
    except Exception:
        pass


def _unsubscribe_workspace_activation() -> None:
    if bpy is None:
        return
    try:
        bpy.msgbus.clear_by_owner(_msgbus_owner)
    except Exception:
        pass


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
        return result == {"FINISHED"} and region.alignment == alignment
    except Exception:
        return False


def _toggle_region(window: Any, area: Any, region_type: str) -> bool:
    space = area.spaces.active
    window_region = next((region for region in area.regions if region.type == "WINDOW"), None)
    if window_region is None:
        return False
    try:
        with bpy.context.temp_override(
            window=window,
            screen=window.screen,
            area=area,
            region=window_region,
            space_data=space,
        ):
            result = bpy.ops.screen.region_toggle(region_type=region_type)
        return result == {"FINISHED"}
    except Exception:
        return False


def _join_stray_split_editors(window: Any, screen: Any) -> bool:
    """Collapse the v3 TEXT_EDITOR palette column back into the 3D view."""
    leftovers = [
        area
        for area in screen.areas
        if area.type == "TEXT_EDITOR" and area.x < 400 and area.width < 480
    ]
    views = [area for area in screen.areas if area.type == "VIEW_3D"]
    if not leftovers or not views:
        return False
    left = min(leftovers, key=lambda area: area.x)
    main = max(views, key=lambda area: area.width * area.height)
    source_xy = (left.x + 4, left.y + max(left.height // 2, 4))
    target_xy = (main.x + max(main.width // 2, 8), main.y + max(main.height // 2, 4))
    try:
        with bpy.context.temp_override(window=window, screen=screen):
            result = bpy.ops.screen.area_join(source_xy=source_xy, target_xy=target_xy)
        return result == {"FINISHED"}
    except Exception:
        return False


def _main_view(screen: Any) -> Any:
    views = [area for area in screen.areas if area.type == "VIEW_3D"]
    if not views:
        return None
    return max(views, key=lambda area: area.width * area.height)


def _configure_workspace(workspace: "WorkSpace", window: Any) -> bool:
    """Configure supported Blender regions for the Voxel Workspace."""
    if window.workspace != workspace or all(window.screen != screen for screen in workspace.screens):
        return False
    screen = window.screen
    try:
        if _join_stray_split_editors(window, screen):
            return False

        main = _main_view(screen)
        if main is None:
            return False

        main_space = main.spaces.active
        main_space.show_region_ui = True
        main_space.show_region_toolbar = True
        main_space.show_region_header = True
        if hasattr(main_space, "show_region_tool_header"):
            main_space.show_region_tool_header = True
        _flip_region(window, main, "TOOLS", "LEFT")
        _flip_region(window, main, "TOOL_HEADER", "TOP")

        if not bool(getattr(main_space, "show_region_asset_shelf", False)):
            _toggle_region(window, main, "ASSET_SHELF")

        try:
            from .shelf import ensure_tool_assets
            ensure_tool_assets()
        except Exception:
            pass

        sidebar_right = _flip_region(window, main, "UI", "RIGHT")

        workspace["voxel_workspace_layout"] = _WORKSPACE_UUID
        shelf_on = bool(getattr(main_space, "show_region_asset_shelf", False))
        return sidebar_right and shelf_on
    except Exception:
        return False


def _workspace_is_configured(workspace: "WorkSpace") -> bool:
    if workspace.get("voxel_workspace_layout") != _WORKSPACE_UUID:
        return False
    for screen in workspace.screens:
        if any(area.type == "TEXT_EDITOR" and area.x < 400 and area.width < 480 for area in screen.areas):
            continue
        main = _main_view(screen)
        if main is None:
            continue
        space = main.spaces.active
        sidebar_right = bool(space.show_region_ui) and any(
            region.type == "UI" and region.alignment == "RIGHT" for region in main.regions
        )
        shelf_on = bool(getattr(space, "show_region_asset_shelf", False))
        if sidebar_right and shelf_on:
            return True
    return False


def _duplicate_current_workspace(window: Any) -> Optional["WorkSpace"]:
    """Copy the active workspace. Does not change ``window.workspace``."""
    before = set(bpy.data.workspaces.keys())
    try:
        with bpy.context.temp_override(window=window, screen=window.screen):
            result = bpy.ops.workspace.duplicate()
    except Exception:
        return None
    if result != {"FINISHED"}:
        return None
    created = list(set(bpy.data.workspaces.keys()) - before)
    if not created:
        return None
    return bpy.data.workspaces[created[0]]


def install_workspace_handlers() -> None:
    """Attach load/activation handlers without touching window.workspace."""
    _register_load_handler()
    _subscribe_workspace_activation()


def _restore_pending_workspace(window: Any) -> None:
    """Return to the pre-duplicate workspace on a later timer tick."""
    global _pending_restore_name
    name = _pending_restore_name
    _pending_restore_name = None
    if window is None or not name or bpy is None:
        return
    target = bpy.data.workspaces.get(name) or bpy.data.workspaces.get("Layout")
    if target is not None and window.workspace != target:
        window.workspace = target


def register_voxel_workspace() -> Optional["WorkSpace"]:
    """Create/reuse the Voxel Workspace tab without leaving the user there.

    ``workspace.duplicate()`` activates the copy, so we remember the previous
    tab and switch back on a later timer tick — never in the same call, and
    never from ``register()`` / install notifiers.
    """
    global _pending_restore_name, _created_this_pass
    _created_this_pass = False
    if bpy is None:
        return None
    _register_load_handler()
    _subscribe_workspace_activation()
    window = _window()
    if window is None:
        return None

    workspace = _find_workspace()
    if workspace is None:
        current = getattr(window.workspace, "name", None)
        if current and current != WORKSPACE_NAME and _pending_restore_name is None:
            _pending_restore_name = current
        workspace = _duplicate_current_workspace(window)
        if workspace is None:
            return None
        workspace.name = WORKSPACE_NAME
        workspace["voxel_workspace_layout"] = _WORKSPACE_UUID
        _created_this_pass = True

    if window.workspace == workspace:
        _configure_workspace(workspace, window)
    return workspace


def _deferred_register() -> Optional[float]:
    global _timer_registered
    window = _window()
    workspace = register_voxel_workspace()
    if workspace is None or window is None:
        return 0.5
    # duplicate() just activated Voxel Workspace; restore next tick.
    if _created_this_pass:
        return 0.5
    if window.workspace == workspace and not _workspace_is_configured(workspace):
        return 0.5
    if _pending_restore_name and window.workspace == workspace:
        _restore_pending_workspace(window)
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
    _unsubscribe_workspace_activation()
    _unregister_load_handler()
