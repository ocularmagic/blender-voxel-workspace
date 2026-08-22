"""Voxel Workspace - Author bounded voxel volumes directly in Blender."""
import importlib
from typing import List, Type


try:
    import bpy
except ImportError:
    bpy = None

from .blender.properties import register_properties, unregister_properties
from .blender.runtime import register_runtime, unregister_runtime
from .operators import OPERATOR_CLASSES
from .ui import PANEL_CLASSES
from .ui.palette_icons import register_palette_icons, unregister_palette_icons
from .ui import workspace as _workspace_ui

# Blender can retain an old extension submodule in ``sys.modules`` after an
# in-session reinstall. Reload only that stale module before binding its API so
# enabling the freshly copied extension cannot fail on a newly added symbol.
if not hasattr(_workspace_ui, "schedule_voxel_workspace_registration"):
    _workspace_ui = importlib.reload(_workspace_ui)

register_voxel_workspace = _workspace_ui.register_voxel_workspace
schedule_voxel_workspace_registration = _workspace_ui.schedule_voxel_workspace_registration
unregister_voxel_workspace = _workspace_ui.unregister_voxel_workspace

# Central ordered registry of Blender types
CLASSES: List[Type] = [
    *OPERATOR_CLASSES,
    *PANEL_CLASSES,
]

_registered: bool = False


def is_registered() -> bool:
    """Return True if the extension is currently registered."""
    return _registered


def register() -> None:
    """Register all extension classes and state."""
    global _registered
    if _registered:
        return

    register_palette_icons()
    register_properties()
    register_runtime()

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    # Create/activate the custom Voxel Workspace (left palette + bottom tools).
    # Safe no-op in background import contexts without a window.
    try:
        register_voxel_workspace()
        schedule_voxel_workspace_registration()
    except Exception:
        schedule_voxel_workspace_registration()

    _registered = True


def unregister() -> None:
    """Unregister all extension classes and cleanup state."""
    global _registered
    if not _registered:
        return

    # Draw handlers and saved viewport overlay state must be gone before
    # classes/properties disappear underneath their callbacks.
    try:
        from .blender.gpu_preview import cleanup_gpu_preview
        cleanup_gpu_preview()
    except Exception:
        pass

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

    # Leave the Voxel Workspace layout in place on unregister; removing a
    # workspace is unsafe from Python and would destroy the user's layout.
    try:
        unregister_voxel_workspace()
    except Exception:
        pass

    unregister_runtime()
    unregister_properties()
    unregister_palette_icons()

    _registered = False
