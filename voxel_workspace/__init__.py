"""Voxel Workspace - Author bounded voxel volumes directly in Blender."""
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

    unregister_runtime()
    unregister_properties()
    unregister_palette_icons()

    _registered = False
