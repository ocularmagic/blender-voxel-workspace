"""Voxel Workspace - Author bounded voxel volumes directly in Blender."""
from typing import List, Type
try:
    import bpy
except ImportError:
    bpy = None

# Central ordered registry of Blender types
CLASSES: List[Type] = []

_registered: bool = False


def is_registered() -> bool:
    """Return True if the extension is currently registered."""
    return _registered


def register() -> None:
    """Register all extension classes and state."""
    global _registered
    if _registered:
        return

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    _registered = True


def unregister() -> None:
    """Unregister all extension classes and cleanup state."""
    global _registered
    if not _registered:
        return

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)

    _registered = False
