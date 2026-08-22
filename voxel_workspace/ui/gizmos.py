"""Large viewport-bottom voxel tool gizmos for the custom workspace."""
from typing import Any

try:
    import bpy
    from bpy.types import GizmoGroup
except ImportError:
    bpy = None
    GizmoGroup = object

_TOOL_SPECS = (
    ("ADD_SURFACE", "voxel.start_surface", 'BRUSH_DATA'),
    ("ADD_VOLUME", "voxel.start_volume", 'MOD_FLUIDSIM'),
    ("ERASE", "voxel.start_erase", 'TRASH'),
    ("STOP", "voxel.stop_editing", 'CANCEL'),
)


def _is_main_voxel_view(context: Any) -> bool:
    if context is None or getattr(context, "workspace", None) is None:
        return False
    if context.workspace.name != "Voxel Workspace" or context.area is None:
        return False
    return any(region.type == 'UI' and region.alignment == 'RIGHT' for region in context.area.regions)


class VOXEL_GGT_workspace_tools(GizmoGroup):
    """Sculpt-toolbar-sized Add Surface, Add Volume, Erase, and Stop buttons."""
    bl_idname = "VOXEL_GGT_workspace_tools"
    bl_label = "Voxel Workspace Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SCALE', 'SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls, context: Any) -> bool:
        return _is_main_voxel_view(context)

    def setup(self, context: Any) -> None:
        self._buttons = []
        for mode, operator_id, icon in _TOOL_SPECS:
            button = self.gizmos.new("GIZMO_GT_button_2d")
            button.icon = icon
            button.draw_options = {'BACKDROP', 'OUTLINE'}
            button.color = (0.18, 0.18, 0.18)
            button.alpha = 0.92
            button.color_highlight = (0.45, 0.62, 1.0)
            button.alpha_highlight = 1.0
            button.scale_basis = 20.0
            button.target_set_operator(operator_id)
            self._buttons.append((mode, button))

    def draw_prepare(self, context: Any) -> None:
        active_tool = getattr(context.scene.voxel_workspace, "active_tool", "NONE")
        spacing = 58.0
        start_x = (context.region.width - spacing * (len(self._buttons) - 1)) * 0.5
        y = 50.0
        for index, (mode, button) in enumerate(self._buttons):
            button.matrix_basis[0][3] = start_x + index * spacing
            button.matrix_basis[1][3] = y
            active = mode == active_tool or (mode == "STOP" and active_tool != "NONE")
            button.color = (0.18, 0.38, 0.8) if active else (0.18, 0.18, 0.18)


GIZMO_CLASSES = [VOXEL_GGT_workspace_tools]
