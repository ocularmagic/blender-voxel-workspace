# Voxel Workspace

Voxel Workspace is an installable Blender 5.1 extension for authoring independent bounded voxel volumes directly in Blender.

## Vertical-slice features

- Sparse NumPy-backed 32³ bricks with ordinary Blender Mesh objects.
- Mouse-driven one-voxel **Place** and **Erase** strokes in the 3D Viewport.
- One Blender Undo/Redo step per completed drag; `Esc` cancels an in-progress drag.
- `.blend` persistence with continued editing after a separate-process reopen.
- Multiple independent volume objects identified by Mesh UUID rather than object name.
- Per-brick, depth-aware GPU preview on OpenGL and Vulkan.
- Depth-aware voxel cell outlines keep adjacent same-color voxels readable while editing.
- Committed palette mesh geometry rendered by both EEVEE and Cycles.

Layers, selection, symmetry, `.vox` import/export, custom Workspace layouts, and other deferred tools are intentionally outside this vertical slice.

## Requirements

- Blender **5.1.0 or newer** (verified with Blender 5.1.2).
- No pip dependencies at runtime; Blender supplies NumPy.

## Install

1. Download or build `voxel_workspace-0.1.3.zip`.
2. In Blender, open **Edit → Preferences → Get Extensions**.
3. Use the repository menu and choose **Install from Disk…**.
4. Select the ZIP and enable **Voxel Workspace** if Blender does not enable it automatically.

For development builds:

```bash
"blender" \
  --command extension build \
  --source-dir "<repository-root>/voxel_workspace" \
  --output-dir "<repository-root>/dist"
```

## Controls

Open **3D Viewport → N-panel → Voxel**.

1. Click **Create Volume**. The operator defaults to 32×32×32 voxels at voxel size 1.0.
2. Choose one of the eight visible **Placement Color** swatches. Neutral Gray is the default.
3. Select a voxel volume and click **Start Place** or **Start Erase**.
   The color swatches remain clickable while either tool is active, so you do not need to exit editing to change colors.
4. Move the pointer to preview the target; drag with **LMB** to edit.
   Each drag targets the surface that existed when the drag began, preventing newly placed voxels from stacking toward the camera.
5. Release LMB to commit the entire drag as one Undo step.
6. Press **Esc while dragging** to cancel that drag, or **Esc while idle** to stop editing.
7. MMB, wheel, numpad navigation, Ctrl+Z, and Ctrl+Shift+Z pass through to Blender.

Use **Show Voxel Edges** in the Voxel Brush panel to toggle the editing outlines. This affects only the live editing preview, not EEVEE or Cycles renders.

Only one volume is actively edited at a time. Object transforms are honored because picking and preview operate in object-local voxel space.

## Development and verification

```bash
uv run pytest
```

Blender integration and acceptance scripts are under `tests/blender/`. The complete verified command set and artifact list are documented in [`tests/ACCEPTANCE.md`](tests/ACCEPTANCE.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
