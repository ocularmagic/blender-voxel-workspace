# Voxel Workspace

Voxel Workspace is an installable Blender **5.2** extension for authoring independent, bounded Surface and Volume voxel fields directly in Blender.

Current release: **0.5.0**.

## Features

- Sparse NumPy-backed 32³ bricks persisted inside ordinary `.blend` files.
- A rooted object hierarchy: `Voxel Root` with equal Surface and Volume render children.
- Tagged per-cell authority: Empty, Surface palette index, or Volume palette index.
- Independent Surface and Volume palettes with native Blender materials.
- New Surface palettes start with one **Neutral Gray** material; users add additional entries as needed.
- Add Surface, Add Volume, Erase, and Stop tools in the bottom Asset Shelf.
- Surface/Volume N-panel tabs switch the corresponding placement mode, and bottom tools switch the matching palette tab.
- Material-derived live placement colors that update after Surface or Volume material changes.
- Blender-native material previews and editable Principled shader inputs in the Voxel Palette panel.
- Palette panel action buttons arranged two-per-row (Pick + Add, Compact + Sort) for readability.
- **Fill Interior** fills every voxel with no exposed face — buried solid voxels and enclosed air pockets of any size — with the active palette color, leaving the surface shell untouched.
- Mouse-driven one-voxel strokes with one Blender Undo step per completed drag; `Esc` cancels an in-progress drag.
- Object-local 3D DDA picking with a Z=0 work-plane fallback for empty fields.
- Per-brick, depth-aware GPU editing previews and voxel-cell outlines on OpenGL and Vulkan.
- One committed Surface mesh plus derived closed Volume proxy hulls.
- GLB/glTF import with occupancy classification, material/texture sampling, and palette quantization.
- A persistent **Voxel Workspace** layout recreated after New/Open while leaving the user on Blender's normal Layout workspace.

## Requirements

- Blender **5.1.0 or newer**.
- Current shipping and verification target: **Blender 5.2.0 LTS** (`fbe6228777e7`).
- No pip dependencies at runtime; Blender supplies NumPy.

## Install

1. Download or build `voxel_workspace-0.5.0.zip`.
2. In Blender, open **Edit → Preferences → Get Extensions**.
3. Open the repository menu and choose **Install from Disk…**.
4. Select the ZIP and enable **Voxel Workspace** if Blender does not enable it automatically.

For development builds:

```bash
python build_zip.py
```

The deterministic build script writes `dist/voxel_workspace-0.5.0.zip`.

Blender's extension builder is also supported:

```bash
"blender" \
  --command extension build \
  --source-dir "<repository-root>/voxel_workspace" \
  --output-dir "<repository-root>/dist"
```

## Basic workflow

1. Open **3D Viewport → N-panel → Voxel** and click **Create Volume**.
2. Open the **Voxel Palette** N-panel tab.
3. Select **Surface** or **Volume**. This activates the matching placement mode.
4. Choose or add a palette material and edit its native Blender shader properties.
5. Move the pointer over the field to preview the target, then drag with **LMB** to place voxels.
6. Select **Erase** in the bottom Asset Shelf to remove whichever tagged voxel is hit.
7. Release LMB to commit the drag as one Undo step.
8. Press **Esc while dragging** to cancel that drag, or **Esc while idle** to stop editing.

MMB, wheel, numpad navigation, Ctrl+Z, and Ctrl+Shift+Z pass through to Blender. Each placement drag targets the field state that existed when the drag began, preventing newly placed voxels from stacking toward the camera.

In the **Voxel Palette** panel, select a color then click **Fill Interior** to recolor every buried voxel and fill all enclosed air pockets (hollow interiors, voids) with that color. Voxels with any exposed face are left untouched. The button fills using the active palette tab's selected color.

Use **Show Voxel Edges** in the Voxel panel to toggle editing outlines. This affects the live editing preview, not EEVEE or Cycles renders.

> **Asset Shelf note:** tool behavior is synchronized with the N-panel, but Blender 5.2 keeps the real Asset Shelf tile's blue active highlight in an internal `AssetWeakReference`. Changing tools from the N-panel cannot programmatically move that highlight for these local Object assets.

## Import GLB/glTF

1. Select a voxel field.
2. In the Voxel panel click **Import GLB into Volume** and choose a `.glb` or `.gltf` file.
3. Default settings use contain-fit with one voxel of padding, centered X/Y, resting on the field floor, solid occupancy, and a 64-color palette.
4. Confirm the import. Conversion is committed as one Undo step.

A nonempty target requires **Clear and Replace Volume** in the file dialog. Open or non-manifold meshes fall back to a surface shell. Large grids above 128³ cells warn that conversion may be slow.

Only one voxel field is actively edited at a time. Object transforms are honored because rays are converted into object-local voxel coordinates before DDA traversal.

## Development and verification

Run the unit suite:

```bash
uv run pytest -q
```

Current result: **92 passed** on Blender Workspace release `0.5.0` source.

Blender integration and acceptance scripts are under `tests/blender/`. The verified commands and artifact list are documented in [`tests/ACCEPTANCE.md`](tests/ACCEPTANCE.md). Architecture decisions and current validation status are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
