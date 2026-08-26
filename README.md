# Voxel Workspace

Voxel Workspace is a Blender add-on for creating and editing bounded voxel
models with separate Surface and Volume voxel types.

Current release: **0.14.0**

## What you can do

- Create voxel fields with adjustable dimensions and voxel size.
- Paint Surface voxels and Volume voxels independently.
- Use separate editable color palettes for Surface and Volume materials.
- Add, erase, and repaint voxels with Blender undo support.
- Import GLB and glTF models and convert them into voxel fields.
- Fill enclosed interiors while preserving the visible surface shell.
- Mirror edits across an axis: live stroke symmetry, or one-shot half-volume
  copies with an optional paint-only recolor mode.
- Show voxel boundaries in final camera renders.
- Export exact visible Surface voxel geometry as a vertex-color OBJ.

## Requirements

- Blender **5.2** (LTS).

## Installation

1. Download the Voxel Workspace ZIP file.
2. Open Blender and choose **Edit → Preferences**.
3. Select **Add-ons** in the Preferences sidebar.
4. Open the install menu in the upper-right corner of the Add-ons window.
5. Choose **Install from Disk…** and select the Voxel Workspace ZIP file.
6. Enable **Voxel Workspace** in the add-on list if it is not enabled automatically.

## Basic workflow

1. Switch to the **Voxel Workspace** workspace using Blender’s workspace tabs.
2. Open the **Voxel** tab in the 3D Viewport’s right-side N-panel.
3. Click **Create Volume**.
4. Use the Voxel Palette tab to choose a Surface or Volume palette color.
5. In the bottom **Asset Shelf**, choose **Add Surface** or **Add Volume**.
6. Move over the voxel field and drag with the left mouse button to paint.
7. Use the Asset Shelf’s **Erase** or **Repaint** tools when needed.

Choosing a palette color or clicking a volume icon does not activate a voxel
placement tool by itself. The **Add Surface** and **Add Volume** tools must be
selected from the Asset Shelf.

## Importing a model

1. Create or select a voxel field.
2. In the Voxel panel, choose **Import GLB into Volume**.
3. Select a `.glb` or `.gltf` file.
4. Choose the padding, occupancy mode, and palette size, then confirm.

Imported colors are sampled and reduced to the selected palette size. Larger
palette sizes preserve more color variation.

## Mirroring

The **Mirror** box at the bottom of the Voxel Palette panel has two tools.

**Active Mirror** repeats brush strokes symmetrically while you paint. Check
one or more axes (X, Y, Z); every voxel you add, erase, or repaint is also
applied at its mirrored position. Combined axes give multi-way symmetry.

**Instant Mirror** copies one half of the volume onto the other in one click:

1. Pick the axis with the X / Y / Z radio buttons.
2. Optionally enable **Paint Only** to recolor existing voxels without adding
   or removing any geometry (same-type voxels only; cross-domain and empty
   cells are left untouched).
3. Click **+ -> -** to copy the positive half onto the negative side, or
   **- -> +** for the reverse.

Without Paint Only, the copy is exact: target voxels whose source partner is
empty are removed, and source voxels overwrite whatever the target had. Each
mirror is a single undo step.

## Rendered voxel lines

Enable **Show in Final Render** under **Rendered Surface Edges** to show lines
around visible Surface voxel faces. The line width and line color can be
adjusted in the same section. The default line color is black.

## Exact voxel-lined OBJ export

Choose **Export Exact Voxel-Lined OBJ** in the Voxel panel to export visible
Surface voxel faces. Each face contains a palette-colored center and a line
strip around its perimeter. Volume voxels are not exported.

The OBJ contains vertex colors and does not create texture or material
sidecar files. The export uses the current Surface palette colors and line
settings at the time of export.

## License

This add-on is free software licensed under **GPL-3.0-or-later** — the same
license as Blender. As a Blender add-on, it is a derivative work of Blender
and must remain GPL-compatible. See [LICENSE](LICENSE).
