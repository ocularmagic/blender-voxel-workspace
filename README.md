# Voxel Workspace

Voxel Workspace is a Blender add-on for creating and editing bounded voxel
models with separate Surface and Volume voxel types.

Current release: **0.17.0**

## What you can do

- Create voxel fields with adjustable dimensions and voxel size.
- Paint Surface voxels and Volume voxels independently.
- Draw on the interior of ANY bounding-box wall, not just the floor.
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

## Drawing on interior walls

When an empty (or partly empty) volume is in front of you, placement lands on
whichever bounding-box wall the view ray exits through — the wall whose
interior surface faces you. Looking down draws on the floor; looking up from
below draws on the ceiling; orbiting around the side moves drawing onto that
side wall automatically.

The reference grid follows the same rule: grid lines appear only on walls
whose inner surface fronts the viewer. A wall seen through its backside (for
example the ceiling when looking down) shows no grid and cannot be drawn on.
Occupied voxels always take priority — placing against existing geometry,
mirroring, erasing, and repainting all behave exactly as before.

## Adjusting the voxel root size

To resize the voxel root interactively:

1. Open the **Voxel** tab in the 3D Viewport’s right-side N-panel.
2. Click **Adjust voxel root size**.
3. Drag one of the colored axis arrows at a root corner:
   - **Red** arrows resize along X.
   - **Green** arrows resize along Y.
   - **Blue** arrows resize along Z.
4. Continue dragging any arrow to make additional one-axis changes.
5. Click **Accept** to keep all changes, or click **Cancel** to restore the
   root dimensions from before adjustment mode began.

The Accept and Cancel controls appear below the adjustment button while the
tool is active. They are highlighted to make the active confirmation scope
clear. Pressing **Esc** has the same effect as Cancel.

Adjustment mode prevents Surface/Volume painting, erasing, and repainting
until it is accepted or cancelled. A resize never moves existing voxels or
silently deletes them. An empty root can be reduced to one voxel per axis;
when voxels are present, the root cannot be reduced past the occupied voxel
bounds. Growth is limited to 512 voxels per axis.

## Stretching and squashing the interior

Where **Adjust voxel root size** adds or removes *empty* space, **Stretch /
squash interior voxels** scales the voxels themselves together with the root,
so a shape gets taller, wider, or flatter instead of gaining room around it:

1. Open the **Voxel** tab in the 3D Viewport’s right-side N-panel.
2. Click **Stretch / squash interior voxels** — its own section directly below
   the Resize Volume box.
3. Drag one of the colored axis arrows at a root corner:
   - **Red** arrows scale along X.
   - **Green** arrows scale along Y.
   - **Blue** arrows scale along Z.
4. Continue dragging any arrow for additional one-axis changes.
5. Click **Accept** to keep all changes, or click **Cancel** to restore the
   volume exactly as it was before this scaling session began. **Esc**
   cancels as well.

Stretching fills solidly: any run of voxels along the dragged axis expands
into the full space between its endpoints, so shapes stay contiguous instead
of turning into dotted outlines — even an isolated voxel grows into a short
column of voxels. Squashing compacts overlapping voxels into single cells and
never deletes content. Like adjustment mode, scaling blocks brush editing
until accepted or cancelled, and each released drag is its own undo step.

**Boundary protection:** while scaling is active or not, an add brush never
overwrites voxels at the root's edge. If the root is full to a side, drawing
against that side does nothing instead of replacing the outermost voxels.

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
