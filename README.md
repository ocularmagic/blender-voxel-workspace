# Voxel Workspace

Voxel Workspace is a Blender extension for authoring independent bounded voxel volumes directly in Blender.

## Vertical-slice scope

The first milestone targets a single active edit volume at a time and provides:

- independent 32×32×32 voxel volumes backed by sparse NumPy bricks;
- mouse-driven one-voxel Place and Erase strokes;
- one Blender Undo/Redo step per stroke;
- `.blend` persistence and continued editing after reopen;
- committed mesh geometry with palette colors for EEVEE and Cycles;
- a depth-aware per-brick GPU preview while editing.

Layers, selection, symmetry, `.vox` import/export, custom workspace layouts, and other deferred tools are intentionally outside this vertical slice.

## Target environment

Tested against Blender **5.1.2**, its bundled Python 3.13 and NumPy 2.3.4. The pure core supports Python 3.11 or newer and is tested with pytest via `uv`.

## Development

```bash
uv run pytest
```

Blender integration tests are standalone scripts under `tests/blender/` and are run with Blender's `--background --factory-startup --python` command line.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
