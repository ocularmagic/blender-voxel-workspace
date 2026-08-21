# Architecture (under test)

Decisions the spikes can falsify. Amend the matching row when a verdict lands; do not leave the change only in a spike README.

Shipping target: **Blender 5.x** Python API (Workbench / EEVEE Next / Cycles). 3.x addon prior art is stale until re-checked.

Status: `untested` | `validated` | `partial` | `invalidated`

| ID | Decision | Status | Settled by |
|----|----------|--------|------------|
| **D1** | Authoritative data is a **chunked sparse brick map**: 16³ or 32³ NumPy `uint8` (index 0 = empty). Missing brick = empty. A 32³ `uint8` brick plus one-cell neighbor apron is validated as the meshing/invalidation unit; 64 such bricks built a 128³ test volume correctly. The sparse map container itself remains to be exercised. Not a Python dict of individual cells and not Blender topology. | **partial (003)** | 003 PARTIAL verdict; sparse container later |
| **D2** | Persist bricks as packed **signed-int ID-property arrays** (4 raw bytes/int) on a Blender ID, with schema version and byte length. Random 1 KB / 1 MB / 20 MB payloads survived save/quit/reopen exactly. Serialize dirty bricks rather than rewriting giant aggregate properties; 50 MB assignment cost ~405 ms. Append/library-override workflows remain untested. | **validated (001)** | 001 PARTIAL verdict |
| **D3** | **Blob/memfile state is authoritative for correctness.** Blender 5.1 restored ID-property state via Undo/Redo without a private journal; unrelated 10 MB undo push was 1.29 ms (<100 ms bar), 50 MB was 4.19 ms. Commit each mouse stroke with explicit `bpy.ops.ed.undo_push()` and do **not** use `{'REGISTER','UNDO'}` for the stroke. Do not write datablocks from `undo_post`. An optional runtime delta journal may optimize preview/commands but must not compete with memfile state. | **replaced/validated (001)** | 001 PARTIAL verdict |
| **D4** | Live editing display is a POST_VIEW GPU draw-handler with **one indexed merged VBO per dirty brick**, not per-voxel instancing (Blender 5.1 Python has `draw_instanced` but no scalable per-instance position/palette attributes). Use `LESS_EQUAL`, depth writes, and strict state restoration. OpenGL/Vulkan both passed scene-depth correctness; 100k and 1M full cubes exceeded 30 fps; a dirty 32³ VBO rebuilt through completed draw in 4.66/3.94 ms. Exposed one-voxel quad perimeters are cached as a second dirty-brick line batch so adjacent same-color cells remain readable; representative 32³ edge extraction measured 1.31 ms. Never rebuild a scene-scale VBO per stroke. Hide Blender's later-drawn floor grid and draw a depth-aware voxel work grid. Preview is absent from Viewport Render Image and F12; D5 mesh is the render source. | **partial/replaced (002)** | 002 PARTIAL verdict + 0.1.3 edge smoke |
| **D5** | Two meshers behind one interface: **vectorized naive visible faces** for synchronous live fallback (32³: 5.32 ms cold, 6.95 ms after 1% edit), and **greedy per-brick** for debounced/final commit or worker execution (Python greedy 32³: 43.51 ms, but 128³/64 bricks cold: 1.37 s). Dirty bricks only, one-cell neighbor apron, `foreach_set` bulk writes. D4 may fall back to the naive path, not synchronous Python greedy. | **partial (003)** | 003 PARTIAL verdict; 002 chooses live path |
| **D6** | **One material per volume**; mesh carries CORNER-domain `INT` `palette_index`; shader samples nearest from that volume's 256×1 palette image (`VoxelPalette_<uuid>`) at `(index + 0.5) / 256`. Indices 1–7 rendered in Blender 5.1 EEVEE (`BLENDER_EEVEE`) and Cycles with one slot per mesh (PNG RMSE 0.0057). Do not rely on setting deprecated `Material.use_nodes` in 5.1+. | **validated (003/M1)** | 003 PARTIAL verdict; M1 per-volume update |
| **D7** | Picking is **3D DDA in object-local space**; miss hits a work plane. Not `scene.ray_cast` as the production picker. | untested | later (not 001–003) |
| **D8** | No custom `SpaceType` or `Object.mode`. “Voxel workspace” = **saved Workspace layout** + `WorkSpaceTool`s + `is_voxel_editing` flag. One volume edited at a time. | untested | later |
| **D9** | Tools mutate only through **commands**. Identity is **UUID**, not object name. Rename preserves UUID; linked duplicate sharing the same mesh intentionally shares volume identity; plain `Shift+D` with copied data receives a new UUID (dedupe rule: same UUID + different data → regenerate). Undo/redo of duplication and cross-file append/link remain untested. | **partial (001)** | 001 PARTIAL verdict; lifecycle later |
| **D10** | GLB/glTF import voxelizes with a **direct triangle occupancy classifier** (conservative center-to-triangle distance + outside flood-fill for solids), not Geometry Nodes Mesh-to-SDF-Grid and not the old GrokVoxConvert add-on. Fit is uniform contain with padding, centered X/Y, min-Z rest. Colors come from Principled base color / linked image UV samples, then `quantize_colors_median_cut`. The conversion is one explicit `undo_push`. Open meshes in solid mode fall back to shell. | **partial** | GLB import 2026-08-21 |

## Spike map

| Spike | Tests | Must not claim |
|-------|-------|----------------|
| 001 | D1 (brick in RAM), D2, D3, D9 (copy/rename) | GPU, meshing, palette |
| 003 | D1 (brick → mesh), D5, D6 | Undo, GPU instances |
| 002 | D4 (and whether D5 must become the live path) | Persistence, palette (except as already settled by 003) |

Run order: **001 → 003 → 002**. Storage first. Mesh throughput next, because it is D4's escape hatch. GPU last, so an INVALIDATED D4 is still actionable.

## Fallback ladder (pre-committed)

If a spike invalidates a D#, use the next rung. Do not invent a new product in the verdict.

| If this dies | Then |
|--------------|------|
| D2 binary-on-ID | Sidecar file next to the `.blend` (path + hash on the object). Own undo or accept no Blender-undo for volume data. |
| D3 journal reconcile | **Plan B1:** blob-only, trust memfile, keep blobs small (brick-granular IDs or separate datablocks) so unrelated undo stays cheap. **Plan B2:** sidecar + addon-owned undo; disable Blender undo while `is_voxel_editing`. |
| D4 GPU depth or instancing | Live path = D5 on a timer / on stroke; GPU handler only for bounds, hover, work plane. |
| D5 bulk mesh write too slow | Smaller bricks, or a compiled mesher later; do not GPU-preview your way around a mesh you cannot commit. |
| D6 attribute lookup | Explode used indices to material slots (capped). |

## What this file is not

Not a Vengi clone spec. Not a tool list. Layers, `.vox`, voxconvert, slice, symmetry stay off this table until 001–003 have verdicts.
