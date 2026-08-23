# Architecture decisions and validation

Decisions the spikes can falsify. Amend the matching row when a verdict lands; do not leave the change only in a spike README.

Shipping target: **Blender 5.2.0 LTS** (`blender`). Python API is 5.x (Workbench / EEVEE Next / Cycles). Rows below that name 5.1 are historical spike evidence; re-check on 5.2 before treating them as current. 3.x addon prior art is stale until re-checked.

Status: `untested` | `validated` | `partial` | `invalidated`

| ID | Decision | Status | Settled by |
|----|----------|--------|------------|
| **D1** | Authoritative data is a **chunked sparse tagged brick map**: 32³ NumPy index and domain arrays, where each coordinate is Empty, Surface(index), or Volume(index). Missing brick = empty. One-cell neighbor aprons are the meshing/invalidation boundary; Blender topology, slots, and proxies are derived. | **validated/replaced by D11** | 003 brick evidence + D11 tagged persistence |
| **D2** | Persist bricks as packed **signed-int ID-property arrays** (4 raw bytes/int) on a Blender ID, with schema version and byte length. Random 1 KB / 1 MB / 20 MB payloads survived save/quit/reopen exactly. Serialize dirty bricks rather than rewriting giant aggregate properties; 50 MB assignment cost ~405 ms. Append/library-override workflows remain untested. | **validated (001)** | 001 PARTIAL verdict |
| **D3** | **Blob/memfile state is authoritative for correctness.** Blender 5.1 restored ID-property state via Undo/Redo without a private journal; unrelated 10 MB undo push was 1.29 ms (<100 ms bar), 50 MB was 4.19 ms. Commit each mouse stroke with explicit `bpy.ops.ed.undo_push()` and do **not** use `{'REGISTER','UNDO'}` for the stroke. Do not write datablocks from `undo_post`. An optional runtime delta journal may optimize preview/commands but must not compete with memfile state. | **replaced/validated (001)** | 001 PARTIAL verdict |
| **D4** | Live editing display is a POST_VIEW GPU draw-handler with **one indexed merged VBO per dirty brick**, not per-voxel instancing (Blender 5.1 Python has `draw_instanced` but no scalable per-instance position/palette attributes). Use `LESS_EQUAL`, depth writes, and strict state restoration. OpenGL/Vulkan both passed scene-depth correctness; 100k and 1M full cubes exceeded 30 fps; a dirty 32³ VBO rebuilt through completed draw in 4.66/3.94 ms. Exposed one-voxel quad perimeters are cached as a second dirty-brick line batch so adjacent same-color cells remain readable; representative 32³ edge extraction measured 1.31 ms. Never rebuild a scene-scale VBO per stroke. Hide Blender's later-drawn floor grid and draw a depth-aware voxel work grid. Preview is absent from Viewport Render Image and F12; D5 mesh is the render source. | **partial/replaced (002)** | 002 PARTIAL verdict + voxel-edge smoke |
| **D5** | Two meshers behind one interface: **vectorized naive visible faces** for synchronous live fallback (32³: 5.32 ms cold, 6.95 ms after 1% edit), and **greedy per-brick** for debounced/final commit or worker execution (Python greedy 32³: 43.51 ms, but 128³/64 bricks cold: 1.37 s). Dirty bricks only, one-cell neighbor apron, `foreach_set` bulk writes. D4 may fall back to the naive path, not synchronous Python greedy. | **partial (003)** | 003 PARTIAL verdict; 002 chooses live path |
| **D6** | Rendering uses **native typed material domains**. The one committed Surface mesh assigns polygons to native Blender material slots by Surface palette index. Each used Volume palette index renders through a derived closed proxy hull with a native Principled Volume material. Palette Material datablocks are authoritative for display color; the tagged grid remains authoritative for occupancy. The former single atlas material is migration input only. | **validated/replaced** | `test_mesh_material.py`, `test_volume_proxy.py`, `test_dual_palette.py`, D14 |
| **D7** | Picking is **3D DDA in object-local voxel space**; an Add Surface/Add Volume miss falls back to the bounded Z=0 work plane. Production brush targeting converts world rays through the Voxel Root transform and voxel size before DDA traversal; it does not use `scene.ray_cast`. Axis-aligned, tied-diagonal supercover, boundary, transformed-object, work-plane, Add, and Erase cases pass on Blender 5.2. | **validated** | `test_dda.py`, `test_brush_helpers.py`, `test_brush_modal.py` (2026-08-23) |
| **D8** | No custom `SpaceType` or `Object.mode`. “Voxel workspace” is a file-local saved **Workspace layout** plus standard N-panels, a real Asset Shelf, runtime `active_tool`/editing state, and one active volume UUID. The persistent `load_post` handler recreates layout v7 after New/Open without forcing the user out of Layout. Switching fields stops/replaces the prior edit session so only one field is active. | **validated** | `test_workspace_foreground.py`, `test_workspace_load_recreate.py`, `test_brush_modal.py` (2026-08-23) |
| **D9** | Tools mutate only through **commands**. Identity is **UUID**, not object name. Rename preserves UUID; linked duplicate sharing the same mesh intentionally shares volume identity; plain `Shift+D` with copied data receives a new UUID (dedupe rule: same UUID + different data → regenerate). Undo/redo of duplication and cross-file append/link remain untested. | **partial (001)** | 001 PARTIAL verdict; lifecycle later |
| **D10** | GLB/glTF import voxelizes with a **direct triangle occupancy classifier** (conservative center-to-triangle distance + outside flood-fill for solids), not Geometry Nodes Mesh-to-SDF-Grid and not the old GrokVoxConvert add-on. Fit is uniform contain with padding, centered X/Y, min-Z rest. Colors come from Principled base color / linked image UV samples, then `quantize_colors_median_cut`. The conversion is one explicit `undo_push`. Open meshes in solid mode fall back to shell. | **partial** | GLB import 2026-08-21 |
| **D11** | **Tagged occupancy field & domain mask persistence**: Each coordinate is `EMPTY`, `SURFACE(1..255)`, or `VOLUME(1..255)`. Persisted as two IDProperty channels per 32³ brick: `vox_brick_*` (uint8 index packed into signed i32) and `vox_domain_*` (1-bit domain mask packed into signed i32). Separate-process reload matched all SHA256 hashes across random occupancy, negative brick coords, and boundary indices (1, 137, 255). Pack+assign: 0.20–0.36 ms/brick. Native memfile Undo/Redo restored exact byte state (undo: 2.41 ms, redo: 2.29 ms). | **validated** | `voxel_field_dual_palette` spike D1 |
| **D12** | **Canonical Empty root hierarchy & transform ownership**: A plain-axes Empty named `Voxel Root` controls world transform; generated Surface Mesh and Volume proxy meshes are equal direct children with identity local transforms (`matrix_local == I`). Separate-process reload preserved hierarchy, local identity, and world transforms (`matrix_world` diff = 0.0); Undo/Redo restored transform/hierarchy edits in 2.50 ms. | **validated** | `voxel_field_dual_palette` spike D2 |
| **D13** | **Instance and mesh identity separation**: Root object carries `voxel_instance_uuid`; Surface Mesh datablock carries authoritative `voxel_uuid`. Linked duplicates share authoritative Mesh data while maintaining distinct root instance UUIDs and independent proxies. Resolvers unambiguously identify context from root or any child; orphan detection safely purges unlinked child proxies. | **validated** | `voxel_field_dual_palette` spike D3 |
| **D14** | **Root-child volume proxy render parity**: Volume proxies parented under `Voxel Root` Empty render identically to legacy direct-mesh-parented proxies in Blender 5.1 EEVEE / Cycles (RMSE 0.000000, MaxDiff 0.0). | **validated** | `voxel_field_dual_palette` spike D4 |

## Spike map

| Spike | Tests | Must not claim |
|-------|-------|----------------|
| 001 | D1 (brick in RAM), D2, D3, D9 (copy/rename) | GPU, meshing, palette |
| 003 | D1 (brick → mesh), D5, D6 | Undo, GPU instances |
| 002 | D4 (and whether D5 must become the live path) | Persistence, palette (except as already settled by 003) |
| voxel_field_dual_palette | D11 (tagged persistence), D12 (root hierarchy), D13 (instance identity), D14 (proxy render parity) | Live GPU drawing, full palette UI |

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
| D11 domain mask binary-on-ID | Raw uint8 domain byte channel (1 byte/cell) or secondary datablock. |
| D12 Empty root transform hierarchy | Single parent mesh object with internal sub-mesh components. |
| D13 multi-instance resolution | Reject linked duplication; force deep duplicate on copy. |
| D14 root proxy render parity | Revert volume proxy parenting directly to Surface mesh object. |

## What this file is not

Not a Vengi clone spec. Not a tool list. Layers, `.vox`, voxconvert, slice, symmetry stay off this table until 001–003 have verdicts.
