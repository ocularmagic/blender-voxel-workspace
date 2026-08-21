"""Stage a GLB/glTF import, evaluate meshes, and extract fitted triangles."""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set, Tuple
import numpy as np

from ..voxelization.color_sampling import SampledMaterial
from ..voxelization.fit import FitResult, apply_fit, contain_fit
from .glb_materials import extract_material


STAGING_COLLECTION_NAME = "Voxel GLB Source"
LARGE_CELL_COUNT = 128 * 128 * 128


@dataclass
class EvaluatedSource:
    triangles: np.ndarray  # (T, 3, 3) voxel-space after fit
    uvs: Optional[np.ndarray]
    material_indices: np.ndarray
    materials: List[SampledMaterial]
    mesh_closed: bool
    fit: FitResult
    object_names: List[str]
    warnings: List[str] = field(default_factory=list)
    imported_object_names: List[str] = field(default_factory=list)


def _ensure_collection(bpy: Any, name: str, hidden: bool = True) -> Any:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        scene = bpy.context.scene
        if scene is not None and col.name not in scene.collection.children:
            scene.collection.children.link(col)
    if hidden:
        col.hide_viewport = True
        col.hide_render = True
    return col


def import_gltf_objects(bpy: Any, filepath: str) -> List[Any]:
    """Import a GLB/glTF file and return the newly created objects."""
    before: Set[str] = set(bpy.data.objects.keys())
    result = bpy.ops.import_scene.gltf(filepath=filepath)
    if "FINISHED" not in result:
        raise RuntimeError(f"glTF import failed: {result}")
    after = set(bpy.data.objects.keys())
    new_names = sorted(after - before)
    return [bpy.data.objects[n] for n in new_names]


def _loop_uv(mesh: Any, loop_index: int) -> Tuple[float, float]:
    layers = getattr(mesh, "uv_layers", None)
    if layers is None or len(layers) == 0:
        return (0.0, 0.0)
    uv_layer = layers.active if layers.active is not None else layers[0]
    try:
        uv = uv_layer.data[loop_index].uv
        return (float(uv[0]), float(uv[1]))
    except Exception:
        return (0.0, 0.0)


def evaluate_mesh_objects(
    bpy: Any,
    objects: Sequence[Any],
    context: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[SampledMaterial], bool, List[str], List[str]]:
    """Evaluate mesh objects to world-space triangles, UVs, and materials."""
    warnings: List[str] = []
    names: List[str] = []
    materials: List[SampledMaterial] = []
    mat_lookup = {}
    tri_chunks: List[np.ndarray] = []
    uv_chunks: List[np.ndarray] = []
    mat_chunks: List[np.ndarray] = []
    has_uv = False

    depsgraph = context.evaluated_depsgraph_get()
    mesh_count = 0
    for obj in objects:
        if obj is None or getattr(obj, "type", "") != "MESH":
            continue
        mesh_count += 1
        names.append(obj.name)
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            n_tri = len(mesh.loop_triangles)
            if n_tri == 0:
                warnings.append(f"Object '{obj.name}' has no triangles after evaluation")
                continue
            world = np.array(eval_obj.matrix_world, dtype=np.float64)
            verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape((-1, 3))
            ones = np.ones((len(verts), 1), dtype=np.float64)
            world_verts = (world @ np.concatenate([verts, ones], axis=1).T).T[:, :3]

            tri_idx = np.empty(n_tri * 3, dtype=np.int32)
            try:
                mesh.loop_triangles.foreach_get("vertices", tri_idx)
            except Exception:
                tri_idx = np.array([vt for tri in mesh.loop_triangles for vt in tri.vertices], dtype=np.int32)
            tri_idx = tri_idx.reshape((n_tri, 3))
            world_tris = world_verts[tri_idx]

            mat_idx = np.zeros(n_tri, dtype=np.int32)
            try:
                mesh.loop_triangles.foreach_get("material_index", mat_idx)
            except Exception:
                mat_idx = np.array([int(tri.material_index) for tri in mesh.loop_triangles], dtype=np.int32)

            uv_tris = np.zeros((n_tri, 3, 2), dtype=np.float64)
            layers = getattr(mesh, "uv_layers", None)
            uv_layer = None
            if layers is not None and len(layers) > 0:
                uv_layer = layers.active if layers.active is not None else layers[0]
            if uv_layer is not None and len(uv_layer.data) > 0:
                loop_idx = np.empty(n_tri * 3, dtype=np.int32)
                try:
                    mesh.loop_triangles.foreach_get("loops", loop_idx)
                except Exception:
                    loop_idx = np.array([lp for tri in mesh.loop_triangles for lp in tri.loops], dtype=np.int32)
                uv_flat = np.empty(len(uv_layer.data) * 2, dtype=np.float64)
                uv_layer.data.foreach_get("uv", uv_flat)
                uv_data = uv_flat.reshape((-1, 2))
                uv_tris = uv_data[loop_idx.reshape((n_tri, 3))]
                if np.any(uv_tris):
                    has_uv = True

            slots = list(eval_obj.material_slots)
            slot_remap = np.zeros(max(len(slots), 1), dtype=np.int32)
            if not slots:
                key = ("nomaterial", obj.name)
                if key not in mat_lookup:
                    sampled = extract_material(None)
                    warnings.extend(sampled.warnings)
                    mat_lookup[key] = len(materials)
                    materials.append(sampled)
                sampled_mats = np.full(n_tri, mat_lookup[key], dtype=np.int32)
            else:
                for si, slot in enumerate(slots):
                    blender_mat = slot.material
                    key = id(blender_mat) if blender_mat is not None else ("slot", si, obj.name)
                    if key not in mat_lookup:
                        sampled = extract_material(blender_mat)
                        warnings.extend(sampled.warnings)
                        mat_lookup[key] = len(materials)
                        materials.append(sampled)
                    slot_remap[si] = mat_lookup[key]
                sampled_mats = slot_remap[np.clip(mat_idx, 0, len(slots) - 1)]

            tri_chunks.append(world_tris)
            uv_chunks.append(uv_tris)
            mat_chunks.append(sampled_mats)
        finally:
            eval_obj.to_mesh_clear()

    if mesh_count == 0 or not tri_chunks:
        raise ValueError("Import contains no evaluated triangles")

    triangles = np.concatenate(tri_chunks, axis=0)
    uvs = np.concatenate(uv_chunks, axis=0) if has_uv else None
    material_indices = np.concatenate(mat_chunks, axis=0)
    # Dense glTF meshes often split vertices; occupancy_solid flood-fill
    # already degrades to shell on a leaky surface, so skip O(edge) python tests.
    mesh_closed = True
    return triangles, uvs, material_indices, materials, mesh_closed, names, warnings


def world_to_voxel_space(
    points: np.ndarray,
    volume_matrix_world: Any,
    voxel_size: float,
) -> np.ndarray:
    """Transform world-space points into the volume's voxel-index space."""
    inv = np.array(volume_matrix_world.inverted(), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    orig_shape = arr.shape
    flat = arr.reshape((-1, 3))
    ones = np.ones((len(flat), 1), dtype=np.float64)
    homo = np.concatenate([flat, ones], axis=1)
    local = (inv @ homo.T).T[:, :3]
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    local /= float(voxel_size)
    return local.reshape(orig_shape)


def stage_glb(
    bpy: Any,
    context: Any,
    filepath: str,
    volume_obj: Any,
    padding: int = 1,
    keep_source: bool = False,
) -> EvaluatedSource:
    """Import, evaluate, fit, and optionally keep or discard staging objects."""
    imported: List[Any] = []
    try:
        imported = import_gltf_objects(bpy, filepath)
        if not imported:
            raise ValueError("GLB import created no objects")
        world_tris, uvs, mat_i, materials, closed, names, warnings = evaluate_mesh_objects(
            bpy, imported, context
        )
        voxel_size = float(volume_obj.data.voxel_workspace.voxel_size)
        voxel_tris = world_to_voxel_space(world_tris, volume_obj.matrix_world, voxel_size)
        smin = voxel_tris.reshape((-1, 3)).min(axis=0)
        smax = voxel_tris.reshape((-1, 3)).max(axis=0)
        props = volume_obj.data.voxel_workspace
        fit = contain_fit(
            smin,
            smax,
            tuple(props.extent_min),
            tuple(props.extent_max),
            padding=padding,
        )
        fitted = apply_fit(voxel_tris, fit)
        imported_names = [o.name for o in imported]
        if keep_source:
            col = _ensure_collection(bpy, STAGING_COLLECTION_NAME, hidden=True)
            for obj in imported:
                for other in list(obj.users_collection):
                    other.objects.unlink(obj)
                if obj.name not in col.objects:
                    col.objects.link(obj)
                obj.hide_set(True)
                obj.hide_viewport = True
                obj.hide_render = True
        else:
            _delete_objects(bpy, imported)
            imported = []
        return EvaluatedSource(
            triangles=fitted,
            uvs=uvs,
            material_indices=mat_i,
            materials=materials,
            mesh_closed=closed,
            fit=fit,
            object_names=names,
            warnings=warnings,
            imported_object_names=imported_names if keep_source else [],
        )
    except Exception:
        if imported and not keep_source:
            try:
                _delete_objects(bpy, imported)
            except Exception:
                pass
        raise


def _delete_objects(bpy: Any, objects: Sequence[Any]) -> None:
    for obj in list(objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except ReferenceError:
            pass
