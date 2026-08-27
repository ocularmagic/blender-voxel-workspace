"""Temporary GLB geometry analysis used by the import UI."""
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

import numpy as np

from .glb_materials import extract_material
from .glb_scene import _delete_objects, import_gltf_objects
from ..voxelization.palette_analysis import PaletteRecommendations, recommend_palette_sizes
from ..voxelization.resolution_analysis import ResolutionRecommendations, recommend_resolutions


@dataclass
class MeshObjectAnalysis:
    name: str
    triangle_count: int
    dimensions: Tuple[float, float, float]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    triangles: np.ndarray = field(repr=False)
    color_samples: np.ndarray = field(repr=False)
    included: bool = True
    primary: bool = False


@dataclass
class GLBAnalysis:
    filepath: str
    objects: List[MeshObjectAnalysis]
    recommendations: ResolutionRecommendations
    palette_recommendations: PaletteRecommendations

    @property
    def included_objects(self) -> List[MeshObjectAnalysis]:
        return [item for item in self.objects if item.included]

    @property
    def included_names(self) -> List[str]:
        return [item.name for item in self.included_objects]

    @property
    def triangles(self) -> np.ndarray:
        selected = [item.triangles for item in self.included_objects if len(item.triangles)]
        if not selected:
            return np.empty((0, 3, 3), dtype=np.float64)
        return np.concatenate(selected, axis=0)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        tris = self.triangles
        if not len(tris):
            raise ValueError("Select at least one mesh object")
        points = tris.reshape((-1, 3))
        return points.min(axis=0), points.max(axis=0)

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        mn, mx = self.bounds
        size = mx - mn
        return (float(size[0]), float(size[1]), float(size[2]))

    def refresh_recommendations(self) -> None:
        self.recommendations = recommend_resolutions(self.triangles)
        selected = [item.color_samples for item in self.included_objects if len(item.color_samples)]
        colors = np.concatenate(selected, axis=0) if selected else np.empty((0, 4), dtype=np.float32)
        self.palette_recommendations = recommend_palette_sizes(colors)


def _evaluated_object_triangles(obj: Any, depsgraph: Any) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        count = len(mesh.loop_triangles)
        if count == 0:
            return np.empty((0, 3, 3), dtype=np.float64)
        vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", vertices)
        vertices = vertices.reshape((-1, 3))
        ones = np.ones((len(vertices), 1), dtype=np.float64)
        world = np.asarray(evaluated.matrix_world, dtype=np.float64)
        world_vertices = (world @ np.concatenate((vertices, ones), axis=1).T).T[:, :3]
        indices = np.empty(count * 3, dtype=np.int32)
        try:
            mesh.loop_triangles.foreach_get("vertices", indices)
        except Exception:
            indices = np.asarray([v for tri in mesh.loop_triangles for v in tri.vertices], dtype=np.int32)
        return world_vertices[indices.reshape((-1, 3))]
    finally:
        evaluated.to_mesh_clear()


def _default_object_selection(items: Sequence[MeshObjectAnalysis]) -> None:
    if not items:
        return
    primary = max(items, key=lambda item: item.triangle_count)
    primary.primary = True
    p_dims = np.asarray(primary.dimensions, dtype=np.float64)
    primary_volume = float(np.prod(np.maximum(p_dims, 1e-12)))
    detail_floor = max(32, int(primary.triangle_count * 0.002))
    for item in items:
        if item is primary:
            item.included = True
            continue
        volume = float(np.prod(np.maximum(np.asarray(item.dimensions), 1e-12)))
        obvious_bounds_helper = item.triangle_count < detail_floor and volume > primary_volume * 2.0
        item.included = not obvious_bounds_helper


def _object_color_samples(obj: Any, maximum_per_image: int = 8192) -> np.ndarray:
    samples: List[np.ndarray] = []
    seen = set()
    for slot in getattr(obj, "material_slots", []):
        material = getattr(slot, "material", None)
        key = material.as_pointer() if material is not None and hasattr(material, "as_pointer") else id(material)
        if key in seen:
            continue
        seen.add(key)
        extracted = extract_material(material)
        samples.append(np.asarray(extracted.base_color, dtype=np.float32).reshape((1, 4)))
        if extracted.image is not None and extracted.image.size:
            pixels = np.asarray(extracted.image, dtype=np.float32).reshape((-1, 4))
            if len(pixels) > maximum_per_image:
                indices = np.linspace(0, len(pixels) - 1, maximum_per_image, dtype=np.int64)
                pixels = pixels[indices]
            samples.append(pixels)
    if not samples:
        samples.append(np.asarray([(0.8, 0.8, 0.8, 1.0)], dtype=np.float32))
    return np.concatenate(samples, axis=0)


def analyze_glb_file(bpy: Any, context: Any, filepath: str) -> GLBAnalysis:
    """Import a GLB temporarily and return evaluated per-object geometry analysis."""
    imported: List[Any] = []
    try:
        imported = import_gltf_objects(bpy, filepath)
        depsgraph = context.evaluated_depsgraph_get()
        rows: List[MeshObjectAnalysis] = []
        for obj in imported:
            if obj is None or getattr(obj, "type", "") != "MESH":
                continue
            triangles = _evaluated_object_triangles(obj, depsgraph)
            if not len(triangles):
                continue
            points = triangles.reshape((-1, 3))
            mn = points.min(axis=0)
            mx = points.max(axis=0)
            dims = mx - mn
            rows.append(
                MeshObjectAnalysis(
                    name=str(obj.name),
                    triangle_count=int(len(triangles)),
                    dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
                    bounds_min=(float(mn[0]), float(mn[1]), float(mn[2])),
                    bounds_max=(float(mx[0]), float(mx[1]), float(mx[2])),
                    triangles=triangles,
                    color_samples=_object_color_samples(obj),
                )
            )
        if not rows:
            raise ValueError("Import contains no evaluated mesh triangles")
        _default_object_selection(rows)
        selected = [row.triangles for row in rows if row.included]
        recommendations = recommend_resolutions(np.concatenate(selected, axis=0))
        selected_colors = [row.color_samples for row in rows if row.included and len(row.color_samples)]
        colors = np.concatenate(selected_colors, axis=0) if selected_colors else np.empty((0, 4), dtype=np.float32)
        palette_recommendations = recommend_palette_sizes(colors)
        return GLBAnalysis(
            filepath=filepath,
            objects=rows,
            recommendations=recommendations,
            palette_recommendations=palette_recommendations,
        )
    finally:
        if imported:
            _delete_objects(bpy, imported)
