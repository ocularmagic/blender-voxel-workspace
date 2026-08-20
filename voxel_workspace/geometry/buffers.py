from dataclasses import dataclass
import numpy as np


@dataclass
class MeshBuffers:
    """CPU buffers ready for Blender mesh foreach_set or GPU VBO uploads."""
    positions: np.ndarray        # float32, (N, 3) where N = quad_count * 4
    indices: np.ndarray          # int32, (T, 3) where T = quad_count * 2
    palette_indices: np.ndarray  # int32, (N,) one value per corner/vertex
    quad_count: int
