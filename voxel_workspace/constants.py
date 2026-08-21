from typing import Tuple

BRICK_SIZE: int = 32
EMPTY: int = 0

BrickCoord = Tuple[int, int, int]
VoxelCoord = Tuple[int, int, int]

# 256x1 lookup image default colors; indices 1-8 are the built-in MVP colors.
# Verbatim linear floats (do not retype to sRGB bytes).
DEFAULT_PALETTE: Tuple[Tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 0.0, 0.0),      # 0: Empty / Background (reserved transparent black)
    (0.5, 0.5, 0.5, 1.0),      # 1: Neutral Gray (default)
    (1.0, 0.03, 0.03, 1.0),    # 2: Red
    (0.03, 1.0, 0.03, 1.0),    # 3: Green
    (0.03, 0.15, 1.0, 1.0),    # 4: Blue
    (1.0, 0.8, 0.03, 1.0),     # 5: Yellow
    (0.8, 0.03, 1.0, 1.0),     # 6: Magenta
    (0.03, 1.0, 1.0, 1.0),     # 7: Cyan
    (1.0, 0.3, 0.03, 1.0),     # 8: Orange
)
