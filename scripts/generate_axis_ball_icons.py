"""Generate crisp colored axis-ball icons for the instant-mirror buttons.

Per (axis, direction) the panel composes three elements:
  [filled colored ball with black letter] [arrow] [hollow colored ball]

This script renders each element as its own square PNG at high resolution:
  - ball_<axis>_filled.png   (red X / green Y / blue Z, black letter)
  - ball_<axis>_hollow.png
  - arrow_left.png / arrow_right.png

Style matches Blender's viewport navigation gizmo colors.
"""
import struct
import zlib
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "voxel_workspace" / "assets" / "toolbar"
SIZE = 64
SS = 4  # supersample factor

# Blender viewport gizmo axis colors.
AXIS_RGB = {
    "X": (0.9608, 0.2980, 0.2510),  # red
    "Y": (0.3137, 0.6902, 0.1490),  # green
    "Z": (0.1804, 0.4000, 0.9137),  # blue
}

# Letter glyphs on a 5x7 grid.
GLYPHS = {
    "X": ["#...#",
          "#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#",
          "#...#"],
    "Y": ["#...#",
          "#...#",
          ".#.#.",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "Z": ["#####",
          "....#",
          "...#.",
          "..#..",
          ".#...",
          "#....",
          "#####"],
}


def render_ball(axis: str, filled: bool) -> np.ndarray:
    """One colored ball centered in a square canvas."""
    big = SIZE * SS
    cy = big / 2.0
    margin = big * 0.08
    r = (big - 2 * margin) / 2

    rgb = AXIS_RGB[axis]
    color = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
    black = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)
    d = np.sqrt((xx - big / 2) ** 2 + (yy - cy) ** 2)

    ring_w = max(4 * SS, int(r * 0.14))
    img = np.zeros((big, big, 4), dtype=np.float32)

    if filled:
        img[d <= r] = color
    else:
        ring = (d <= r) & (d >= r - ring_w)
        img[ring] = color

    if filled:
        # Prominent black letter punched on top.
        g = GLYPHS[axis]
        cell = int((r * 1.45) / len(g))
        gh = len(g) * cell
        gw = len(g[0]) * cell
        oy = int(round(cy - gh / 2))
        ox = int(round(big / 2 - gw / 2))
        glyph = np.zeros((big, big), dtype=bool)
        for gy, row in enumerate(g):
            for gx, ch in enumerate(row):
                if ch == "#":
                    y0, x0 = oy + gy * cell, ox + gx * cell
                    glyph[y0:y0 + cell, x0:x0 + cell] = True
        img[glyph & (d <= r)] = black

    alpha = (img[:, :, 3] > 0).astype(np.float32)
    img[:, :, 3] = alpha
    out = img.reshape(SIZE, SS, SIZE, SS, 4).mean(axis=(1, 3))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def render_arrow(direction: str) -> np.ndarray:
    """A black right- or left-pointing arrow with a sharp triangular head."""
    big = SIZE * SS
    img = np.zeros((big, big, 4), dtype=np.float32)
    black = np.array([0.10, 0.10, 0.11, 1.0], dtype=np.float32)

    cy = big / 2
    # Sharp head: proper triangle from tail_x to tip_x.
    head_len = big * 0.30   # horizontal extent of the head
    head_half_h = big * 0.24  # half-height at the head's base
    shaft_half_h = big * 0.055
    shaft_len = big * 0.22

    yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)

    if direction == "RIGHT":
        base_x = big / 2 - head_len * 0.35
        tip_x = base_x + head_len
        # Head: |y-cy| shrinks linearly to exactly 0 at the tip -> sharp point.
        head = (
            (xx >= base_x) & (xx <= tip_x)
            & (np.abs(yy - cy) <= head_half_h * (tip_x - xx) / head_len)
        )
        shaft = (
            (np.abs(yy - cy) <= shaft_half_h)
            & (xx >= base_x - shaft_len) & (xx < base_x + 1)
        )
    else:
        base_x = big / 2 + head_len * 0.35
        tip_x = base_x - head_len
        head = (
            (xx <= base_x) & (xx >= tip_x)
            & (np.abs(yy - cy) <= head_half_h * (xx - tip_x) / head_len)
        )
        shaft = (
            (np.abs(yy - cy) <= shaft_half_h)
            & (xx <= base_x + shaft_len) & (xx > base_x - 1)
        )

    img[head | shaft] = black
    alpha = (img[:, :, 3] > 0).astype(np.float32)
    img[:, :, 3] = alpha
    out = img.reshape(SIZE, SS, SIZE, SS, 4).mean(axis=(1, 3))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def write_png(path: Path, rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for axis in ("X", "Y", "Z"):
        write_png(OUT_DIR / f"ball_{axis.lower()}_filled.png", render_ball(axis, True))
        write_png(OUT_DIR / f"ball_{axis.lower()}_hollow.png", render_ball(axis, False))
    write_png(OUT_DIR / "arrow_right.png", render_arrow("RIGHT"))
    write_png(OUT_DIR / "arrow_left.png", render_arrow("LEFT"))
    print(f"wrote ball/arrow icons to {OUT_DIR}")


if __name__ == "__main__":
    main()
