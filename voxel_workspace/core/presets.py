"""Palette preset data structures, sRGB/Linear conversions, and JSON interchange."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PRESET_SCHEMA_VERSION = 3


def linear_to_srgb_byte(val: float) -> int:
    """Convert a linear float component [0.0..1.0] to an sRGB byte [0..255]."""
    clamped = max(0.0, min(1.0, float(val)))
    if clamped <= 0.0031308:
        srgb = clamped * 12.92
    else:
        srgb = 1.055 * (clamped ** (1.0 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, srgb)) * 255))


def srgb_byte_to_linear(val: int) -> float:
    """Convert an sRGB byte [0..255] to a linear float [0.0..1.0]."""
    c = max(0.0, min(255.0, float(val))) / 255.0
    if c <= 0.04045:
        return c / 12.92
    else:
        return float(((c + 0.055) / 1.055) ** 2.4)


def rgba_linear_to_srgb_bytes(rgba: Tuple[float, float, float, float]) -> List[int]:
    """Convert (R, G, B, A) linear floats to [R, G, B, A] sRGB/alpha bytes (0..255)."""
    return [
        linear_to_srgb_byte(rgba[0]),
        linear_to_srgb_byte(rgba[1]),
        linear_to_srgb_byte(rgba[2]),
        int(round(max(0.0, min(1.0, float(rgba[3]))) * 255)),
    ]


def rgba_srgb_bytes_to_linear(rgba_bytes: List[int]) -> Tuple[float, float, float, float]:
    """Convert [R, G, B, A] sRGB/alpha bytes (0..255) to (R, G, B, A) linear floats."""
    r = srgb_byte_to_linear(rgba_bytes[0]) if len(rgba_bytes) > 0 else 0.0
    g = srgb_byte_to_linear(rgba_bytes[1]) if len(rgba_bytes) > 1 else 0.0
    b = srgb_byte_to_linear(rgba_bytes[2]) if len(rgba_bytes) > 2 else 0.0
    a = float(rgba_bytes[3]) / 255.0 if len(rgba_bytes) > 3 else 1.0
    return (r, g, b, a)


@dataclass
class PalettePresetEntry:
    name: str
    color_srgb: List[int]  # [r, g, b, a] in sRGB bytes
    domain: str = "SURFACE"  # "SURFACE" or "VOLUME"


@dataclass
class PalettePreset:
    name: str
    schema_version: int
    color_space: str
    colors: List[PalettePresetEntry]
    palette_type: str = "SURFACE"  # "SURFACE" or "VOLUME"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "color_space": self.color_space,
            "palette_type": self.palette_type,
            "colors": [
                {
                    "name": c.name,
                    "color": c.color_srgb,
                    "domain": c.domain,
                }
                for c in self.colors
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PalettePreset":
        name = str(data.get("name", "Unnamed Preset"))
        schema_version = int(data.get("schema_version", 1))
        color_space = str(data.get("color_space", "sRGB"))
        pal_type = str(data.get("palette_type", "")).upper()
        if schema_version >= 3 and pal_type not in {"SURFACE", "VOLUME"}:
            raise ValueError("schema-3 preset palette_type must be SURFACE or VOLUME")
        raw_colors = data.get("colors", [])
        colors = []
        for item in raw_colors:
            c_name = str(item.get("name", ""))
            c_val = list(item.get("color", [0, 0, 0, 255]))
            c_dom = str(item.get("domain", "")).upper()
            if not c_dom:
                c_dom = pal_type if pal_type in {"SURFACE", "VOLUME"} else "SURFACE"
            if c_dom not in {"SURFACE", "VOLUME"}:
                c_dom = "SURFACE"
            # Ensure 4 components
            while len(c_val) < 4:
                c_val.append(255)
            colors.append(PalettePresetEntry(name=c_name, color_srgb=c_val[:4], domain=c_dom))

        if not pal_type:
            # Infer from entries if legacy
            if colors and all(c.domain == "VOLUME" for c in colors):
                pal_type = "VOLUME"
            else:
                pal_type = "SURFACE"

        if schema_version >= 3 and any(color.domain != pal_type for color in colors):
            raise ValueError("schema-3 preset entry domain must match palette_type")

        return cls(
            name=name,
            schema_version=schema_version,
            color_space=color_space,
            colors=colors,
            palette_type=pal_type,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PalettePreset":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "PalettePreset":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def find_nearest_palette_index(
    target_rgba: Tuple[float, float, float, float],
    candidate_indices_and_colors: List[Tuple[int, Tuple[float, float, float, float]]],
) -> int:
    """Find the candidate palette index whose linear RGB is closest to target_rgba (Euclidean in linear RGB)."""
    if not candidate_indices_and_colors:
        return 1

    best_idx = candidate_indices_and_colors[0][0]
    best_dist = float("inf")
    tr, tg, tb = target_rgba[:3]

    for idx, (cr, cg, cb, *_) in candidate_indices_and_colors:
        dist = (tr - cr) ** 2 + (tg - cg) ** 2 + (tb - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


# Built-in lightweight presets (minimal set)
BUILTIN_PRESETS: Dict[str, PalettePreset] = {
    "PICO-8": PalettePreset(
        name="PICO-8",
        schema_version=1,
        color_space="sRGB",
        colors=[
            PalettePresetEntry(name="Black", color_srgb=[0, 0, 0, 255]),
            PalettePresetEntry(name="Dark Blue", color_srgb=[29, 43, 83, 255]),
            PalettePresetEntry(name="Dark Purple", color_srgb=[126, 37, 83, 255]),
            PalettePresetEntry(name="Dark Green", color_srgb=[0, 135, 81, 255]),
            PalettePresetEntry(name="Brown", color_srgb=[171, 82, 54, 255]),
            PalettePresetEntry(name="Dark Gray", color_srgb=[95, 87, 79, 255]),
            PalettePresetEntry(name="Light Gray", color_srgb=[194, 195, 199, 255]),
            PalettePresetEntry(name="White", color_srgb=[255, 241, 232, 255]),
            PalettePresetEntry(name="Red", color_srgb=[255, 0, 77, 255]),
            PalettePresetEntry(name="Orange", color_srgb=[255, 163, 0, 255]),
            PalettePresetEntry(name="Yellow", color_srgb=[255, 236, 39, 255]),
            PalettePresetEntry(name="Green", color_srgb=[0, 228, 54, 255]),
            PalettePresetEntry(name="Blue", color_srgb=[41, 173, 255, 255]),
            PalettePresetEntry(name="Lavender", color_srgb=[131, 118, 156, 255]),
            PalettePresetEntry(name="Pink", color_srgb=[255, 119, 168, 255]),
            PalettePresetEntry(name="Peach", color_srgb=[255, 204, 170, 255]),
        ],
    ),
    "GameBoy": PalettePreset(
        name="GameBoy",
        schema_version=1,
        color_space="sRGB",
        colors=[
            PalettePresetEntry(name="Darkest Green", color_srgb=[15, 56, 15, 255]),
            PalettePresetEntry(name="Dark Green", color_srgb=[48, 98, 48, 255]),
            PalettePresetEntry(name="Light Green", color_srgb=[139, 172, 15, 255]),
            PalettePresetEntry(name="Lightest Green", color_srgb=[155, 188, 15, 255]),
        ],
    ),
    "Endesga 8": PalettePreset(
        name="Endesga 8",
        schema_version=1,
        color_space="sRGB",
        colors=[
            PalettePresetEntry(name="Void", color_srgb=[25, 20, 28, 255]),
            PalettePresetEntry(name="Blood", color_srgb=[143, 44, 52, 255]),
            PalettePresetEntry(name="Rust", color_srgb=[224, 111, 74, 255]),
            PalettePresetEntry(name="Gold", color_srgb=[247, 214, 111, 255]),
            PalettePresetEntry(name="Moss", color_srgb=[92, 148, 56, 255]),
            PalettePresetEntry(name="Ocean", color_srgb=[44, 91, 143, 255]),
            PalettePresetEntry(name="Sky", color_srgb=[107, 186, 224, 255]),
            PalettePresetEntry(name="Cloud", color_srgb=[238, 245, 247, 255]),
        ],
    ),
}
