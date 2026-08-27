"""Headless integration test: import a MagicaVoxel .vox as a new volume."""
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.blender.bootstrap import setup_test_environment  # noqa: E402

setup_test_environment()

import bpy  # noqa: E402
import voxel_workspace  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
voxel_workspace.register()

from voxel_workspace.importers.vox_import import (  # noqa: E402
    VoxParseError,
    parse_vox_bytes,
    srgb_bytes_to_linear,
)


def _chunk(chunk_id: bytes, content: bytes) -> bytes:
    return chunk_id + struct.pack("<II", len(content), 0) + content


def _build_vox(size, voxels, palette=None, version=150):
    """Build a minimal .vox byte stream."""
    data = b"VOX " + struct.pack("<I", version)
    children = b""
    children += _chunk(b"SIZE", struct.pack("<III", *size))
    xyzi = struct.pack("<I", len(voxels)) + b"".join(
        struct.pack("<BBBB", x, y, z, c) for (x, y, z, c) in voxels
    )
    children += _chunk(b"XYZI", xyzi)
    if palette is not None:
        rgba = b"".join(bytes(c) for c in palette)
        children += _chunk(b"RGBA", rgba)
    data += _chunk(b"MAIN", b"") + children
    return data


PALETTE = [(0, 0, 0, 0)] * 256
PALETTE[0] = (255, 0, 0, 255)     # vox index 1: red
PALETTE[1] = (0, 255, 0, 255)     # vox index 2: green
PALETTE[2] = (0, 0, 255, 0)       # vox index 3: transparent blue


def test_parser():
    vox = _build_vox(
        size=(4, 3, 2),
        voxels=[
            (0, 0, 0, 1),
            (1, 1, 1, 2),
            (2, 0, 0, 3),   # transparent color -> imported as empty
            (3, 2, 1, 1),
            (0, 0, 1, 0),   # index 0 = empty in .vox, skipped
        ],
        palette=PALETTE,
    )
    doc = parse_vox_bytes(vox)
    assert doc.version == 150
    assert len(doc.models) == 1
    model = doc.models[0]
    assert model.size == (4, 3, 2)
    assert model.voxels == {(0, 0, 0): 1, (1, 1, 1): 2, (2, 0, 0): 3, (3, 2, 1): 1}
    assert doc.used_color_indices() == [1, 2, 3]

    # sRGB -> linear conversion sanity
    lin = srgb_bytes_to_linear((255, 0, 0, 255))
    assert abs(lin[0] - 1.0) < 1e-6 and lin[1] == 0.0 and abs(lin[3] - 1.0) < 1e-6
    mid = srgb_bytes_to_linear((128, 128, 128, 255))
    assert 0.2 < mid[0] < 0.22  # 128 sRGB ~ 0.216 linear


def test_parser_rejects_raw_version():
    try:
        parse_vox_bytes(_build_vox(size=(1, 1, 1), voxels=[], version=200))
    except VoxParseError:
        pass
    else:
        raise AssertionError("version 200 should be rejected")


def test_parser_rejects_bad_magic():
    try:
        parse_vox_bytes(b"NOPE" + b"\x00" * 8)
    except VoxParseError:
        pass
    else:
        raise AssertionError("bad magic should be rejected")


def test_operator_import():
    vox = _build_vox(
        size=(4, 3, 2),
        voxels=[
            (0, 0, 0, 1),
            (1, 1, 1, 2),
            (2, 0, 0, 3),   # transparent -> empty
            (3, 2, 1, 1),
        ],
        palette=PALETTE,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.vox"
        path.write_bytes(vox)

        before = set(bpy.data.objects.keys())
        result = bpy.ops.voxel.import_vox(filepath=str(path))
        assert result == {"FINISHED"}, result

        # A new Voxel Root appeared.
        new_names = set(bpy.data.objects.keys()) - before
        assert "Voxel Root" in new_names, new_names

        from voxel_workspace.blender.object_graph import resolve_volume_context
        v_ctx = resolve_volume_context(bpy.context)
        assert v_ctx is not None and v_ctx.mesh is not None
        mesh = v_ctx.mesh
        props = mesh.voxel_workspace
        emin = tuple(int(v) for v in props.extent_min)
        emax = tuple(int(v) for v in props.extent_max)
        assert (emax[0] - emin[0], emax[1] - emin[1], emax[2] - emin[2]) == (4, 3, 2)
        assert emin[2] == 0  # bottom layer at z=0, centered X/Y convention

        from voxel_workspace.blender.runtime import get_or_load
        entry = get_or_load(mesh)
        assert entry is not None
        grid = entry.grid

        def cell(x, y, z):
            return grid.get_cell((emin[0] + x, emin[1] + y, emin[2] + z))

        # (0,0,0) -> color index 1 (SURFACE domain)
        c = cell(0, 0, 0)
        assert c.index == 1 and int(c.domain) == 1, c
        # (1,1,1) -> color index 2
        c = cell(1, 1, 1)
        assert c.index == 2 and int(c.domain) == 1, c
        # (3,2,1) -> color index 1
        c = cell(3, 2, 1)
        assert c.index == 1, c
        # (2,0,0) transparent -> empty
        c = cell(2, 0, 0)
        assert c.index == 0, c

        # Fresh palette: exactly indices 0..2 (transparent 3 skipped).
        indices = sorted(int(e.index) for e in props.surface_palette)
        assert indices == [0, 1, 2], indices
        red_entry = next(e for e in props.surface_palette if int(e.index) == 1)
        assert red_entry.color[0] > 0.99 and red_entry.color[1] < 1e-6
        green_entry = next(e for e in props.surface_palette if int(e.index) == 2)
        assert green_entry.color[1] > 0.99

        # Occupancy: 3 written voxels -> mesh has faces.
        assert len(mesh.polygons) > 0


def test_operator_rejects_invalid_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.vox"
        path.write_bytes(b"not a vox file")
        try:
            bpy.ops.voxel.import_vox(filepath=str(path))
        except RuntimeError as exc:
            assert "VOX import failed" in str(exc), exc
        else:
            raise AssertionError("invalid file should be rejected")


def main():
    test_parser()
    test_parser_rejects_raw_version()
    test_parser_rejects_bad_magic()
    test_operator_import()
    test_operator_rejects_invalid_file()
    print("ALL VOX IMPORT TESTS PASSED")


if __name__ == "__main__":
    main()
