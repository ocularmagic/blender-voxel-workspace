"""Build a Blender extension zip from the voxel_workspace/ source tree.

The zip is a flattened copy of voxel_workspace/ (blender_manifest.toml at the
archive root), excluding __pycache__ and *.pyc. Deterministic entry ordering
and timestamps so re-running yields a stable archive.
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "voxel_workspace")
OUT = os.path.join(ROOT, "dist", "voxel_workspace-0.2.15.zip")

# Fixed timestamp keeps the archive reproducible.
FIXED_DT = (2026, 8, 22, 8, 45, 0)


def iter_files(base):
    for dirpath, dirnames, filenames in os.walk(base):
        # Skip caches so the archive only carries source.
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            arc = os.path.relpath(full, base)
            yield full, arc


def main():
    files = sorted(iter_files(SRC), key=lambda t: t[1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for full, arc in files:
            with open(full, "rb") as fh:
                data = fh.read()
            info = zipfile.ZipInfo(arc, date_time=FIXED_DT)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    print(f"wrote {OUT}")
    print(f"entries: {len(files)}")
    for full, arc in files:
        print("  " + arc)


if __name__ == "__main__":
    sys.exit(main())
