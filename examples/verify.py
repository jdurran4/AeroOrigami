"""
Steps 1 & 2 verification — run this to visually check mesh loading and crease
pattern alignment for all examples.

Run from the AeroOrigami root directory:
    python examples/verify.py

Or open in VSCode and press the Run (▷) button.

Each example produces three plots:
  1. Mesh node scatter + wireframe (Step 1)
  2. Crease pattern alone (Step 2)
  3. Crease pattern overlaid on mesh (Step 2 alignment check)

The console also prints node/element counts, coordinate extents, and a
nearest-node proximity check for all crease endpoints.

Before running for the first time, generate the DGB crease CSVs:
    python examples/dgb_parachute/convert_alexandra_creases.py
"""

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyaeroori import load_mesh, load_creases
from pyaeroori.plot import (
    mesh_stats,
    crease_stats,
    plot_mesh,
    plot_creases,
    plot_creases_on_mesh,
    check_crease_coverage,
)

EXAMPLES = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Simple chute
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SIMPLE CHUTE")
print("=" * 60)

simple_mesh    = load_mesh(EXAMPLES / "simple_chute/simple_chute_mesh.fem")
simple_creases = load_creases(EXAMPLES / "simple_chute/simple_chute_creases.csv")

mesh_stats(simple_mesh)
print()
crease_stats(simple_creases)
print()
# tol=0.3: the simple chute mesh element size is ~0.3m, so crease endpoints
# won't land exactly on nodes — we're checking they're within the mesh region.
check_crease_coverage(simple_mesh, simple_creases, tol=0.3)
print()

plot_mesh(simple_mesh, title="Simple Chute — Mesh (Step 1)")
plot_creases(simple_creases, title="Simple Chute — Crease Pattern (Step 2)")
plot_creases_on_mesh(simple_mesh, simple_creases,
                     title="Simple Chute — Crease on Mesh (Step 2 alignment)")

# ─────────────────────────────────────────────────────────────────────────────
# DGB parachute
# ─────────────────────────────────────────────────────────────────────────────

disk_csv = EXAMPLES / "dgb_parachute/dgb_disk_creases.csv"
band_csv = EXAMPLES / "dgb_parachute/dgb_band_creases.csv"

if not disk_csv.exists() or not band_csv.exists():
    print("=" * 60)
    print("DGB PARACHUTE — skipped")
    print("  Run first:  python examples/dgb_parachute/convert_alexandra_creases.py")
    print("=" * 60)
else:
    print("=" * 60)
    print("DGB PARACHUTE")
    print("=" * 60)

    dgb_mesh    = load_mesh(EXAMPLES / "dgb_parachute/dgb_mesh.fem")
    dgb_creases = load_creases(disk_csv, band_csv)

    mesh_stats(dgb_mesh)
    print()
    crease_stats(dgb_creases)
    print()
    # tol=0.1: crease nodes are embedded in the disk/band mesh, so most
    # endpoints should land close to existing mesh nodes.
    check_crease_coverage(dgb_mesh, dgb_creases, tol=0.1)
    print()

    plot_mesh(dgb_mesh, title="DGB Parachute — Mesh (Step 1)")
    plot_creases(dgb_creases, title="DGB Parachute — Crease Pattern (Step 2)")
    plot_creases_on_mesh(dgb_mesh, dgb_creases,
                         title="DGB Parachute — Crease on Mesh (Step 2 alignment)")
