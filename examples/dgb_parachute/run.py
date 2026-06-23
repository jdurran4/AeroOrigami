"""
DGB Parachute — AeroOrigami pipeline driver
============================================
Demonstrates Steps 1–6 for the DGB parachute with Miura-Ori creases.
Run from the AeroOrigami root directory:

    python examples/dgb_parachute/run.py

Before the first run, generate the crease CSV files from Alexandra's data:
    python examples/dgb_parachute/convert_alexandra_creases.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyaeroori import load_mesh, load_creases, Region, remesh, build_surrogate, write_aeros, SimConfig, add_physics, N
from pyaeroori.plot import (
    mesh_stats,
    crease_stats,
    plot_mesh,
    plot_creases,
    plot_creases_on_mesh,
    check_crease_coverage,
    check_mesh_crease_resolution,
    plot_surrogate_axes,
    plot_physics,
)

HERE = Path(__file__).parent

# =============================================================================
# CONFIGURATION
# =============================================================================

mesh_file        = HERE / "dgb_mesh.fem"
disk_crease_file = HERE / "dgb_disk_creases.csv"
band_crease_file = HERE / "dgb_band_creases.csv"

mesh_size          = 0.22    # Target element size for Gmsh remesh (Path A, meters)
penalty_stiffness  = 8e9
actuator_ramp_time = 3.0
min_radius         = 0.05
include_cables     = True
output_dir         = HERE / "sim_files"

# =============================================================================
# STEP 1 — Load the original mesh
# =============================================================================

mesh = load_mesh(mesh_file)

print("=" * 50)
print("STEP 1 — Mesh")
print("=" * 50)
mesh_stats(mesh)

plot_mesh(mesh, title="DGB Parachute — Mesh (Step 1)")

# =============================================================================
# STEP 2 — Load the crease pattern
# =============================================================================

# To convert Alexandra's original CSVs first:
#   python examples/dgb_parachute/convert_alexandra_creases.py

disk_creases = load_creases(disk_crease_file)
band_creases = load_creases(band_crease_file)
all_creases  = load_creases(disk_crease_file, band_crease_file)

print()
print("=" * 50)
print("STEP 2 — Crease Pattern")
print("=" * 50)
print("Disk:")
crease_stats(disk_creases)
print("Band:")
crease_stats(band_creases)
print()
check_crease_coverage(mesh, all_creases, tol=0.1)

plot_creases(all_creases, title="DGB Parachute — Crease Pattern (Step 2)")
plot_creases_on_mesh(mesh, all_creases,
                     title="DGB Parachute — Crease on Mesh (Step 2 alignment)")

# =============================================================================
# STEP 3 — Remesh
# =============================================================================
#
# PATH B (crease-as-mesh) — default for structured patterns.
# Disk and band share a snap map so their common rim nodes stitch.
# Every crease vertex is a mesh node — full actuation coverage guaranteed.
# Viewer: matplotlib plot_mesh window.
#
# PATH A (Gmsh remesh) — uncomment the block below to use instead.
# Processes disk (planar) and band (cylindrical) in one Gmsh session.
# Run check_mesh_crease_resolution() after to verify actuation coverage.
# Viewer: Gmsh GUI.
# =============================================================================

print()
print("=" * 50)
print("STEP 3 — Remesh  [Path B: crease-as-mesh]")
print("=" * 50)

output_dir.mkdir(exist_ok=True)
disk_region = Region(disk_creases, name="disk", use_crease_mesh=True)
band_region = Region(band_creases, name="band", use_crease_mesh=True)
coarse      = remesh(mesh, disk_region, band_region, show=True)

# print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
#       f"{len(coarse.elements)} elements, "
#       f"{len(set(coarse.panel_map.values()))} panels")

# ── Path A alternative ───────────────────────────────────────────────────────
# disk_region_a = Region(disk_creases, mesh_size=mesh_size, name="disk")
# band_region_a = Region(band_creases, mesh_size=mesh_size, name="band")
# coarse        = remesh(mesh, disk_region_a, band_region_a,
#                        out_file=str(output_dir / "origami.msh"),
#                        show=True)
# print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
#       f"{len(coarse.elements)} elements, "
#       f"{len(set(coarse.panel_map.values()))} panels")
# print()
# check_mesh_crease_resolution(coarse, all_creases)
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# STEP 4 — Build surrogate (node duplication + driver joints)
# =============================================================================

print()
print("=" * 50)
print("STEP 4 — Build surrogate")
print("=" * 50)

surrogate = build_surrogate(
    coarse,
    all_creases,
    penalty_stiffness=penalty_stiffness,
    actuator_ramp_time=actuator_ramp_time,
    # vertex_joint_type=120,  # force spherical at all crease endpoints (research)
    # vertex_joint_type=126,  # force revolute at all crease endpoints (Path A research)
)

# print(f"  Revolute joints : {len(surrogate.revolute_joints)}")
# print(f"  Spherical joints: {len(surrogate.spherical_joints)}")

# plot_surrogate_axes(surrogate, title="DGB Parachute — Hinge Axes (Step 4)",arrow_length=0.1)

# =============================================================================
# STEP 5 — Add physics (BCs, loads, cables)
# =============================================================================

# NodeQuery resolves at runtime and prints matched nodes — no separate
# node-ID lookup run needed.

# For DGB cables: either name the blocks explicitly, or use all_bars=True
# to automatically pick up every 2-node element in the original mesh.

config = add_physics(
    surrogate,
    mesh=mesh,                          # required for block= and all_bars= lookups
    disp=[
        # Triple bridle: pin all DOF to keep connection to payload
        (N.near(x=-0.146,y=-0.253,z=-0.738,tol=0.1), [1, 2, 3, 4, 5, 6]),
        (N.near(x=0.292,y=0.0,z=-0.738,tol=0.05), [1, 2, 3, 4, 5, 6]),
        (N.near(x=-0.146,y=0.253,z=-0.738,tol=0.05), [1, 2, 3, 4, 5, 6]),
    ],
    lmpc=[
        {"type": "min_z",      "z_min":  46.7, "nodes": N.above(z=46.8)},
        {"type": "min_radius", "r_min":   0.1},
    ],
    cables=[
        # Use named blocks (recommended for DGB — avoids beam elements
        # embedded in the canopy surface).  Each chain of bar elements
        # in the block becomes a single type-203 tension-only spring.
        {"blocks": ["Suspension_Lines", "Vent_Lines", "Gap_Lines", "Riser_Line", "TripleBridle_Lines"], "tol": 0.05},
        # Alternative — one block at a time:
        # {"block": "Suspension_Lines"},
    ],
)
plot_physics(surrogate, config, title="DGB Parachute — Physics Overview (Step 5)", arrow_length=0.1)

# =============================================================================
# STEP 6 — Write AEROS files
# =============================================================================

print()
print("=" * 50)
print("STEP 6 — Write AEROS files")
print("=" * 50)

# Physics + simulation config:
sim = SimConfig(
    project_name    = "DGB_Parachute",
    sim_name        = "dgb_fold",
    end_time        = 14.0,
    shell_E         = 1e7,
    shell_nu        = 0.4,
    shell_rho       = 40000.0,
    shell_t         = 1.0,
    cable_stiffness = 100000.0,
    a_damp          = 1e-7,
    b_damp          = 2.0,
)
write_aeros(surrogate, output_dir=output_dir, config=config, sim=sim)

print(f"Done. Files written to {output_dir}")

# =============================================================================
# STEP 7 — Post-fold displacement mapping
# =============================================================================

# disp_file  = HERE / "foldfiles/gdisplac6.xpost.5"
# fold_mesh  = output_dir / "ORIGAMI_MESH.include"
# idisp_file = HERE / "foldfiles/IDISP6.include"
# mapped = pyaeroori.map_displacements(mesh, fold_mesh, disp_file, rbf_neighbors=100)
# pyaeroori.write_idisp6(mapped, idisp_file)
