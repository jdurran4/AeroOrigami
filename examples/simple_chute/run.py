"""
Simple Chute — AeroOrigami pipeline driver
==========================================
Demonstrates Steps 1–3 for the flat rectangular membrane example.
Run from the AeroOrigami root directory:

    python examples/simple_chute/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyaeroori import load_mesh, load_creases, Region, remesh, build_surrogate, write_aeros, add_physics, N
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

mesh_file   = HERE / "simple_chute_mesh.fem"
crease_file = HERE / "simple_chute_creases.csv"

mesh_size          = 0.5     # Target element size for Gmsh remesh (Path A, metres)
penalty_stiffness  = 8e9
actuator_ramp_time = 3.0
min_radius         = 0.05
output_dir         = HERE / "results"

# =============================================================================
# STEP 1 — Load the original mesh
# =============================================================================

mesh = load_mesh(mesh_file)

print("=" * 50)
print("STEP 1 — Mesh")
print("=" * 50)
mesh_stats(mesh)

plot_mesh(mesh, title="Simple Chute — Mesh (Step 1)")

# =============================================================================
# STEP 2 — Load the crease pattern
# =============================================================================

creases = load_creases(crease_file)

print()
print("=" * 50)
print("STEP 2 — Crease Pattern")
print("=" * 50)
crease_stats(creases)
print()
check_crease_coverage(mesh, creases, tol=0.3)

plot_creases(creases, title="Simple Chute — Crease Pattern (Step 2)")
plot_creases_on_mesh(mesh, creases,
                     title="Simple Chute — Crease on Mesh (Step 2 alignment)")

# =============================================================================
# STEP 3 — Remesh
# =============================================================================
#
# PATH B (crease-as-mesh) — default for structured patterns.
# Every crease-segment endpoint is a mesh node, so every crease fold is
# guaranteed a driver-joint site in Step 4.  No Gmsh required.
# Viewer: matplotlib plot_mesh window.
#
# PATH A (Gmsh remesh) — uncomment the block below to use instead.
# Gmsh adds interior nodes along long crease segments; useful when you need
# fine sub-panel resolution or the surface is curved / non-structured.
# Run check_mesh_crease_resolution() after to verify actuation coverage.
# Viewer: Gmsh GUI.
# =============================================================================

print()
print("=" * 50)
print("STEP 3 — Remesh  [Path B: crease-as-mesh]")
print("=" * 50)

output_dir.mkdir(exist_ok=True)
region = Region(creases, name="simple_chute", use_crease_mesh=True)
coarse = remesh(mesh, region, show=True)

print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
      f"{len(coarse.elements)} elements, "
      f"{len(set(coarse.panel_map.values()))} panels")

# ── Path A alternative ───────────────────────────────────────────────────────
# region_a = Region(creases, mesh_size=mesh_size, name="simple_chute")
# coarse   = remesh(mesh, region_a,
#                   out_file=str(output_dir / "origami.msh"),
#                   show=True)
# print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
#       f"{len(coarse.elements)} elements, "
#       f"{len(set(coarse.panel_map.values()))} panels")
# print()
# check_mesh_crease_resolution(coarse, creases)
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
    creases,
    penalty_stiffness=penalty_stiffness,
    actuator_ramp_time=actuator_ramp_time,
    # vertex_joint_type=120,  # force spherical at all crease endpoints (research)
    # vertex_joint_type=126,  # force revolute at all crease endpoints (Path A research)
)

print(f"  Revolute joints : {len(surrogate.revolute_joints)}")
print(f"  Spherical joints: {len(surrogate.spherical_joints)}")

plot_surrogate_axes(surrogate, title="Simple Chute — Hinge Axes (Step 4)", arrow_length=0.5)

# =============================================================================
# STEP 5 — Add physics (BCs, loads, cables)
# =============================================================================
#
# NodeQuery resolves at runtime and prints matched nodes so you don't need
# a separate plotting run to identify node IDs.
#
# Simple chute has no cable elements in the mesh (all 2-node elements would
# be empty), but you can add explicit cable chains with {"points": [...]}.
#
# Uncomment and adjust coordinates to match your mesh:
#
# config = add_physics(
#     surrogate,
#     mesh=mesh,                        # required for all_bars cable detection
#     disp=[
#         # Pin top-edge nodes (z≈16, x=±0.8) — fix xyz translation
#         (N.along_line((-0.8, 0, 16), (0.8, 0, 16), tol=0.05), [1, 2, 3]),
#     ],
#     lmpc=[
#         {"type": "min_z", "z_min": -1.0},
#     ],
#     cables=[
#         # The mesh has 4 suspension lines (type-6 beams) in the TOPOLOGY section
#         # (no named blocks). all_bars detects them automatically and converts
#         # each 5-element chain into a single type-203 tension-only spring.
#         {"all_bars": True, "tol": 0.05},
#     ],
# )
# plot_physics(surrogate, config, title="Simple Chute — Physics Overview (Step 5)")

# =============================================================================
# STEP 6 — Write AEROS files
# =============================================================================

print()
print("=" * 50)
print("STEP 6 — Write AEROS files")
print("=" * 50)

# Without physics config (fold geometry only):
write_aeros(surrogate, output_dir=output_dir)

# With physics config (uncomment after configuring Step 5 above):
# write_aeros(surrogate, output_dir=output_dir, config=config)

print(f"Done. Files written to {output_dir}")

# =============================================================================
# STEP 7 — Post-fold displacement mapping
# =============================================================================

# disp_file  = HERE / "foldfiles/gdisplac6.xpost.5"
# fold_mesh  = output_dir / "mesh_modified.include"
# idisp_file = HERE / "foldfiles/IDISP6.include"
# mapped = pyaeroori.map_displacements(mesh, fold_mesh, disp_file, rbf_neighbors=100)
# pyaeroori.write_idisp6(mapped, idisp_file)
