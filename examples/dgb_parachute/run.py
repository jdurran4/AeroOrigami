"""
DGB Parachute — AeroOrigami pipeline driver
============================================
Runs the full 7-step AeroOrigami pipeline for the DGB parachute with Miura-Ori creases.

FIRST-TIME SETUP
----------------
1. Generate crease CSVs from Alexandra's raw data (one-time):
       python examples/dgb_parachute/convert_alexandra_creases.py

2. Run this script to generate AERO-S fold simulation input files (Steps 1–6)
   and cache the pipeline state for fast Step 7 iteration:
       python examples/dgb_parachute/run.py

3. Sync to the cluster and run the fold simulation:
       ./sync_to_cluster.sh examples/dgb_parachute dgb_v1
       # on cluster: cd /scratch/.../dgb_v1 && sbatch run.sbatch

4. After the fold simulation finishes, copy gdisplac6.xpost back to sim_files/,
   then re-run this script (Step 7 runs automatically if the file is present).
   To iterate on mapping settings without re-running Steps 1–6 (~30 s), use:
       python examples/dgb_parachute/step7.py   # ~1 s, loads cached state

Run all commands from the AeroOrigami root directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyaeroori import (
    load_mesh, load_creases, Region, remesh, build_surrogate, write_aeros, SimConfig,
    add_physics, N,
    map_displacements, write_idisp6, write_folded_vtk,
)
from pyaeroori.plot import (
    mesh_stats,
    crease_stats,
    plot_mesh,
    plot_creases,
    plot_creases_on_mesh,
    check_crease_coverage,
    check_mesh_crease_resolution,
    plot_panel_colors,
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

mesh_size          = 0.35    # Target element size for Gmsh remesh (Path A, meters)
penalty_stiffness  = 8e8
actuator_ramp_time = 3.0
min_radius         = 0.1
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

# plot_mesh(mesh, title="DGB Parachute — Mesh (Step 1)")

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

# plot_creases(all_creases, title="DGB Parachute — Crease Pattern (Step 2)")
# plot_creases_on_mesh(mesh, all_creases,
#                      title="DGB Parachute — Crease on Mesh (Step 2 alignment)")

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

# print()
# print("=" * 50)
# print("STEP 3 — Remesh  [Path B: crease-as-mesh]")
# print("=" * 50)

# output_dir.mkdir(exist_ok=True)
# disk_region = Region(disk_creases, name="disk", use_crease_mesh=True,
#                      outward_normal=(0, 0, 1))  # disk is flat; auto-detect fails for planar surfaces
# band_region = Region(band_creases, name="band", use_crease_mesh=True)
#                      # band is cylindrical — outward normal auto-detected from geometry
# coarse      = remesh(mesh, disk_region, band_region, show=True)

# print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
#       f"{len(coarse.elements)} elements, "
#       f"{len(set(coarse.panel_map.values()))} panels")

# ── Path A alternative ───────────────────────────────────────────────────────
disk_region_a = Region(disk_creases, mesh_size=mesh_size, name="disk")
band_region_a = Region(band_creases, mesh_size=mesh_size, name="band")
coarse        = remesh(mesh, disk_region_a, band_region_a,
                       out_file=str(output_dir / "origami.msh"),
                       show=True)
print(f"  Coarse mesh : {len(coarse.nodes)} nodes, "
      f"{len(coarse.elements)} elements, "
      f"{len(set(coarse.panel_map.values()))} panels")
print()
check_mesh_crease_resolution(coarse, all_creases)
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# STEP 4 — Build surrogate (node duplication + driver joints)
# =============================================================================f

print()
print("=" * 50)
print("STEP 4 — Build surrogate")
print("=" * 50)

surrogate = build_surrogate(
    coarse,
    all_creases,
    penalty_stiffness=penalty_stiffness,
    actuator_ramp_time=actuator_ramp_time,
    fold_fraction=-0.99,
    split_quads=False
    # vertex_joint_type=120,  # force spherical at all crease endpoints (research)
    # vertex_joint_type=126,  # force revolute at all crease endpoints (Path A research)
)

# print(f"  Revolute joints : {len(surrogate.revolute_joints)}")
# print(f"  Spherical joints: {len(surrogate.spherical_joints)}")

# plot_panel_colors(coarse, surrogate, title="DGB Parachute — Panel 2-Coloring (Step 4)")
# plot_surrogate_axes(surrogate, title="DGB Parachute — Hinge Axes (Step 4)", arrow_length=0.1)

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
        {"type": "min_radius", "r_min":   min_radius}, # Defaults to use all canopy nodes
        {"type": "radial_motion", "delta": 0.05}
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
# plot_physics(surrogate, config, title="DGB Parachute — Physics Overview (Step 5)", arrow_length=0.1)

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
    end_time        = 16.0,
    shell_E         = 1e7,
    shell_nu        = 0.4,
    shell_rho       = 40000.0,
    shell_t         = 0.2,
    cable_stiffness = 10000.0,
    a_damp          = 1e-7,
    b_damp          = 5.0,
    time_step       = 8e-5,
)
write_aeros(surrogate, output_dir=output_dir, config=config, sim=sim, beta_factor=1.0)

print(f"Done. Files written to {output_dir}")

# Cache pipeline state so step7.py can reload it without re-running Steps 1–6.
# The pickle contains surrogate, mesh, and config — everything Step 7 needs.
import pickle
with open(output_dir / "pipeline_state.pkl", "wb") as _f:
    pickle.dump({"surrogate": surrogate, "mesh": mesh, "config": config}, _f)
print(f"  Pipeline state cached → {output_dir / 'pipeline_state.pkl'}")
print(f"  To iterate on Step 7 without re-running Steps 1–6, use: python examples/dgb_parachute/step7.py")

# =============================================================================
# STEP 7 — Post-fold displacement mapping
# =============================================================================
# Maps folded displacements from the coarse surrogate back onto the original
# fine mesh so the FSI simulation starts from the folded configuration.
#
# This step runs automatically if gdisplac6.xpost is present in sim_files/.
# Copy it back from the cluster after the fold simulation completes.
#
# Outputs:
#   IDISP6.include       — 6-DOF initial displacements for the fine FSI mesh
#   folded_fine_mesh.vtk — deformed fine mesh for ParaView quality check
#
# To iterate on mapping settings without re-running Steps 1–6, edit
# examples/dgb_parachute/step7.py and run that instead (~1 s vs ~30 s here).
# =============================================================================

disp_file  = output_dir / "gdisplac6.xpost"
idisp_file = output_dir / "IDISP6.include"
vtk_file   = output_dir / "folded_fine_mesh.vtk"

if disp_file.exists():
    print()
    print("=" * 50)
    print("STEP 7 — Displacement mapping")
    print("=" * 50)

    # Two mapping methods available — swap method= to compare:
    #
    #   "rbf"         Multiquadric RBF across all coarse nodes. Smooth and
    #                 global, but can blur across fold lines near the vent
    #                 where many panels converge. More rbf_neighbors = smoother.
    #
    #   "panel_rigid" Per-panel Procrustes rigid-body transform (Kabsch SVD).
    #                 Correct for large-angle folds; no cross-panel averaging.
    #                 Best near the vent / high-curvature regions.
    #
    # Cable intermediate nodes are always reconstructed via BFS arc-length
    # interpolation regardless of method.
    displacements = map_displacements(
        surrogate,
        mesh,
        disp_file,
        config=config,             # carries cable endpoint nodes
        step=-1,                   # use last time step in xpost
        method="rbf",
        rbf_neighbors=50,          # only used by method="rbf"
        rbf_smoothing=1e-6,
    )

    write_idisp6(displacements, mesh, idisp_file)
    write_folded_vtk(mesh, displacements, vtk_file)
    print("  Open folded_fine_mesh.vtk in ParaView to visually verify the mapping.")
    print("  Displacement vectors are stored as VECTORS 'displacement' on each node.")
else:
    print()
    print(f"Step 7 skipped — {disp_file} not found.")
    print("  Run the fold simulation on the cluster, copy gdisplac6.xpost to sim_files/,")
    print("  then re-run this script.")
