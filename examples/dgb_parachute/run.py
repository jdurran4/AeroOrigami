"""
DGB Parachute Origami Fold Setup
=================================
Driver script for generating AEROS input files for an origami-folded
DGB parachute. Edit the configuration block below, then run:

    python run.py

Outputs are written to the directory specified by `output_dir`.
"""

import pyaeroori

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Input files ---
mesh_file       = "data/mesh_ORTHO_mem.fem"   # High-fidelity AEROS mesh
parachute_file  = "data/parachute.include"    # AEROS parachute settings include
control_template = "data/control_template.C"  # AEROS control file template

disk_crease_file = "data/disk_creases.csv"    # Mountain/valley folds on the disk
band_crease_file = "data/band_creases.csv"    # Mountain/valley folds on the band

# --- Remeshing ---
mesh_size = 0.22          # Target element size for Gmsh remesh (meters)

# --- Surrogate build ---
penalty_stiffness = 8e9   # Stiffness for spherical joint constraints (Pa)
target_angle      = 3.12  # Default fold target angle (radians), used if not
                           # specified per-crease in the CSV
actuator_ramp_time = 3.0  # Time over which joints ramp to target angle (seconds)

# --- Constraints ---
min_radius     = 0.05     # LMPC minimum radial constraint (meters)
include_cables = True     # Add cable elements as tension-only springs

# --- Output ---
output_dir = "results/"   # All AEROS include files written here

# =============================================================================
# PIPELINE
# =============================================================================

# Step 1 — Load and process the original high-fidelity mesh
mesh = pyaeroori.load_mesh(mesh_file)

# Step 2 — Load the crease pattern (mountain/valley fold line segments + angles)
creases = pyaeroori.load_creases(disk_crease_file, band_crease_file)

# Step 3 — Remesh using Gmsh, aligned to the crease pattern
surrogate = pyaeroori.remesh(mesh, creases, mesh_size=mesh_size)

# Step 4 — Build the origami surrogate: detect panels, duplicate crease nodes,
#           embed driver hinge joints and actuator materials
surrogate = pyaeroori.build_surrogate(
    surrogate,
    creases,
    penalty_stiffness=penalty_stiffness,
    target_angle=target_angle,
    actuator_ramp_time=actuator_ramp_time,
)

# Step 5 — Add supplementary constraints and elements
surrogate = pyaeroori.add_constraints(
    surrogate,
    mesh,
    min_radius=min_radius,
    include_cables=include_cables,
)

# Step 6 — Write all AEROS input files to output_dir
pyaeroori.write_aeros(
    surrogate,
    mesh,
    parachute_file=parachute_file,
    control_template=control_template,
    output_dir=output_dir,
)

print(f"Done. AEROS input files written to {output_dir}")

# =============================================================================
# STEP 7 — POST-FOLD DISPLACEMENT MAPPING
# Run this block after completing the AEROS folding simulation.
# Supply the displacement output file from AEROS and re-run the script,
# or run this section separately.
# =============================================================================

# disp_file  = "foldfiles/gdisplac6.xpost.5"   # AEROS displacement output
# fold_mesh  = f"{output_dir}/mesh_modified.include"  # Surrogate mesh written in Step 6
# idisp_file = "foldfiles/IDISP6.include"       # Output: initial displacements for FSI run

# mapped = pyaeroori.map_displacements(
#     mesh,
#     fold_mesh,
#     disp_file,
#     rbf_neighbors=100,
# )
# pyaeroori.write_idisp6(mapped, idisp_file)
# print(f"Displacement mapping done. Wrote {idisp_file}")
