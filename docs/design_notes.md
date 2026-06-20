# Design Notes

Rationale behind key technical decisions. Update this when a decision changes.

---

## Crease pattern input format

**Decision:** One or more CSV files, one row per line segment, columns:
`x1, y1, z1, x2, y2, z2, angle, type`
where `type = C` (crease) or `B` (boundary), and `angle` sign encodes
mountain (positive) vs. valley (negative).

**Why:** The prototype used 6–8 separate CSV files per region (nodes, edges,
angles, edge loop, vent loop, etc.) because the schema grew organically. Consolidating
to one row-per-segment file with a consistent column schema reduces the reader to a
single function and makes the required data immediately obvious. `type` is kept (rather
than inferring boundary vs. crease from `angle = 0`) because a flat fold with
`angle = 0` is physically different from a boundary edge — using 0 as a sentinel
would be ambiguous. CSV was chosen over JSON/YAML/FOLD because researchers generating
crease patterns in MATLAB or Python can export CSV trivially.

---

## Mesh format: no required block names

**Decision:** `load_mesh` only requires `NODES` and `TOPOLOGY` sections.
No specific block names or mesh structure is assumed. Block names are parsed when
present and stored in `Mesh.blocks` for optional use by `add_physics` cable detection.

**Why:** The prototype hard-coded DGB-specific block names (`Disk_Gores`,
`Band_Gores`, etc.) throughout the code. This made the tool unusable with any other
mesh without renaming blocks. The new design auto-detects membrane elements (3- or
4-node) and cable elements (2-node) from connectivity. Named blocks are opt-in.

---

## Two remesh paths (Path A and Path B)

**Decision:** Support both Gmsh remeshing (Path A) and crease-as-mesh (Path B) in the
same `remesh()` function. Controlled per-region via `Region(use_crease_mesh=True/False)`.

**Why:** Path B (crease-as-mesh) guarantees full crease coverage because every
crease-segment endpoint is a mesh node. For structured Miura-Ori patterns this
produces a high-quality mesh without Gmsh. Path A is better for curved surfaces,
unstructured patterns, or when uniform element size is needed. Both produce the same
`CoarseMesh` interface so all downstream steps are identical.

---

## Panel 2-coloring

**Decision:** Detect panels via BFS on non-crease edges, then 2-color the panel
adjacency graph. Panels of color 0 retain original crease nodes; panels of color 1
get duplicate nodes.

**Why:** Each crease fold is a hinge between two panels. To allow relative rotation,
the shared crease nodes must be split — one copy per panel side. BFS on the mesh
with crease edges treated as barriers naturally groups elements into flat panels.
2-coloring ensures a consistent assignment of which side gets the original node and
which gets the duplicate.

**Assumption:** The mesh must be bipartite across creases (no three panels meeting at
a single crease edge). This holds for Miura-Ori patterns but should be validated for
other crease patterns.

---

## Joint types for hinges

**Decision:** Interior crease nodes get revolute joints (AERO-S type 126). Boundary
crease nodes and junction nodes (where multiple creases meet) get spherical joints
(type 120).

**Why:** Revolute joints constrain rotation to a single axis (the crease tangent),
which is correct for interior fold lines. At boundaries and junctions the fold
direction is ambiguous or the mesh has less regularity, so the less-constrained
spherical joint avoids over-constraining the simulation.

---

## CONMAT RAMP actuators vs. USDF forces

**Decision:** Fold actuation is driven by AERO-S `CONMAT RAMP` revolute joints
(ACTUATORS.include). Optional helper forces are applied via `USDF` + `control.C`.

**Why:** The prototype used USDF forces as the primary fold driver. This required
knowing the force direction and magnitude for each fold node up front. CONMAT RAMP
actuators are self-contained — they drive toward a target angle at a specified ramp
rate, which is physically cleaner and requires only the target angle (already in
the crease CSV). USDF forces are now a secondary mechanism for users who need to
add explicit nodal loads to assist convergence or simulate inflation.

**USDF vs. static FORCE:** The fold simulation uses the AERO-S DYNAMICS solver
(Newmark integration). The static `FORCE` section is only parsed under STATICS.
For any dynamic load, USDF + `control.so` is the correct mechanism. `write_aeros`
therefore writes `USDF.include` + `control.C` (not `FORCE.include`) when
`force_bcs` are present.

---

## Cable representation as tension-only springs (type 203)

**Decision:** Each cable chain from the original mesh is collapsed to a single
AERO-S type-203 tension-only spring between the chain's two endpoint nodes.

**Why:** During the fold simulation, cables need to:
1. prevent the panels from spreading beyond the cable length, and
2. not resist compression (cables don't push).

Type 203 achieves both with a single spring stiffness parameter (`cable_stiffness`
in `SimConfig`). Keeping the full chain of bar elements would over-constrain the
fold by enforcing rigid intermediate node positions. The chain-collapse approach
(`_build_cable_chains` in `physics.py`) handles both linear chains and star
topologies (N suspension lines meeting at a confluence node).

---

## LMPC minimum-radius constraint

**Decision:** `add_physics(lmpc=[{"type": "min_radius", "r_min": ...}])` appends
LMPC inequality constraints enforcing `r >= r_min` for all canopy nodes.

**Why:** Without this, the disk can fold past the vent centerline and nodes collapse
to zero radius, causing the nonlinear solver to diverge. The constraint is a soft
floor rather than a fixed BC, so it only activates when a node would otherwise cross
the vent center.

---

## NodeQuery lazy resolution

**Decision:** Nodes for BCs are selected via `NodeQuery` (alias `N`) which resolves
at `add_physics` runtime and prints matched nodes with coordinates.

**Why:** The alternative — requiring users to find node IDs beforehand — needs either
a separate plotting run or manual inspection of the mesh file. The lazy resolver
eliminates that friction: users specify geometry (a point, a line segment) and the
query reports exactly which nodes matched, so they can verify immediately without a
pre-run.

---

## RBF interpolation for displacement mapping (Step 7 — not yet implemented)

**Decision (planned):** Multiquadric RBF with 100 nearest neighbors, smoothing=1e-7.

**Why:** The coarse origami surrogate and the fine FSI mesh have different node
locations. Linear interpolation (Delaunay-based) fails near mesh boundaries where
the convex hull of the coarse mesh doesn't fully contain fine mesh nodes — these
extrapolated points return NaN. RBF handles extrapolation gracefully without
special-casing boundary nodes. The multiquadric kernel with a small smoothing term
gave smooth, accurate displacement fields in the prototype.

---

## Cable path reconstruction (Step 7 — not yet implemented)

**Decision (planned):** Cable node positions are NOT interpolated via RBF. Instead,
each cable chain is reconstructed by arc-length parameterization between its two
deformed endpoint nodes.

**Why:** Cables are 1D structures. RBF interpolation of a thin 1D cable embedded
in a 3D displacement field produces physically unrealistic lateral deflections.
Reconstructing along the line between deformed endpoints (which ARE known from the
canopy interpolation) is correct for taut cables under tension.
