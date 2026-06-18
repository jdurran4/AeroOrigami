# Design Notes

Rationale behind key technical decisions. Update this when a decision changes.

---

## Crease pattern input format

**Decision:** One or more CSV files, one row per line segment, columns:
`x1, y1, z1, x2, y2, z2, angle, type`
where `type = C` (crease) or `B` (boundary), and `angle` sign encodes
mountain (positive) vs. valley (negative).

**Why:** The prototype used 6–8 separate CSV files per region (nodes, edges,
angles, edge loop, vent loop, etc.) because the schema grew organically over a
year of research. Consolidating to one row-per-segment file with a consistent
column schema reduces the reader to a single function and makes the required
data immediately obvious to a new user. `type` was kept (rather than inferring
boundary vs. crease from `angle = 0`) because a flat fold with `angle = 0` is
physically different from a boundary edge — using 0 as a sentinel would be
ambiguous. CSV was chosen over JSON/YAML/FOLD format because researchers
generating crease patterns in MATLAB or Python can export CSV trivially, and
the data is simple enough that a standard origami format adds no value.

**Angle convention:** Positive = mountain fold, negative = valley fold, in radians.

---

## Mesh format: no required block names

**Decision:** `load_mesh` only requires `NODES` and `TOPOLOGY` sections.
No specific block names or mesh structure is assumed. The origami surface
(membrane elements) is auto-detected from element connectivity: 3- and 4-node
elements are membranes; 2-node elements are cables. The user can override this
with explicit node IDs or an attribute ID filter.

**Why:** The prototype hard-coded DGB-specific block names (`Disk_Gores`,
`Band_Gores`, etc.) throughout the parsing and constraint code. This made it
impossible to use the tool with any other mesh without renaming blocks. The
generalization allows any AERO-S mesh to be used directly. The element-connectivity
heuristic for membrane vs. cable detection works for all standard AERO-S membrane
models and requires no user input for typical cases.

---

## RBF interpolation for displacement mapping (Step 7)

**Decision:** Multiquadric RBF with 100 nearest neighbors, smoothing=1e-7.

**Why:** The coarse origami surrogate and the fine FSI mesh have different node
locations. Linear interpolation (Delaunay-based) fails near mesh boundaries where
the convex hull of the coarse mesh doesn't fully contain fine mesh nodes — these
extrapolated points return NaN. RBF handles extrapolation gracefully without
special-casing boundary nodes. The multiquadric kernel with a small smoothing term
was found empirically to give smooth, accurate displacement fields for the DGB case.

**Limitation:** 100 neighbors is hard-coded as a default. For very coarse surrogate
meshes this may need to be reduced. Expose as a parameter in `map_displacements`.

---

## Cable reconstruction (Step 7)

**Decision:** Cable node positions are not interpolated via RBF. Instead, each cable
chain (vent lines, gap lines, suspension lines) is reconstructed by arc-length
parameterization between its two known end nodes.

**Why:** Cables are 1D structures. RBF interpolation of a thin 1D cable embedded
in a 3D displacement field tends to produce physically unrealistic lateral deflections.
Reconstructing along the straight line between the deformed end nodes (which ARE
known from the canopy/band interpolation) is physically correct for taut cables.

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

**Assumption:** The mesh must be bipartite across creases (i.e., no three panels
meet at a single crease edge). This holds for Miura-Ori patterns but should be
validated for other crease patterns.

---

## Joint types for hinges

**Decision:** Interior crease nodes get revolute joints (AERO-S element type 126).
Boundary crease nodes and crease junction nodes (where multiple creases meet) get
spherical joints (type 120).

**Why:** Revolute joints constrain rotation to a single axis (the crease tangent),
which is correct for interior fold lines. At boundaries and junctions the fold
direction is ambiguous or the mesh has less regularity, so the less-constrained
spherical joint avoids over-constraining the simulation.

---

## LMPC minimum-radius constraint

**Decision:** Append LMPC inequality constraints enforcing `r >= min_radius` for all
canopy nodes.

**Why:** Without this, the disk can fold past the vent centerline and nodes collapse
to zero radius, causing the nonlinear solver to diverge. The constraint is a soft
floor rather than a fixed boundary condition, so it only activates when a node would
otherwise cross the vent center.
