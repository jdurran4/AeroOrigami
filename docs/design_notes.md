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

## Cable representation as axial springs (type 200)

**Decision:** Each cable chain from the original mesh is collapsed to a single
AERO-S type-200 axial spring between the chain's two endpoint nodes.

**Why:** Keeping the full chain of bar elements over-constrains the fold by
enforcing rigid intermediate node positions. The chain-collapse approach
(`_build_cable_chains` in `physics.py`) handles both linear chains and star
topologies (N suspension lines meeting at a confluence node) and reduces
hundreds of elements per cable to a single spring with one stiffness parameter
(`cable_stiffness` in `SimConfig`).

Type 200 (axial spring) replaced the earlier type-203 (tension-only spring)
because the fold geometry can transiently put cables in compression during
dynamic relaxation, and type-203 dropping to zero stiffness in those moments
caused instability.

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

## Membrane interpolation for displacement mapping (Step 7)

**Decision:** Two methods, selectable via `method=` in `map_displacements`:

- `"rbf"` (default) — multiquadric `RBFInterpolator`, `degree=0`, tunable
  `rbf_neighbors` and `rbf_smoothing`.
- `"panel_rigid"` — per-panel Procrustes/Kabsch SVD rigid-body transform with
  nearest-centroid assignment for fine nodes.

**Why `"rbf"`:** The coarse surrogate and fine FSI mesh have different node
locations. Delaunay-based linear interpolation fails near mesh boundaries (fine
nodes outside the coarse convex hull return NaN). RBF handles extrapolation
gracefully. Multiquadric with `degree=0` always produces a full-rank system
because the only polynomial augmentation is a single constant term.

**Why `"panel_rigid"`:** Near the vent, many panels converge at large fold angles
with relatively few coarse nodes. RBF treats the field as globally smooth and
blurs displacement across crease lines. A per-panel rigid-body transform is
physically exact for a panel that has undergone a rigid fold: each fine node is
assigned to its nearest surrogate panel and displaced by that panel's Procrustes
R and t. There is no cross-panel averaging.

**Why not thin-plate-spline (TPS):** TPS requires polynomial augmentation to be
full-rank. In 3D with points lying on a 2D surface (the parachute canopy), degree-3
TPS has 20 polynomial basis functions but only 10 are linearly independent on a 2D
manifold — the system is singular (observed error: "rank 10/20"). Use multiquadric
(`degree=0`) or `panel_rigid` instead.

---

## Self-contact surface (ContactConfig)

**Decision:** `write_aeros(..., contact=ContactConfig(...))` optionally emits
`SURFACETOPO.include` + `CONTACTSURFACES.include`. The whole shell surrogate is
one surface paired with itself. Off by default; omitting `contact` reproduces
prior output byte-for-byte.

**One-sided.** A 2-sided ("shell") contact surface requires explicit dynamics
with `flagTDENFORCE On` plus a `SURFACETOPO` thickness attribute (AERO-S manual
Note 5). The fold runs implicit dynamic (Newmark), so the surface is
unavoidably 1-sided: contact is only detected between two facets whose normals
point in *opposite* directions. Consequences — (a) facet winding must be
consistent, see below; (b) a fold that closes two panels along a *mountain*
crease brings their inner faces together with normals diverging, which 1-sided
detection misses. Accepted for now; revisit if mountain-fold self-intersection
becomes a real problem (options: reversed-winding duplicate facets, or moving
the fold to explicit dynamics).

**Facets are remapped to original (pre-duplication) nodes** via
`Surrogate.node_origin`. Panels meeting at a crease then reference the same node
chain along that crease, so the surface is watertight and ACME automatically
excludes the hinge-adjacent facet slivers (they share an edge) from interaction
testing — that sliver is exactly the spurious "always in contact" pair we want
gone. The duplicated crease nodes are held coincident by their joints
(`penalty_stiffness` 8e9), so building the contact surface on the original node
tracks the true deformed structure to within the joint compliance. Cost: a
hinge cannot self-detect being folded past flat, but the driver controls that
angle anyway, and any genuinely flat fold still has plenty of non-edge-adjacent
facet pairs across the two panels that do get caught.

**Facets are wound outward** using `Surrogate.panel_normals` (the area-weighted
outward normals already computed in `build_surrogate`): a facet whose winding
normal opposes its panel normal is reversed before writing. Required because
1-sided detection reads the facet normal from node order. Caveat: for a
perfectly flat starting sheet the outward orientation from
`_compute_panel_normals_centroids` is degenerate (dot of normal with an
in-plane vector), so consistency there depends on `panel_outward_hints`; curved
canopies are well-defined.

**CONTACTSURFACES row** uses the static / implicit-dynamic form
`SURF_PAIR_ID# MASTER SLAVE MORTAR_TYPE NORMAL_TOL TANGENTIAL_TOL`.
`CONSTRAINT_METHOD` is omitted so it inherits from the `CONSTRAINTS` command in
`fold.fem` (penalty, `SimConfig.lmpc_penalty`) — emitting `penalty` explicitly
would force a `beta` value that has no default. `MASTER == SLAVE == surf_id`
gives self-contact. Manual defaults are used for the tolerances
(`NORMAL_TOL 0.1`, `TANGENTIAL_TOL 0.001`); `NORMAL_TOL` must exceed the
distance a surface point moves in one time step, so raise it (or cut the time
step) if AERO-S reports penetration.

---

## Cable path reconstruction (Step 7)

**Decision:** Cable intermediate nodes are NOT interpolated via RBF or
`panel_rigid`. Instead:
1. Cable chain endpoint and junction nodes are seeded via nearest-coarse-node
   KD-tree lookup (handles suspension line top/bottom nodes not shared with the
   canopy shell mesh).
2. Interior nodes are filled by arc-length parameterized linear interpolation
   between the two resolved chain endpoints.
3. Junction nodes (degree ≥ 3) are resolved as the mean of their resolved cable
   neighbours.

**Why separate treatment:** Cables are 1D structures. RBF interpolation of a thin
cable embedded in a 3D displacement field produces physically unrealistic lateral
deflections. Arc-length reconstruction along the deformed cable axis is correct for
taut cables. The KD-tree anchor seeding step was added after discovering that
suspension lines whose attachment nodes are not shared with the canopy shell mesh
(i.e., not membrane nodes) would be left at zero without it — BFS needs both
endpoints resolved before it can fill a chain.
