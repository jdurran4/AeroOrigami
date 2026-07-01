# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

AeroOrigami automates the fold-simulation setup for origami-inspired (Miura-Ori) crease
patterns on parachute FEM meshes, producing all AERO-S input files needed to run the fold.
After the AERO-S simulation completes, it maps the folded displacements back onto the
original fine mesh for downstream FSI simulations.

## Running the examples

```bash
# Simple rectangular chute (minimal example, no external data needed)
python examples/simple_chute/run.py

# DGB parachute (requires dgb_mesh.fem and crease CSVs in examples/dgb_parachute/)
# Convert Alexandra's raw CSVs first:
python examples/dgb_parachute/convert_alexandra_creases.py
python examples/dgb_parachute/run.py

# After running the fold simulation on the cluster, re-run run.py to execute Step 7,
# or iterate on mapping settings alone without re-running Steps 1-6:
python examples/dgb_parachute/step7.py
```

Output lands in `examples/<example>/sim_files/`. When `SimConfig` is passed to
`write_aeros`, cluster scripts (`run.sh`, `run.sbatch`, `postpro.sh`, `clean.sh`)
are written there automatically alongside the AERO-S include files.

To sync to a cluster:
```bash
./sync_to_cluster.sh examples/dgb_parachute dgb_v1
# Then on the cluster: cd /scratch/.../dgb_v1 && sbatch run.sbatch
```

No test suite. No build step. `requirements.txt` is a full conda environment dump.
Core dependencies: `numpy scipy matplotlib gmsh meshio`.

## Architecture: 7-step pipeline

| Step | Function | Module | Output |
|---|---|---|---|
| 1 | `load_mesh` | `mesh.py` | `Mesh` |
| 2 | `load_creases` | `crease.py` | `CreasePattern` |
| 3 | `remesh` | `remesh.py` | `CoarseMesh` |
| 4 | `build_surrogate` | `surrogate.py` | `Surrogate` |
| 5 | `add_physics` | `physics.py` | `ModelConfig` |
| 6 | `write_aeros` | `writer.py` | AERO-S files + cluster scripts on disk |
| 7 | `map_displacements` / `write_idisp6` | `mapping.py` | `IDISP6.include` + optional VTK |

All public exports are in `pyaeroori/__init__.py`.

## Key files in `pyaeroori/`

- **`mesh.py`**: Parses AERO-S `.fem`/`.include` files. Sections: `NODES`, `TOPOLOGY`,
  `ATTRIBUTES`. Block names from `*  name: BlockName` headers go into `Mesh.blocks`.
  2-node elements are stored separately in `Mesh.cable_elements`.

- **`remesh.py`**: Two paths controlled per `Region`:
  - **Path B** (`use_crease_mesh=True`): crease endpoints become mesh nodes — no Gmsh,
    guaranteed crease coverage, used by default.
  - **Path A** (`use_crease_mesh=False`): Gmsh generates mesh at target `mesh_size`,
    saves `.msh` (gitignored). Better for curved / unstructured surfaces. Key Gmsh
    settings: per-vertex `lc` passed to `addPoint`, `CharacteristicLengthMin/Max`,
    `CharacteristicLengthExtendFromBoundary=1` (propagates lc into panel interiors),
    `Algorithm=8` (Delaunay). These produce ~6 elements per panel, enabling revolute
    drivers on interior crease edges.

- **`surrogate.py`**: BFS panel detection, 2-coloring, node duplication. Revolute
  joints (type 126) on interior crease nodes; spherical joints (type 120) on boundary
  and junction nodes. `split_quads=True` (default): any 4-node quad panels are
  fan-split into two triangles (along the 0–2 diagonal) after node duplication; both
  triangles share the same `panel_id` so no joint is placed between them. Set
  `split_quads=False` to emit type-1515 quad shell elements instead.

- **`physics.py`**: `add_physics(surrogate, mesh, disp, lmpc, forces, cables)`.
  Processing order: cables first, then DISP/LMPC/forces — so cable endpoint nodes
  exist when NodeQuery resolves BCs. NodeQuery (alias `N`) resolves against
  `surrogate.nodes + config.cable_nodes` and prints matched nodes at runtime.
  Selectors: `N.near`, `N.along_line`, `N.above(z=...)`, `N.all`, `N.ids`.
  LMPC specs accept an optional `"nodes"` key (NodeQuery) to restrict which nodes
  receive the constraint; omitting it applies to all membrane nodes. Cable chains
  detected via `_build_cable_chains` — collapses each connected chain of 2-node
  elements to a single type-200 axial spring between endpoints.

- **`writer.py`**: `write_aeros(surrogate, output_dir, config, sim, beta_factor)`.
  Always writes `ORIGAMI_MESH.include`, `ACTUATORS.include`, `EFRAMES.include`.
  When `config` provided: adds `DISP.include`, `LMPC.include`, and if `force_bcs`
  are present, `USDF.include` + `control.C`. When `sim=SimConfig(...)` provided:
  adds `MATERIAL.include`, `fold.fem` (main AERO-S input file, conditional
  INCLUDE lines omitted when unused), and cluster scripts `run.sh`, `run.sbatch`,
  `postpro.sh`, `clean.sh`. Update path variables in `run.sh`/`postpro.sh` once
  for your cluster.

- **`plot.py`**: Visualization helpers — `plot_mesh`, `plot_creases`,
  `plot_surrogate_axes`, `plot_physics`, `mesh_stats`, `crease_stats`,
  `check_crease_coverage`, `check_mesh_crease_resolution`.

- **`mapping.py`**: Step 7 implementation — see section below.

## AERO-S element types and attribute IDs

Element types in TOPOLOGY: 15 (tri shell), 1515 (quad shell), 120 (spherical joint),
126 (revolute driver), 200 (axial spring / cable).

Attribute IDs are fixed constants in `writer.py`:
- 1 → shell material (in `MATERIAL.include`)
- 2 → spherical joint CONMAT (in `ORIGAMI_MESH.include`)
- 3+ → revolute joint CONMAT RAMP, one per joint (in `ACTUATORS.include`)
- 10000 → cable SPRINGMAT (in `MATERIAL.include`) — large fixed value to avoid colliding with revolute joint attrs, which can number in the thousands

Multiple MATERIAL sections in different INCLUDE files are merged by AERO-S at load
time — IDs must be globally unique, which they are.

## Key design decisions (short form)

- **CONMAT RAMP, not USDF, for fold actuation**: USDF forces were the prototype
  approach; CONMAT RAMP (intrinsic AERO-S actuators) is cleaner. USDF is now for
  optional explicit helper forces only.
- **USDF not FORCE for dynamic forces**: The fold simulation uses DYNAMICS / Newmark.
  Static `FORCE` is ignored in DYNAMICS runs. `config.force_bcs` → `USDF.include` +
  `control.C` (compile with `g++ -shared -fPIC control.C -o control.so`).
- **Cable chain collapse**: Each connected chain of bars → single type-200 axial spring
  between endpoints. Avoids over-constraining the fold. (Previously type-203
  tension-only; changed to type-200 to allow compression as well.)
- **Co-located node pinning**: When a DISP BC targets a crease node, all co-located
  duplicates (same rounded coords) are also pinned automatically.
- **TPS kernel singular matrix**: `RBFInterpolator` with `thin_plate_spline` fails on
  surface-embedded point clouds because the polynomial augmentation matrix is rank-
  deficient (coarse nodes lie on a 2D manifold in 3D space; degree-3 → rank 10/20).
  Use `multiquadric` with `degree=0` (constant term, always full rank) or
  `method="panel_rigid"` instead.
- See `docs/design_notes.md` for fuller rationale.

## Step 7 (displacement mapping)

`pyaeroori/mapping.py` implements:

1. `read_xpost(disp_file, step=-1, node_ids=None)`
   - Parses `gdisplac6.xpost` into `{nid: (dx,dy,dz,rx,ry,rz)}`
   - Handles both AERO-S output formats automatically:
     - 7-column rows: `node_id dx dy dz rx ry rz` (node ID explicit)
     - 6-column rows: `dx dy dz rx ry rz` (positional; pass `node_ids=` for correct mapping)
   - `map_displacements` passes `coarse_ids` so the 6-column format is always handled correctly

2. `map_displacements(surrogate, fine_mesh, disp_file, config, step, method, ...)`
   - **Step 3** — membrane interpolation (one of two methods):
     - `method="rbf"` (default): multiquadric RBF from all surrogate+cable-endpoint nodes
       → fine membrane nodes. Smooth but can blur across fold lines. Increasing
       `rbf_neighbors` (e.g. 200–500) improves smoothness; fewer neighbors → jagged.
     - `method="panel_rigid"`: per-panel Procrustes/Kabsch SVD rigid-body transform.
       Best for large-angle folds near vent/high-curvature regions — no cross-panel
       averaging. Each fine node is assigned to its nearest panel centroid (KD-tree).
       Rotation DOFs use `scipy.spatial.transform.Rotation.from_matrix(R).as_rotvec()`.
   - **Step 4.5** — cable anchor seeding: degree-1 (chain endpoints) and degree-≥3
     (junction) cable-only nodes not touched by membrane interpolation are seeded via
     nearest-coarse-node KD-tree lookup. This ensures BFS has both ends of every cable
     chain resolved before filling interiors. Without this step, suspension lines whose
     top/bottom nodes are not shared with the canopy shell mesh are left at zero.
   - **Step 5** — BFS arc-length reconstruction for all interior cable nodes. Confluence
     nodes (degree ≥ 3) resolved as mean of their resolved cable neighbours.
   - No block-name hardcoding — works for any cable topology.

3. `write_idisp6(displacements, fine_mesh, output_path, amp=1.0)` → `IDISP6.include`

4. `write_folded_vtk(fine_mesh, displacements, output_path)` → VTK for ParaView

### Fast iteration on Step 7 (without re-running Steps 1–6)

`run.py` pickles `{surrogate, mesh, config}` to `sim_files/pipeline_state.pkl` after
Step 6. Use `examples/dgb_parachute/step7.py` to reload the state and re-run only the
mapping with different settings (method, rbf_neighbors, etc.). This takes ~1 s vs ~30 s
for the full pipeline.

## Hard-coded assumptions

- Shell elements in the surrogate are always attribute 1; cables always attribute 10.
  These are fixed in `writer.py` constants `_SHELL_ATTR` and `_CABLE_ATTR`.
- `beta_factor = 0.1` default: revolute joint beta stiffness = penalty × 0.1.
- AERO-S section terminator is `*`. Block headers are `*  name: BlockName`.
- Crease CSV columns: `x1, y1, z1, x2, y2, z2, angle, type` — any other column
  order will silently produce wrong results.
