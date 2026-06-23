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
| 6 | `write_aeros` | `writer.py` | AERO-S files on disk |
| 7 | `map_displacements` / `write_idisp6` | `mapping.py` | **Not yet implemented** |

All public exports are in `pyaeroori/__init__.py`.

## Key files in `pyaeroori/`

- **`mesh.py`**: Parses AERO-S `.fem`/`.include` files. Sections: `NODES`, `TOPOLOGY`,
  `ATTRIBUTES`. Block names from `*  name: BlockName` headers go into `Mesh.blocks`.
  2-node elements are stored separately in `Mesh.cable_elements`.

- **`remesh.py`**: Two paths controlled per `Region`:
  - **Path B** (`use_crease_mesh=True`): crease endpoints become mesh nodes — no Gmsh,
    guaranteed crease coverage, used by default.
  - **Path A** (`use_crease_mesh=False`): Gmsh generates mesh at target `mesh_size`,
    saves `.msh` (gitignored). Better for curved / unstructured surfaces.

- **`surrogate.py`**: BFS panel detection, 2-coloring, node duplication. Revolute
  joints (type 126) on interior crease nodes; spherical joints (type 120) on boundary
  and junction nodes.

- **`physics.py`**: `add_physics(surrogate, mesh, disp, lmpc, forces, cables)`.
  NodeQuery (alias `N`) resolves at runtime and prints matched nodes. Cable chains
  detected via `_build_cable_chains` — collapses each connected chain of 2-node
  elements to a single type-203 tension-only spring between endpoints.

- **`writer.py`**: `write_aeros(surrogate, output_dir, config, sim, beta_factor)`.
  Always writes `ORIGAMI_MESH.include`, `ACTUATORS.include`, `EFRAMES.include`.
  When `config` provided: adds `DISP.include`, `LMPC.include`, and if `force_bcs`
  are present, `USDF.include` + `control.C`. When `sim=SimConfig(...)` provided:
  adds `MATERIAL.include` and `fold.fem` (main AERO-S input file, conditional
  INCLUDE lines omitted when unused).

- **`plot.py`**: Visualization helpers — `plot_mesh`, `plot_creases`,
  `plot_surrogate_axes`, `plot_physics`, `mesh_stats`, `crease_stats`,
  `check_crease_coverage`, `check_mesh_crease_resolution`.

## AERO-S element types and attribute IDs

Element types in TOPOLOGY: 15 (tri shell), 1515 (quad shell), 120 (spherical joint),
126 (revolute driver), 203 (tension-only spring).

Attribute IDs are fixed constants in `writer.py`:
- 1 → shell material (in `MATERIAL.include`)
- 2 → spherical joint CONMAT (in `ORIGAMI_MESH.include`)
- 3+ → revolute joint CONMAT RAMP, one per joint (in `ACTUATORS.include`)
- 10 → cable SPRINGMAT (in `MATERIAL.include`)

Multiple MATERIAL sections in different INCLUDE files are merged by AERO-S at load
time — IDs must be globally unique, which they are.

## Key design decisions (short form)

- **CONMAT RAMP, not USDF, for fold actuation**: USDF forces were the prototype
  approach; CONMAT RAMP (intrinsic AERO-S actuators) is cleaner. USDF is now for
  optional explicit helper forces only.
- **USDF not FORCE for dynamic forces**: The fold simulation uses DYNAMICS / Newmark.
  Static `FORCE` is ignored in DYNAMICS runs. `config.force_bcs` → `USDF.include` +
  `control.C` (compile with `g++ -shared -fPIC control.C -o control.so`).
- **Cable chain collapse**: Each connected chain of bars → single type-203 spring
  between endpoints. Avoids over-constraining the fold.
- **Co-located node pinning**: When a DISP BC targets a crease node, all co-located
  duplicates (same rounded coords) are also pinned automatically.
- See `docs/design_notes.md` for fuller rationale.

## Step 7 (displacement mapping) — not yet implemented

`pyaeroori/mapping.py` does not exist yet. It should implement:
1. `map_displacements(mesh, fold_mesh, disp_file, rbf_neighbors=100) → dict`
   - Read 6-DOF xpost from AERO-S fold simulation
   - RBF (multiquadric) from coarse surrogate to fine mesh canopy nodes
   - Arc-length reconstruction for cable nodes
2. `write_idisp6(displacements, output_path)` → `IDISP6.include`

Reference implementation in `Origami_Parachute/origami/helper.py`:
`map_to_fine_meshRBF` and related functions. The new implementation should use
the same RBF approach but through `pyaeroori`'s cleaner data structures.

## Hard-coded assumptions

- Shell elements in the surrogate are always attribute 1; cables always attribute 10.
  These are fixed in `writer.py` constants `_SHELL_ATTR` and `_CABLE_ATTR`.
- `beta_factor = 0.1` default: revolute joint beta stiffness = penalty × 0.1.
- AERO-S section terminator is `*`. Block headers are `*  name: BlockName`.
- Crease CSV columns: `x1, y1, z1, x2, y2, z2, angle, type` — any other column
  order will silently produce wrong results.
