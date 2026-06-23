# AeroOrigami Architecture

## Overview

AeroOrigami is a Python library for setting up origami-inspired folding simulations
of deployable structures in AERO-S. It takes a high-fidelity mesh and a crease
pattern and produces all AERO-S input files needed to simulate the fold. After the
fold simulation runs, it maps the folded displacements back onto the original fine
mesh for downstream FSI simulations.

The library lives in the `pyaeroori` package. Users interact with it through a
short driver script (see `examples/dgb_parachute/run.py`).

---

## The 7-Step Pipeline

```
Original mesh (.fem) + crease CSVs
          │
          ▼
[1] load_mesh          → Mesh
[2] load_creases       → CreasePattern
[3] remesh             → CoarseMesh  (Gmsh Path A or crease-as-mesh Path B)
[4] build_surrogate    → Surrogate   (node duplication + driver joints + EFRAMES)
[5] add_physics        → ModelConfig (BCs, LMPC constraints, cable springs)
[6] write_aeros        → AERO-S include files + cluster scripts on disk
          │
          │   (user syncs sim_files/ to cluster, runs AERO-S)
          │
[7] map_displacements  → fine-mesh displacement field  (not yet implemented)
    write_idisp6       → IDISP6.include
```

---

## Module Layout

| Module | Step | Public API |
|---|---|---|
| `mesh.py` | 1 | `load_mesh(path) → Mesh` |
| `crease.py` | 2 | `load_creases(*paths) → CreasePattern` |
| `remesh.py` | 3 | `Region(creases, ...)`, `remesh(mesh, *regions) → CoarseMesh` |
| `surrogate.py` | 4 | `build_surrogate(coarse, creases, ...) → Surrogate` |
| `physics.py` | 5 | `add_physics(surrogate, ...) → ModelConfig`, `N` / `NodeQuery` |
| `writer.py` | 6 | `write_aeros(surrogate, output_dir, config, sim) → dict[str, Path]`, `SimConfig` |
| `plot.py` | — | `plot_mesh`, `plot_surrogate_axes`, `plot_physics`, `mesh_stats`, … |

---

## Key Data Structures

### `Mesh`
The original high-fidelity mesh. Read-only after Step 1.

```python
@dataclass
class Mesh:
    nodes:      dict[int, tuple[float, float, float]]  # nid → (x, y, z)
    elements:   dict[int, tuple[int, list[int]]]        # eid → (etype, [nids])
    attributes: dict[int, int]                          # eid → attr_id
    blocks:     dict[str, list[int]]                    # block name → [eids]

    # Derived properties (not stored fields):
    # membrane_elements, cable_elements, membrane_nodes, cable_nodes
```

AERO-S element types stored in `etype`: 2-node (cable/bar), 3-node (triangular
shell), 4-node (quad shell), etc. Block names come from `*  name: BlockName`
headers in the FEM file. 2-node elements are `cable_elements`; 3+ node elements
are `membrane_elements`.

### `CreasePattern`
Fold-line segments loaded from one or more CSVs.

```python
@dataclass
class CreasePattern:
    segments: list[CreaseSegment]  # all C-type rows
    boundary: list[BoundaryEdge]   # all B-type rows
```

Each `CreaseSegment` carries `(p1, p2, angle, fold_type)` where `fold_type` is
`"mountain"` (angle > 0) or `"valley"` (angle < 0).

### `CoarseMesh`
The crease-aligned remeshed surface, output of Step 3. Same `Mesh` dataclass,
with `panel_map` populated (eid → panel_id).

### `Surrogate`
The origami surrogate, output of Step 4. Contains the duplicated crease nodes
and all joint elements.

```python
@dataclass
class Surrogate:
    nodes:             dict[int, tuple[float, float, float]]
    elements:          dict[int, tuple[int, list[int]]]
    revolute_joints:   list[JointInfo]
    spherical_joints:  list[JointInfo]
    panel_map:         dict[int, int]
    penalty_stiffness: float

@dataclass
class JointInfo:
    eid:          int
    node_a:       int    # original crease node
    node_b:       int    # duplicated copy
    axis:         tuple  # unit vector along crease
    target_angle: float  # radians (revolute only)
    start_time:   float
    end_time:     float
```

### `ModelConfig`
Output of Step 5 (`add_physics`). Passed to `write_aeros`.

```python
@dataclass
class ModelConfig:
    disp_bcs:       list[tuple[int, list[int]]]            # (nid, [dofs])
    lmpc_rows:      list[LmpcRow]
    force_bcs:      list[tuple[int, float, float, float]]  # (nid, fx, fy, fz)
    cable_nodes:    dict[int, tuple[float, float, float]]  # new nodes for cables
    cable_elements: list[tuple[int, int, list[int]]]       # (eid, etype, [nids])
```

### `SimConfig`
Simulation parameters written into `fold.fem` and `MATERIAL.include`.

```python
@dataclass
class SimConfig:
    project_name:    str   = "AeroOrigami"
    sim_name:        str   = "origami_fold"
    time_step:       float = 5e-5
    end_time:        float = 1.0
    rho:             float = 0.7      # Newmark numerical dissipation
    a_damp:          float = 1e-7     # Rayleigh damping coefficient (RAYDAMP param 1)
    b_damp:          float = 2.0      # Rayleigh damping coefficient (RAYDAMP param 2)
    solver:          str   = "sparse"
    lmpc_penalty:    float = 1e8
    output_freq:     int   = 100
    restart_freq:    int   = 100
    shell_E:         float = 1e7
    shell_nu:        float = 0.4
    shell_rho:       float = 40000.0
    shell_t:         float = 1.0
    cable_stiffness: float = 10000.0
```

---

## AERO-S File Format Notes

AERO-S `.include` / `.fem` files use a section-based text format. Each section
ends with `*`. Named blocks (optional) use a `*  name: BlockName` header.

```
NODES
  <nid>  <x>  <y>  <z>
*
TOPOLOGY
  <eid>  <etype>  <nid1>  [<nid2> ...]
*
ATTRIBUTES
  <eid>  <attr_id>
*
MATERIAL
  <attr_id>  0  <E>  <nu>  <rho>  0  0  <t>  0  0  0  0  0  0  0
  <attr_id>  SPRINGMAT  <k>
  <attr_id>  CONMAT  penalty  <stiffness>  [RAMP <angle> 0.0 <t0> <t1>]
*
EFRAMES
  <eid>  <e1x> <e1y> <e1z>  <e2x> <e2y> <e2z>  <e3x> <e3y> <e3z>
*
LMPC
  <cid>  <rhs>  MODE 1
  <nid>  <dof>  <coeff>
*
DISP
  <nid>  <dof>  0.0
*
USDF
  <nid>  <dof>
```

### AERO-S element types used by AeroOrigami

| Type | Nodes | Description |
|---|---|---|
| 15 | 3 | Triangular AQR shell (6 DOF/node) |
| 1515 | 4 | Quadrilateral AQR shell (6 DOF/node) |
| 120 | 2 | Spherical joint (unconstrained rotation) |
| 126 | 2 | Revolute driver joint (CONMAT RAMP) |
| 203 | 2 | Tension-only spring (cable) |

### Attribute ID assignments (fixed in `writer.py`)

| ID | Used for |
|---|---|
| 1 | All shell elements |
| 2 | All spherical joints |
| 3, 4, 5, … | Revolute joints (one per joint, sequential) |
| 10 | Cable tension-only springs |

---

## Step 3: Two remesh paths

**Path B — crease-as-mesh** (`use_crease_mesh=True`, default for structured patterns):
Every crease-segment endpoint becomes a mesh node. The crease geometry drives the
mesh directly without Gmsh. No intermediate `.msh` file. Faster and guarantees full
crease coverage.

**Path A — Gmsh remesh** (default when `use_crease_mesh=False`):
Gmsh generates a new mesh at a target element size, with crease lines embedded as
geometric constraints. Produces more uniform element sizes. Requires Gmsh.
Output saved as `.msh` (gitignored).

Both paths produce a `Mesh` with `panel_map` populated.

---

## Step 5: NodeQuery API

`add_physics` uses a lazy `NodeQuery` (alias `N`) to select nodes without
requiring a pre-run to find IDs. Queries resolve at runtime and print matches:

```python
N.near(x, y, z, tol=0.05)              # sphere around a point
N.along_line(p1, p2, tol=0.05)         # within distance of a segment
N.above(z=46.8)                         # z >= threshold (x=, y= also supported)
N.all()                                 # every node
N.ids([1, 2, 3])                        # explicit IDs
```

All co-located copies of a selected node (created by node duplication in Step 4)
are automatically included in DISP BCs.

NodeQuery resolves against the combined surrogate + cable endpoint node set, so
queries like `N.near(...)` can target cable attachment points even though those
nodes are added during cable processing.

---

## Step 5: Processing order in `add_physics`

Cables are processed **before** DISP/LMPC/force BCs. This ensures that cable
endpoint nodes (added to `config.cable_nodes` during chain detection) are available
when NodeQuery resolves DISP and force BCs. All three BC types resolve against
`{**surrogate.nodes, **config.cable_nodes}`.

---

## Step 5: LMPC constraints

```python
lmpc=[
    # Apply min_z only to disk nodes (z >= 46.8), not the full canopy:
    {"type": "min_z",      "z_min": 46.8, "nodes": N.above(z=46.8)},
    # Apply min_radius to all membrane nodes (no "nodes" key = default):
    {"type": "min_radius", "r_min": 0.1},
]
```

The optional `"nodes"` key accepts any `NodeQuery` (or explicit ID list) to
restrict which nodes receive the constraint. Without it, the constraint applies
to all membrane nodes in the surrogate.

---

## Step 5: Cable detection

Cable elements from the original mesh are collapsed to single tension-only
springs (type 203) between chain endpoints. Three detection modes:

| Spec key | Source | Use case |
|---|---|---|
| `block` / `blocks` | Named block(s) in original mesh | DGB — avoids picking up embedded canopy beams |
| `all_bars` | All 2-node elements in mesh | Simple meshes without unwanted beams |
| `points` | Explicit (x,y,z) list | Manual override |

The chain-collapsing algorithm (`_build_cable_chains`) handles both linear chains
and star topologies (N arms meeting at a confluence node).

---

## Step 6: What `write_aeros` writes

Material sections across multiple INCLUDE files are merged by AERO-S at load time.
Attribute IDs are globally unique so there is no conflict.

| File | Material section contents |
|---|---|
| `ORIGAMI_MESH.include` | Attr 2: spherical joint CONMAT |
| `ACTUATORS.include` | Attrs 3+: revolute joint CONMAT RAMP (one per joint) |
| `MATERIAL.include` | Attr 1: shell material, Attr 10000: cable SPRINGMAT |

The `fold.fem` main input file is written with conditional INCLUDE lines:
`LMPC.include`, `USDF.include`, `LOAD ./control.so`, and `DISP.include` are
omitted when the corresponding sections are empty.

When `sim=` is provided, four cluster scripts are also written into `output_dir/`:

| Script | Purpose |
|---|---|
| `run.sh` | Compile `control.so` (if `control.C` present) and launch AERO-S |
| `run.sbatch` | SLURM wrapper for `run.sh`; job name set to `sim.sim_name` |
| `postpro.sh` | Generate `.top` topology file and convert to Exodus for ParaView |
| `clean.sh` | Wipe xpost/restart/postpro output while keeping directory structure |

Path variables (`AEROS`, `AEROSDIR`, etc.) at the top of `run.sh` and `postpro.sh`
must be updated to match the target cluster before submitting.

---

## sim_files directory layout

Each example writes output to `examples/<example>/sim_files/`. AERO-S expects
three subdirectories to exist before it runs:

```
sim_files/
├── fold.fem, *.include, run.sh, run.sbatch, postpro.sh, clean.sh
├── postpro/      ← xp2exo Exodus output (tracked via .gitkeep)
├── references/   ← AERO-S restart files  (tracked via .gitkeep)
└── results/      ← AERO-S xpost output   (tracked via .gitkeep)
```

`run.sh` creates these subdirectories with `mkdir -p` before launching AERO-S,
so they will exist even if `.gitkeep` files were not synced.
