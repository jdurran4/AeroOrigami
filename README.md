# AeroOrigami

Python library for setting up origami-inspired parachute folding simulations in AERO-S.

Given a high-fidelity parachute FEM mesh and a crease pattern, AeroOrigami generates all
AERO-S input files needed to simulate the fold. After the simulation runs, it maps the
folded displacements back onto the original fine mesh for use in downstream FSI simulations.

## Dependencies

```bash
pip install numpy scipy matplotlib gmsh meshio
```

## Quickstart

```python
from pyaeroori import (
    load_mesh, load_creases, Region, remesh,
    build_surrogate, add_physics, write_aeros, SimConfig, N,
)

# Steps 1–2: load mesh and crease pattern
mesh    = load_mesh("mesh.fem")
creases = load_creases("disk_creases.csv", "band_creases.csv")

# Step 3: generate crease-aligned surrogate mesh
region  = Region(creases, name="chute", use_crease_mesh=True)
coarse  = remesh(mesh, region)

# Step 4: build surrogate (node duplication + driver joints)
surrogate = build_surrogate(coarse, creases, penalty_stiffness=8e9)

# Step 5: add boundary conditions, constraints, and cables
config = add_physics(
    surrogate,
    mesh=mesh,
    disp=[(N.near(0, 0, 47.5, tol=0.1), [1, 2, 3])],        # pin vent apex
    lmpc=[{"type": "min_z", "z_min": 40.0}],                 # floor constraint
    cables=[{"blocks": ["Suspension_Lines"], "tol": 0.05}],  # cable springs
)

# Step 6: write all AERO-S files
sim = SimConfig(project_name="MyChute", end_time=1.0, shell_E=1e7)
write_aeros(surrogate, output_dir="results/", config=config, sim=sim)
```

See `examples/dgb_parachute/run.py` for a complete annotated driver including
plot calls and both Path A (Gmsh remesh) and Path B (crease-as-mesh) options.

## Crease pattern CSV format

Each CSV file has one row per fold-line segment:

```
x1, y1, z1, x2, y2, z2, angle, type
```

| Column | Meaning |
|---|---|
| `x1,y1,z1` / `x2,y2,z2` | 3D endpoints in metres |
| `angle` | Target fold angle in radians — positive = mountain, negative = valley |
| `type` | `C` (crease fold) or `B` (boundary edge, no folding) |

Multiple CSV files can be passed to `load_creases()`; rows are merged into one `CreasePattern`.

## Output files

`write_aeros()` produces AERO-S include files in `output_dir/`:

| File | Contents | Written when |
|---|---|---|
| `ORIGAMI_MESH.include` | NODES, TOPOLOGY, ATTRIBUTES, spherical-joint MATERIAL | always |
| `ACTUATORS.include` | MATERIAL: one CONMAT RAMP per revolute joint | always |
| `EFRAMES.include` | EFRAMES: local axes per revolute joint | always |
| `MATERIAL.include` | Shell + cable spring material properties | `sim=` provided |
| `fold.fem` | Main AERO-S input file with all parameters | `sim=` provided |
| `DISP.include` | Dirichlet BCs | `config` has `disp_bcs` |
| `LMPC.include` | Inequality constraints | `config` has `lmpc_rows` |
| `USDF.include` | User-defined force DOF list | `config` has `force_bcs` |
| `control.C` | C++ force driver (compile → `control.so`) | `config` has `force_bcs` |

The `fold.fem` file is ready to run: `INCLUDE` lines for unused sections are
automatically omitted.

## Project structure

```
AeroOrigami/
├── pyaeroori/          # The library
│   ├── __init__.py     # Public exports
│   ├── mesh.py         # Step 1: load original FEM mesh
│   ├── crease.py       # Step 2: load crease pattern CSVs
│   ├── remesh.py       # Step 3: surrogate mesh (Gmsh Path A or crease-as-mesh Path B)
│   ├── surrogate.py    # Step 4: node duplication, hinge joints, actuators, EFRAMES
│   ├── physics.py      # Step 5: add_physics — BCs, LMPCs, cable springs
│   ├── writer.py       # Step 6: write_aeros — all AERO-S include files
│   ├── plot.py         # Visualization helpers
│   └── mapping.py      # Step 7: map fold displacements to fine mesh (TODO)
├── examples/
│   ├── dgb_parachute/  # DGB parachute — full pipeline with real mesh
│   │   ├── run.py
│   │   ├── dgb_mesh.fem
│   │   ├── dgb_disk_creases.csv
│   │   ├── dgb_band_creases.csv
│   │   ├── convert_alexandra_creases.py
│   │   └── results/    # Generated AERO-S files (from running run.py)
│   └── simple_chute/   # Flat rectangular membrane — minimal working example
│       ├── run.py
│       ├── simple_chute_mesh.fem
│       ├── simple_chute_creases.csv
│       └── results/
└── docs/
    ├── architecture.md  # Pipeline overview, module layout, data structures
    └── design_notes.md  # Rationale behind key technical decisions
```

## Documentation

- [Architecture](docs/architecture.md) — pipeline overview, module layout, data structures
- [Design notes](docs/design_notes.md) — rationale behind key technical decisions
