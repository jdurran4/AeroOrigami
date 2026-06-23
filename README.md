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
    disp=[(N.near(0, 0, 47.5, tol=0.1), [1, 2, 3])],
    lmpc=[
        {"type": "min_z", "z_min": 46.8, "nodes": N.above(z=46.8)},  # disk floor
        {"type": "min_radius", "r_min": 0.1},                          # all membrane nodes
    ],
    cables=[{"blocks": ["Suspension_Lines"], "tol": 0.05}],
)

# Step 6: write all AERO-S files + cluster run scripts
sim = SimConfig(project_name="MyChute", sim_name="my_fold", end_time=1.0, shell_E=1e7)
write_aeros(surrogate, output_dir="sim_files/", config=config, sim=sim)
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

`write_aeros()` writes AERO-S include files and cluster run scripts into `output_dir/`:

| File | Contents | Written when |
|---|---|---|
| `ORIGAMI_MESH.include` | NODES, TOPOLOGY, ATTRIBUTES, spherical-joint MATERIAL | always |
| `ACTUATORS.include` | MATERIAL: one CONMAT RAMP per revolute joint | always |
| `EFRAMES.include` | EFRAMES: local axes per revolute joint | always |
| `MATERIAL.include` | Shell + cable spring material properties | `sim=` provided |
| `fold.fem` | Main AERO-S input file | `sim=` provided |
| `run.sh` | Shell script to compile `control.so` and launch AERO-S | `sim=` provided |
| `run.sbatch` | SLURM batch script | `sim=` provided |
| `postpro.sh` | Post-processing: generate `.top` + convert to Exodus | `sim=` provided |
| `clean.sh` | Wipe AERO-S output while keeping directory structure | `sim=` provided |
| `DISP.include` | Dirichlet BCs | `config` has `disp_bcs` |
| `LMPC.include` | Inequality constraints | `config` has `lmpc_rows` |
| `USDF.include` | User-defined force DOF list | `config` has `force_bcs` |
| `control.C` | C++ force driver (compile → `control.so`) | `config` has `force_bcs` |

`fold.fem` is ready to run immediately — `INCLUDE` lines for unused sections are
automatically omitted.

## Cluster workflow

After running the Python pipeline, sync the generated files to your cluster and submit:

```bash
# Edit HOST and REMOTE_BASE in sync_to_cluster.sh once, then:
./sync_to_cluster.sh examples/dgb_parachute dgb_v1
# → rsyncs sim_files/ to ind2:/home/.../aeroorigami/dgb_v1/

# On the cluster:
cd /home/tdurrant/parachute/aeroorigami/dgb_v1
sbatch run.sbatch          # run the fold simulation
sbatch postpro.sh          # after job finishes: generate .exo for ParaView
bash clean.sh              # wipe output to re-run clean
```

Update the path variables at the top of `run.sh` and `postpro.sh` to match your
cluster's AERO-S installation before the first submit.

## Project structure

```
AeroOrigami/
├── pyaeroori/              # The library
│   ├── __init__.py         # Public exports
│   ├── mesh.py             # Step 1: load original FEM mesh
│   ├── crease.py           # Step 2: load crease pattern CSVs
│   ├── remesh.py           # Step 3: surrogate mesh (Gmsh or crease-as-mesh)
│   ├── surrogate.py        # Step 4: node duplication, hinge joints, actuators
│   ├── physics.py          # Step 5: add_physics — BCs, LMPCs, cable springs
│   ├── writer.py           # Step 6: write_aeros — AERO-S files + cluster scripts
│   └── plot.py             # Visualization helpers
├── examples/
│   ├── dgb_parachute/      # DGB parachute — full pipeline with real mesh
│   │   ├── run.py
│   │   ├── dgb_mesh.fem
│   │   ├── dgb_disk_creases.csv
│   │   ├── dgb_band_creases.csv
│   │   ├── convert_alexandra_creases.py
│   │   └── sim_files/      # Generated by run.py (gitignored except .gitkeep subdirs)
│   │       ├── postpro/    # Post-processing output (.exo files)
│   │       ├── references/ # AERO-S restart files
│   │       └── results/    # AERO-S xpost output files
│   └── simple_chute/       # Flat rectangular membrane — minimal example
│       ├── run.py
│       ├── simple_chute_mesh.fem
│       ├── simple_chute_creases.csv
│       └── sim_files/
├── sync_to_cluster.sh      # rsync sim_files to a computing cluster
└── docs/
    ├── architecture.md     # Pipeline overview, module layout, data structures
    └── design_notes.md     # Rationale behind key technical decisions
```

## Documentation

- [Architecture](docs/architecture.md) — pipeline overview, module layout, data structures
- [Design notes](docs/design_notes.md) — rationale behind key technical decisions
