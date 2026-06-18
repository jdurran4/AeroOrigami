# AeroOrigami

Python library for setting up origami-inspired parachute (or other deployable structure) folding simulations in AERO-S.

Given a high-fidelity parachute mesh and a crease pattern, AeroOrigami generates all
AERO-S input files needed to simulate the fold. After the simulation runs, it maps
the folded displacements back onto the original fine mesh for use in downstream FSI
simulations.

## Dependencies

```bash
pip install numpy scipy matplotlib gmsh meshio
```

## Quickstart

```python
import pyaeroori

mesh     = pyaeroori.load_mesh("data/mesh.fem")
creases  = pyaeroori.load_creases("data/disk_creases.csv", "data/band_creases.csv")

surrogate = pyaeroori.remesh(mesh, creases, mesh_size=0.22)
surrogate = pyaeroori.build_surrogate(surrogate, creases, penalty_stiffness=8e9)
surrogate = pyaeroori.add_constraints(surrogate, mesh)

pyaeroori.write_aeros(surrogate, mesh, output_dir="results/")
```

See `examples/dgb_parachute/run.py` for a complete annotated driver script including
the post-fold displacement mapping step.

## Crease pattern CSV format

Each CSV has one row per fold line segment:

```
x1, y1, z1, x2, y2, z2, angle, type
```

- `x1,y1,z1` and `x2,y2,z2` are the 3D endpoints of the line segment (meters)
- `angle` is the target fold angle in radians — positive = mountain, negative = valley
- `type` is `C` (crease fold) or `B` (boundary edge — no folding, used to identify
  the perimeter of the crease region)

You can supply any number of CSV files to `load_creases()`; a typical case is one
file for the disk and one for the band.

## Project structure

```
AeroOrigami/
├── pyaeroori/          # The library
│   ├── mesh.py         # Step 1: load original mesh
│   ├── crease.py       # Step 2: load crease pattern
│   ├── remesh.py       # Step 3: Gmsh remeshing
│   ├── surrogate.py    # Step 4: build origami surrogate
│   ├── constraints.py  # Step 5: add LMPC + cable elements
│   ├── writer.py       # Step 6: write AEROS input files
│   └── mapping.py      # Step 7: map displacements back to fine mesh
├── examples/
│   └── dgb_parachute/  # Complete working example
│       └── run.py
└── docs/
    ├── architecture.md  # Pipeline overview and data structures
    └── design_notes.md  # Rationale behind key technical decisions
```

## Documentation

- [Architecture](docs/architecture.md) — pipeline overview, module layout, data structures
- [Design notes](docs/design_notes.md) — rationale behind technical decisions
