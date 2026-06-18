# AeroOrigami Architecture

## Overview

AeroOrigami is a Python library for setting up origami-inspired folding simulations
of deployable structures in AERO-S. It takes a high-fidelity mesh and a crease
pattern, and produces all AERO-S input files needed to simulate the fold. After the
fold simulation runs, it maps the folded displacements back onto the original fine
mesh.

The library lives in the `pyaeroori` package. Users interact with it through a
short driver script (see `examples/dgb_parachute/run.py`).

---

## The 7-Step Pipeline

```
Original mesh + crease CSVs
        │
        ▼
[1] load_mesh       → Mesh object
[2] load_creases    → CreasePattern object
[3] remesh          → Surrogate object (Gmsh-generated)
[4] build_surrogate → Surrogate object (with joints + actuators)
[5] add_constraints → Surrogate object (with LMPC + cable springs)
[6] write_aeros     → AEROS .include files on disk
        │
        │   (user runs AERO-S folding simulation externally)
        │
[7] map_displacements → fine-mesh displacement dict
    write_idisp6      → IDISP6.include on disk
```

---

## Module Layout

| Module | Step | Responsibility |
|---|---|---|
| `pyaeroori/mesh.py` | 1 | Parse AERO-S `.fem` / `.include` files into Python data structures |
| `pyaeroori/crease.py` | 2 | Load crease CSVs; build CreasePattern |
| `pyaeroori/remesh.py` | 3 | Drive Gmsh to generate a crease-aligned mesh |
| `pyaeroori/surrogate.py` | 4 | Panel detection, node duplication, hinge joints, actuators, EFRAMES |
| `pyaeroori/constraints.py` | 5 | LMPC writers, cable spring elements, additional boundary conditions |
| `pyaeroori/writer.py` | 6 | Assemble and write all AERO-S input files |
| `pyaeroori/mapping.py` | 7 | RBF interpolation, cable path reconstruction, IDISP6 writer |

---

## Key Data Structures

These are the objects passed between pipeline steps. They are plain Python
dataclasses (or dicts — TBD during implementation), not classes with hidden state.

### `Mesh`
The original high-fidelity mesh. Read-only after Step 1; never modified by later steps.

```python
Mesh:
    nodes      : dict[int, tuple[float,float,float]]  # node_id → (x, y, z)
    elements   : dict[int, tuple[int, list[int]]]      # elem_id → (etype, [node_ids])
    attributes : dict[int, int]                        # elem_id → attr_id (if present)
```

The only required sections in the input file are `NODES` and `TOPOLOGY`.
An `ATTRIBUTES` section is optional and preserved if present.
No specific block names or mesh structure is assumed.

**Identifying the origami surface**

Steps 3–5 need to know which nodes belong to the foldable surface (membranes) vs.
other components (cables, payload, etc.). By default, membrane nodes are
auto-detected as all nodes belonging to 3- or 4-node elements. For meshes where
this heuristic is insufficient, the user can pass an explicit set:

```python
# Default: auto-detect membrane nodes from element connectivity
surrogate = pyaeroori.remesh(mesh, creases, mesh_size=0.22)

# Override: specify surface nodes explicitly
surrogate = pyaeroori.remesh(mesh, creases, mesh_size=0.22,
                             surface_nodes=[1, 2, 3, ...])

# Override: filter by ATTRIBUTES section value
surrogate = pyaeroori.remesh(mesh, creases, mesh_size=0.22,
                             surface_attribute=1)
```

Similarly for the displacement mapping step, cable elements are auto-detected as
2-node elements. The same override pattern applies.

### `CreasePattern`
Fold line segments and boundary edges loaded from one or more CSV files.

```python
CreasePattern:
    mountain : list[tuple[Point3D, Point3D, float]]  # (p1, p2, angle_radians)
    valley   : list[tuple[Point3D, Point3D, float]]  # (p1, p2, angle_radians)
    boundary : list[tuple[Point3D, Point3D]]          # perimeter edges (no folding)
```

Mountain and valley are separated internally based on the sign of `angle`.
Boundary edges define the perimeter of the crease region (e.g., outer rim, vent
opening, band edges) and are used to identify boundary crease nodes, which receive
spherical joints instead of revolute joints.

**CSV format** — one row per line segment:
```
x1, y1, z1, x2, y2, z2, angle, type
```
- `angle > 0` → mountain fold; `angle < 0` → valley fold
- `type = C` → crease fold; `type = B` → boundary edge (angle ignored)

Any number of CSV files can be passed to `load_creases()`; rows from all files are
merged into a single `CreasePattern`.

### `Surrogate`
The origami surrogate mesh. Built up progressively through Steps 3–5.

```python
Surrogate:
    nodes        : dict[int, tuple[float,float,float]]
    elements     : dict[int, tuple[int, list[int]]]    # elem_id → (etype, [node_ids])
    attributes   : dict[int, int]                       # elem_id → attr_id
    crease_edges : set[tuple[int,int]]
    panels       : list[set[int]]                       # sets of element IDs
    panel_colors : dict[int, int]                       # panel_id → 0 or 1
    dup_map      : dict[int, int]                       # orig_node → dup_node
    joint_pairs  : list[tuple[int,int]]
    hinge_axes   : list[tuple[int,int,int,np.ndarray]]  # (eid, n1, n2, axis)
    lmpc_lines   : list[str]                            # raw LMPC text lines
    materials    : list[str]                            # raw MATERIAL text lines
```

---

## AERO-S File Format Notes

AERO-S `.include` / `.fem` files use a section-based text format:

```
NODES
  <node_id>  <x>  <y>  <z>
*
TOPOLOGY
  <elem_id>  <elem_type>  <node_id_1>  <node_id_2>  ...
*
ATTRIBUTES
  <elem_id>  <attr_id>
*
```

Section boundaries use `*` as a terminator. Named blocks (optional) use the
header `*  name: BlockName` before their sections. The parser in `mesh.py`
handles both named-block files and flat files with a single NODES/TOPOLOGY pair.
