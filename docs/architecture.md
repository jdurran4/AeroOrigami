# AeroOrigami Architecture

## Overview

AeroOrigami is a Python library for setting up origami-inspired folding simulations
of parachute structures in AERO-S. It takes a high-fidelity parachute mesh and a
crease pattern, and produces all AERO-S input files needed to simulate the fold.
After the fold simulation runs, it maps the folded displacements back onto the
original fine mesh.

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
| `pyaeroori/crease.py` | 2 | Load mountain/valley crease CSVs; build CreasePattern |
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
The original high-fidelity parachute mesh. Read-only after Step 1; never modified.

```python
Mesh:
    nodes    : dict[int, tuple[float,float,float]]  # node_id → (x, y, z)
    blocks   : dict[str, dict]                       # block_name → {TOPOLOGY, ATTRIBUTES}
```

Named blocks follow AERO-S DGB convention:
`Disk_Gores`, `Band_Gores`, `Disk_Edge_Leading`, `Disk_Edge_Trailing`,
`Band_Edge_Leading`, `Band_Edge_Trailing`, `Suspension_Lines`, `Vent_Lines`,
`Gap_Lines`, `Riser_Lines`.

### `CreasePattern`
Mountain and valley fold line segments with per-fold target angles.

```python
CreasePattern:
    mountain : list[tuple[Point3D, Point3D, float]]  # (p1, p2, angle_radians)
    valley   : list[tuple[Point3D, Point3D, float]]  # (p1, p2, angle_radians)
```

Loaded from two CSV files (disk + band) with columns:
`x1, y1, z1, x2, y2, z2, type, angle`
where `type` is `M` (mountain) or `V` (valley).

### `Surrogate`
The origami surrogate mesh. Built up progressively through Steps 3–5.

```python
Surrogate:
    nodes        : dict[int, tuple[float,float,float]]
    elements     : dict[int, tuple[int, list[int]]]   # elem_id → (etype, [node_ids])
    attributes   : dict[int, int]                      # elem_id → attr_id
    crease_edges : set[tuple[int,int]]
    panels       : list[set[int]]                      # sets of element IDs
    panel_colors : dict[int, int]                      # panel_id → 0 or 1
    dup_map      : dict[int, int]                      # orig_node → dup_node
    joint_pairs  : list[tuple[int,int]]
    hinge_axes   : list[tuple[int,int,int,np.ndarray]] # (eid, n1, n2, axis)
    lmpc_lines   : list[str]                           # raw LMPC text lines
    materials    : list[str]                           # raw MATERIAL text lines
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

Block boundaries use `*` as a section terminator. Named blocks use the header
`*  name: BlockName` before their sections. All parsers live in `mesh.py`.
