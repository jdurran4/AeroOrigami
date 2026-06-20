"""
mesh.py — Step 1 of the AeroOrigami pipeline.

Parses an AERO-S .fem / .include mesh file into a Mesh object.
Only NODES and TOPOLOGY sections are read; everything else (ATTRIBUTES,
EFRAMES, block headers, comment lines) is silently ignored.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mesh:
    """
    AERO-S mesh — used both for the original hi-fi mesh (Step 1) and the
    coarse origami surrogate mesh produced by remesh() (Step 3).

    nodes     : node_id → (x, y, z)
    elements  : elem_id → (etype, [node_ids])
                elem_ids are globally unique integers, starting at 1.
    panel_map : elem_id → panel_id (populated by remesh(); empty otherwise)
    """
    nodes:      dict[int, tuple[float, float, float]]
    elements:   dict[int, tuple[int, list[int]]]
    panel_map:  dict[int, int]  = field(default_factory=dict)
    path:       str             = field(default="")   # "A" (Gmsh) or "B" (crease-as-mesh)
    blocks:     dict[str, list[int]] = field(default_factory=dict)  # block name → [seq_eid]
    attributes: dict[int, int]  = field(default_factory=dict)       # seq_eid → attr_id

    # ------------------------------------------------------------------
    # Derived views (computed on access, not stored)
    # ------------------------------------------------------------------

    @property
    def membrane_elements(self) -> dict[int, tuple[int, list[int]]]:
        """Elements with 3 or more nodes (triangles, quads)."""
        return {eid: v for eid, v in self.elements.items() if len(v[1]) >= 3}

    @property
    def cable_elements(self) -> dict[int, tuple[int, list[int]]]:
        """Elements with exactly 2 nodes (bars, cables)."""
        return {eid: v for eid, v in self.elements.items() if len(v[1]) == 2}

    @property
    def membrane_nodes(self) -> set[int]:
        """Node IDs belonging to at least one membrane element."""
        return {nid for _, nodes in self.membrane_elements.values() for nid in nodes}

    @property
    def cable_nodes(self) -> set[int]:
        """Node IDs belonging to cable elements only (not shared with membranes)."""
        return (
            {nid for _, nodes in self.cable_elements.values() for nid in nodes}
            - self.membrane_nodes
        )

    def __repr__(self) -> str:
        return (
            f"Mesh("
            f"nodes={len(self.nodes)}, "
            f"membrane_elements={len(self.membrane_elements)}, "
            f"cable_elements={len(self.cable_elements)})"
        )


def load_mesh(filepath: str | Path) -> Mesh:
    """
    Parse an AERO-S .fem / .include file and return a Mesh.

    The parser handles two file layouts:

      Flat (simple meshes):
        NODES
          <id>  <x>  <y>  <z>
        *
        TOPOLOGY
          <id>  <etype>  <node> ...
        *

      Blocked (complex meshes with named component blocks):
        NODES
          ...
        *
        *  name: Band_Edge_Leading
        *  ...
        *
        TOPOLOGY
          ...
        ATTRIBUTES
          ...        <- ends topology section, ignored
        EFRAMES
          ...        <- ignored
        *
        *  name: next block ...

    Element IDs in the source file are discarded and replaced with
    globally unique integers starting from 1. This is necessary because
    blocked files restart element numbering at 1 for each block.

    Parameters
    ----------
    filepath : path to the mesh file

    Returns
    -------
    Mesh

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if no NODES or no TOPOLOGY section is found
    """
    filepath = Path(filepath)

    nodes:      dict[int, tuple[float, float, float]] = {}
    elements:   dict[int, tuple[int, list[int]]] = {}
    attributes: dict[int, int] = {}
    blocks:     dict[str, list[int]] = {}
    next_eid    = 1
    section     = None          # 'nodes' | 'topology' | 'attributes' | None
    orig_to_seq: dict[int, int] = {}   # block-local eid → sequential eid
    current_block: str | None  = None  # active block name (from *  name: ...)

    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Section keywords
            if line == "NODES":
                section = "nodes"
                continue
            if line == "TOPOLOGY":
                section = "topology"
                orig_to_seq = {}        # reset mapping; each block restarts at eid=1
                continue
            if line == "ATTRIBUTES":
                section = "attributes"
                continue
            if line == "EFRAMES" or line.startswith("*"):
                # Detect block name header before resetting section
                if line.startswith("*") and "name:" in line:
                    current_block = line.split("name:")[-1].strip()
                    if current_block not in blocks:
                        blocks[current_block] = []
                section = None
                continue

            parts = line.split()

            if section == "nodes":
                if len(parts) >= 4:
                    nid = int(parts[0])
                    nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))

            elif section == "topology":
                if len(parts) >= 3:
                    orig_eid  = int(parts[0])
                    etype     = int(parts[1])
                    node_ids  = [int(p) for p in parts[2:]]
                    elements[next_eid] = (etype, node_ids)
                    orig_to_seq[orig_eid] = next_eid
                    if current_block is not None:
                        blocks[current_block].append(next_eid)
                    next_eid += 1

            elif section == "attributes":
                if len(parts) >= 2:
                    orig_eid = int(parts[0])
                    attr_id  = int(parts[1])
                    seq_eid  = orig_to_seq.get(orig_eid)
                    if seq_eid is not None:
                        attributes[seq_eid] = attr_id

    if not nodes:
        raise ValueError(f"No NODES section found in {filepath}")
    if not elements:
        raise ValueError(f"No TOPOLOGY section found in {filepath}")

    return Mesh(nodes=nodes, elements=elements,
                blocks=blocks, attributes=attributes)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python mesh.py <path/to/mesh.fem>")
        sys.exit(1)

    mesh = load_mesh(path)
    print(mesh)
    print(f"  Node ID range   : {min(mesh.nodes)} – {max(mesh.nodes)}")
    print(f"  Membrane nodes  : {len(mesh.membrane_nodes)}")
    print(f"  Cable nodes     : {len(mesh.cable_nodes)}")
