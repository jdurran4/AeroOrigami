"""
physics.py — Step 5 of the AeroOrigami pipeline.

add_physics(surrogate, ...) adds Dirichlet BCs, LMPC inequality constraints,
point forces, and cable elements to the surrogate model.

Returns a ModelConfig passed to write_aeros(..., config=config) to emit
DISP.include, LMPC.include, USDF.include + control.C, and cable elements
in ORIGAMI_MESH.include.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING
import math

import numpy as np

if TYPE_CHECKING:
    from .mesh import Mesh
    from .surrogate import Surrogate

_COORD_DIGITS = 4
_CABLE_ATTR   = 10   # attribute ID for cable bar elements in mesh_modified.include


# ── NodeQuery ─────────────────────────────────────────────────────────────────

class NodeQuery:
    """
    Lazy node selector resolved at add_physics time.

    Use the class-method constructors:

        N.near(x, y, z, tol=0.1)           — nodes within tol of a point
        N.along_line(p1, p2, tol=0.1)      — nodes within tol of a segment
        N.all()                             — every node in the surrogate
        N.ids([1, 2, 3])                    — exact node IDs

    When add_physics resolves a query it prints the matched nodes with
    coordinates so you can verify without a separate plotting run.
    """

    def __init__(self, _mode: str, _params: dict) -> None:
        self._mode   = _mode
        self._params = _params

    @classmethod
    def near(cls, x: float, y: float, z: float, tol: float = 1e-3) -> "NodeQuery":
        """Select nodes within `tol` metres of (x, y, z)."""
        return cls("near", {"x": x, "y": y, "z": z, "tol": tol})

    @classmethod
    def along_line(cls, p1, p2, tol: float = 1e-3) -> "NodeQuery":
        """Select nodes within perpendicular distance `tol` of segment p1→p2."""
        return cls("along_line", {"p1": tuple(p1), "p2": tuple(p2), "tol": tol})

    @classmethod
    def all(cls) -> "NodeQuery":
        """Select every node in the surrogate."""
        return cls("all", {})

    @classmethod
    def ids(cls, nids) -> "NodeQuery":
        """Select nodes by explicit ID list."""
        return cls("ids", {"nids": list(nids)})

    @classmethod
    def above(cls, *, x: float | None = None,
                       y: float | None = None,
                       z: float | None = None) -> "NodeQuery":
        """Select nodes where coordinate(s) are >= the given threshold(s)."""
        return cls("above", {"x": x, "y": y, "z": z})

    def resolve(self, nodes: dict[int, tuple], label: str = "") -> list[int]:
        """
        Evaluate this query against `nodes` (node_id → (x,y,z)).

        Prints matched nodes for immediate verification.
        Returns a sorted list of matching node IDs.
        """
        matched: list[int] = []

        if self._mode == "all":
            matched = list(nodes.keys())

        elif self._mode == "ids":
            for nid in self._params["nids"]:
                if nid in nodes:
                    matched.append(nid)
                else:
                    tag = f" [{label}]" if label else ""
                    print(f"  WARNING{tag}: node {nid} not found in surrogate — skipped")

        elif self._mode == "near":
            ref = np.array([self._params["x"], self._params["y"], self._params["z"]])
            tol = self._params["tol"]
            for nid, xyz in nodes.items():
                if np.linalg.norm(np.array(xyz) - ref) <= tol:
                    matched.append(nid)

        elif self._mode == "above":
            thx = self._params["x"]
            thy = self._params["y"]
            thz = self._params["z"]
            for nid, (nx, ny, nz) in nodes.items():
                if thx is not None and nx < thx:
                    continue
                if thy is not None and ny < thy:
                    continue
                if thz is not None and nz < thz:
                    continue
                matched.append(nid)

        elif self._mode == "along_line":
            p1  = np.array(self._params["p1"], dtype=float)
            p2  = np.array(self._params["p2"], dtype=float)
            tol = self._params["tol"]
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            for nid, (x, y, z) in nodes.items():
                pt = np.array([x, y, z])
                if seg_len_sq < 1e-24:
                    dist = float(np.linalg.norm(pt - p1))
                else:
                    t  = np.dot(pt - p1, seg) / seg_len_sq
                    t  = max(0.0, min(1.0, float(t)))
                    dist = float(np.linalg.norm(pt - (p1 + t * seg)))
                if dist <= tol:
                    matched.append(nid)

        matched.sort()
        tag = f" [{label}]" if label else ""
        print(f"  NodeQuery{tag}: matched {len(matched)} node(s)")
        for nid in matched[:10]:
            x, y, z = nodes[nid]
            print(f"    node {nid:6d}  ({x:.4f}, {y:.4f}, {z:.4f})")
        if len(matched) > 10:
            print(f"    ... and {len(matched) - 10} more")
        return matched


# Ergonomic alias: N.near(...), N.along_line(...), etc.
N = NodeQuery


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class LmpcRow:
    """
    One AERO-S LMPC inequality constraint block.

    Written as:
        <cid> <rhs> MODE 1
        <nid>  <dof>  <coeff>
        [<nid2> <dof2> <coeff2>]
    """
    cid:    int
    rhs:    float
    terms:  list[tuple[int, int, float]]   # [(nid, dof, coeff), ...]


@dataclass
class ModelConfig:
    """
    Output of add_physics() — BCs and loads written alongside the surrogate.

    Pass to write_aeros(surrogate, output_dir, config=config).

    cable_nodes    : new nodes to append to NODES (beyond surrogate.nodes)
    cable_elements : new bar elements (eid, etype=2, [na, nb]) for TOPOLOGY
    """
    disp_bcs:        list[tuple[int, list[int]]]           = field(default_factory=list)
    lmpc_rows:       list[LmpcRow]                         = field(default_factory=list)
    force_bcs:       list[tuple[int, float, float, float]] = field(default_factory=list)
    cable_nodes:     dict[int, tuple[float, float, float]] = field(default_factory=dict)
    cable_elements:  list[tuple[int, int, list[int]]]      = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def add_physics(
    surrogate: "Surrogate",
    mesh:      "Mesh | None" = None,
    disp:      list = (),
    lmpc:      list = (),
    forces:    list = (),
    cables:    list = (),
) -> ModelConfig:
    """
    Step 5: add boundary conditions, loads, and cable elements.

    Parameters
    ----------
    surrogate : Surrogate from build_surrogate()
    mesh      : original Mesh; required only for ``cables`` with block= entries
    disp      : Dirichlet BCs — list of (NodeQuery, dofs) tuples.
                dofs is a list of integers: [1]=x, [2]=y, [3]=z, [4]=rx, [5]=ry, [6]=rz.
                All co-located copies of each selected node are also pinned.
                Example::

                    disp=[
                        (N.near(0, 0, 0, tol=0.5), [1, 2, 3]),
                        (N.along_line((0,0,5),(1,0,5), tol=0.05), [3]),
                    ]

    lmpc      : Inequality constraints — list of dicts, key ``type`` selects behaviour:

                ``{"type": "min_z", "z_min": float}``
                  Prevent every membrane node from going below z_min.
                  Constraint: -u_z <= z0 - z_min  per node.

                ``{"type": "min_radius", "r_min": float}``
                  Prevent radial collapse. Linearized per-node:
                  -n_x*u_x - n_y*u_y <= r0 - r_min  where n = (x,y)/r.

                ``{"type": "custom", "nid": int, "dof": int, "coeff": float,
                   "rhs": float, "nid2": int, "dof2": int, "coeff2": float}``
                  Single or two-term custom constraint. nid2/dof2/coeff2 optional.

    forces    : Point forces — list of (NodeQuery, (fx, fy, fz)) tuples.
                Example::

                    forces=[
                        (N.near(0, 5, 0, tol=0.1), (0, 0, -100)),
                    ]

    cables    : Cable element chains — list of dicts.

                Each cable specification detects connected chains of 2-node
                elements and replaces each chain with a SINGLE type-203
                tension-only spring between the chain's two endpoint nodes.
                For a star topology (N lines converging at one confluence
                node), N springs are created, one per arm.

                ``{"block": "<name>", "tol": float}``
                  From a named block in the original mesh (requires mesh=).
                  Use this for DGB when block names are known and you want
                  to exclude canopy beam elements.

                ``{"blocks": ["<name1>", "<name2>", ...], "tol": float}``
                  Convenience form for multiple blocks in one entry.

                ``{"all_bars": True, "tol": float}``
                  From ALL 2-node elements in the original mesh (requires
                  mesh=). Use for meshes that don't have named blocks, or
                  when all 2-node elements are cables. Avoid on meshes that
                  embed beam elements inside the canopy surface.

                ``{"points": [(x1,y1,z1), ...], "tol": float}``
                  Explicit chain of points. A single spring is created
                  between the node nearest the first point and the node
                  nearest the last point. Intermediate points are ignored.
                  tol defaults to 1e-3.

    Returns
    -------
    ModelConfig
    """
    config = ModelConfig()

    # Build coord → [node_ids] index for co-located copy detection
    coord_map: dict[tuple, list[int]] = defaultdict(list)
    for nid, xyz in surrogate.nodes.items():
        coord_map[_rp(xyz)].append(nid)

    # ── Cable elements (processed first so endpoints exist for DISP resolution) ─
    if cables:
        print("\n  -- CABLES --")
    next_nid = max(surrogate.nodes) + 1
    next_eid = max(surrogate.elements) + 1

    for spec in cables:
        tol = float(spec.get("tol", 1e-3))

        if "block" in spec:
            if mesh is None:
                print(f"  WARNING: cable block='{spec['block']}' requires mesh= parameter — skipped")
                continue
            _add_cable_chains_from_elems(
                spec["block"], mesh, surrogate, config, next_nid, next_eid, tol
            )

        elif "blocks" in spec:
            if mesh is None:
                print(f"  WARNING: cable blocks={spec['blocks']!r} requires mesh= parameter — skipped")
                continue
            for bname in spec["blocks"]:
                _add_cable_chains_from_elems(
                    bname, mesh, surrogate, config, next_nid, next_eid, tol
                )

        elif spec.get("all_bars"):
            if mesh is None:
                print("  WARNING: all_bars=True requires mesh= parameter — skipped")
                continue
            _add_all_cable_chains(
                mesh, surrogate, config, next_nid, next_eid, tol
            )

        elif "points" in spec:
            pts   = [tuple(float(v) for v in p) for p in spec["points"]]
            label = spec.get("label", "explicit")
            _add_cable_from_points(pts, tol, label, surrogate, config, next_nid, next_eid)

        else:
            print(f"  WARNING: cable entry missing 'block', 'blocks', 'all_bars', or 'points' — skipped: {spec}")

        # Advance counters after any additions
        if config.cable_nodes:
            next_nid = max(config.cable_nodes) + 1
        if config.cable_elements:
            next_eid = config.cable_elements[-1][0] + 1

    # Combined node lookup — surrogate nodes + cable endpoint nodes added above
    all_nodes = {**surrogate.nodes, **config.cable_nodes}

    # ── DISP Dirichlet BCs ────────────────────────────────────────────────────
    if disp:
        print("\n  -- DISP --")
    for entry in disp:
        query, dofs = entry
        if isinstance(query, NodeQuery):
            primary = query.resolve(all_nodes, label="DISP")
        else:
            primary = sorted(query)

        # Expand to co-located copies (surrogate nodes only; cable nodes have none)
        all_nids: set[int] = set()
        for nid in primary:
            copies = coord_map.get(_rp(all_nodes[nid]))
            if copies:
                all_nids.update(copies)
            else:
                all_nids.add(nid)

        for nid in sorted(all_nids):
            config.disp_bcs.append((nid, list(dofs)))

    # ── LMPC inequality constraints ───────────────────────────────────────────
    if lmpc:
        print("\n  -- LMPC --")
    cid = 1
    for spec in lmpc:
        ctype = spec.get("type", "custom")

        if ctype == "min_z":
            z_min = float(spec["z_min"])
            target_nids = _resolve_lmpc_nodes(spec, all_nodes, surrogate)
            count = 0
            for nid in sorted(target_nids):
                x, y, z0 = all_nodes[nid]
                rhs = -(z_min - z0)          # = z0 - z_min
                config.lmpc_rows.append(
                    LmpcRow(cid=cid, rhs=rhs, terms=[(nid, 3, -1.0)])
                )
                cid += 1
                count += 1
            print(f"  LMPC min_z={z_min}: {count} constraints added")

        elif ctype == "min_radius":
            r_min = float(spec["r_min"])
            target_nids = _resolve_lmpc_nodes(spec, all_nodes, surrogate)
            count = 0
            for nid in sorted(target_nids):
                x, y, z = all_nodes[nid]
                r0 = math.sqrt(x * x + y * y)
                if r0 < 1e-12:
                    continue
                ax = x / r0
                ay = y / r0
                rhs = -(r_min - r0)          # = r0 - r_min
                config.lmpc_rows.append(
                    LmpcRow(cid=cid, rhs=rhs, terms=[(nid, 1, -ax), (nid, 2, -ay)])
                )
                cid += 1
                count += 1
            print(f"  LMPC min_radius={r_min}: {count} constraints added")

        elif ctype == "custom":
            nid   = int(spec["nid"])
            dof   = int(spec["dof"])
            coeff = float(spec["coeff"])
            rhs   = float(spec["rhs"])
            terms = [(nid, dof, coeff)]
            if "nid2" in spec:
                terms.append((int(spec["nid2"]), int(spec["dof2"]),
                               float(spec["coeff2"])))
            config.lmpc_rows.append(LmpcRow(cid=cid, rhs=rhs, terms=terms))
            cid += 1

        else:
            print(f"  WARNING: unknown LMPC type '{ctype}' — skipped")

    # ── Point forces ──────────────────────────────────────────────────────────
    if forces:
        print("\n  -- FORCES --")
    for entry in forces:
        query, fvec = entry
        fx, fy, fz = float(fvec[0]), float(fvec[1]), float(fvec[2])
        if isinstance(query, NodeQuery):
            nids = query.resolve(all_nodes, label="FORCE")
        else:
            nids = sorted(query)
        for nid in nids:
            config.force_bcs.append((nid, fx, fy, fz))

    print()
    print(f"  ModelConfig: {len(config.disp_bcs)} DISP BCs, "
          f"{len(config.lmpc_rows)} LMPC rows, "
          f"{len(config.force_bcs)} force BCs, "
          f"{len(config.cable_nodes)} new cable nodes, "
          f"{len(config.cable_elements)} cable elements")
    return config


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_lmpc_nodes(
    spec:      dict,
    all_nodes: dict,
    surrogate: "Surrogate",
) -> set[int]:
    """Return the node IDs an LMPC constraint applies to.

    If the spec has a ``nodes`` key (NodeQuery or iterable of IDs), resolve it
    against all_nodes.  Otherwise fall back to all membrane nodes in the surrogate.
    """
    if "nodes" in spec:
        query = spec["nodes"]
        if isinstance(query, NodeQuery):
            return set(query.resolve(all_nodes, label="LMPC"))
        return set(query)
    return _membrane_nids(surrogate)


def _rp(xyz) -> tuple:
    return (round(xyz[0], _COORD_DIGITS),
            round(xyz[1], _COORD_DIGITS),
            round(xyz[2], _COORD_DIGITS))


def _membrane_nids(surrogate: "Surrogate") -> set[int]:
    nids: set[int] = set()
    for eid, (etype, ns) in surrogate.elements.items():
        if len(ns) >= 3:
            nids.update(ns)
    return nids


def _find_or_add_node(
    pt:          tuple,
    surr_nodes:  dict,
    cable_nodes: dict,
    next_nid:    list,   # [int] — mutable counter
    tol:         float,
    label:       str,
) -> int:
    """Return ID of nearest node within tol (surrogate or new cable), or add a new one."""
    ref = np.array(pt, dtype=float)
    best_nid, best_dist = None, float("inf")
    for nid, xyz in surr_nodes.items():
        d = float(np.linalg.norm(np.array(xyz) - ref))
        if d < best_dist:
            best_dist, best_nid = d, nid
    for nid, xyz in cable_nodes.items():
        d = float(np.linalg.norm(np.array(xyz) - ref))
        if d < best_dist:
            best_dist, best_nid = d, nid

    if best_dist <= tol:
        return best_nid

    # No close node — add a new one
    nid = next_nid[0]
    next_nid[0] += 1
    cable_nodes[nid] = (float(pt[0]), float(pt[1]), float(pt[2]))
    print(f"  Cable [{label}]: added node {nid} at "
          f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
          f"  (nearest was {best_dist:.4f} m away)")
    return nid


def _build_cable_chains(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Given a list of (na, nb) node-ID pairs from bar elements, return the
    (start_node, end_node) endpoints of each cable chain.

    Algorithm: degree-1 leaf nodes are the true cable endpoints. From each
    leaf, trace through degree-2 nodes until hitting another leaf or a
    junction (degree ≠ 2). Each traced path becomes one spring.

    Handles both simple chains (2 leaves) and star/tree topologies (N arms
    converging at a shared confluence node).
    """
    adj: dict[int, set[int]] = defaultdict(set)
    for na, nb in pairs:
        adj[na].add(nb)
        adj[nb].add(na)

    all_nodes = set(adj)
    if not all_nodes:
        return []

    degree = {n: len(adj[n]) for n in all_nodes}
    leaves = sorted(n for n, d in degree.items() if d == 1)

    chains: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    visited_edges: set[tuple[int, int]] = set()

    for start in leaves:
        path = [start]
        prev: int | None = None
        cur = start
        while True:
            nexts = adj[cur] - ({prev} if prev is not None else set())
            if not nexts:
                break
            nxt = min(nexts)   # deterministic
            edge = (min(cur, nxt), max(cur, nxt))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(nxt)
            prev, cur = cur, nxt
            if degree[nxt] != 2:
                break  # reached another leaf or a junction

        if len(path) >= 2:
            key = (min(start, path[-1]), max(start, path[-1]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                chains.append((start, path[-1]))

    if not chains:
        # Fallback: no degree-1 nodes (closed loops); return nothing with warning
        print("  WARNING: cable chain detection found no leaf nodes (closed loop?)")

    return chains


def _counters(config: ModelConfig, nid0: int, eid0: int):
    """Return fresh [nid], [eid] counters adjusted for any prior cable additions."""
    nid = max(nid0, (max(config.cable_nodes) + 1) if config.cable_nodes else nid0)
    eid = max(eid0, (config.cable_elements[-1][0] + 1) if config.cable_elements else eid0)
    return [nid], [eid]


def _add_cable_chains_from_elems(
    block_name:  str,
    mesh:        "Mesh",
    surrogate:   "Surrogate",
    config:      ModelConfig,
    next_nid:    int,
    next_eid:    int,
    tol:         float = 1e-3,
) -> None:
    """Map a named mesh block's 2-node elements to surrogate as chain-end springs."""
    block_eids = mesh.blocks.get(block_name)
    if not block_eids:
        print(f"  WARNING: block '{block_name}' not found in mesh — skipped")
        return

    pairs = [
        (mesh.elements[s][1][0], mesh.elements[s][1][1])
        for s in block_eids
        if len(mesh.elements[s][1]) == 2
    ]
    if not pairs:
        print(f"  WARNING: block '{block_name}': no 2-node elements — skipped")
        return

    _nid, _eid = _counters(config, next_nid, next_eid)
    n_added = 0
    for na, nb in _build_cable_chains(pairs):
        sid_a = _find_or_add_node(mesh.nodes[na], surrogate.nodes, config.cable_nodes, _nid, tol, block_name)
        sid_b = _find_or_add_node(mesh.nodes[nb], surrogate.nodes, config.cable_nodes, _nid, tol, block_name)
        config.cable_elements.append((_eid[0], 2, [sid_a, sid_b]))
        _eid[0] += 1
        n_added += 1

    print(f"  Cable block '{block_name}': {len(pairs)} bars → {n_added} spring(s)")


def _add_all_cable_chains(
    mesh:        "Mesh",
    surrogate:   "Surrogate",
    config:      ModelConfig,
    next_nid:    int,
    next_eid:    int,
    tol:         float = 1e-3,
) -> None:
    """Map ALL 2-node elements in the mesh to surrogate as chain-end springs."""
    cable_elems = mesh.cable_elements
    if not cable_elems:
        print("  Cable all_bars: no 2-node elements found in mesh")
        return

    pairs = [(nids[0], nids[1]) for _, nids in cable_elems.values()]
    chains = _build_cable_chains(pairs)

    _nid, _eid = _counters(config, next_nid, next_eid)
    n_added = 0
    for na, nb in chains:
        sid_a = _find_or_add_node(mesh.nodes[na], surrogate.nodes, config.cable_nodes, _nid, tol, "all_bars")
        sid_b = _find_or_add_node(mesh.nodes[nb], surrogate.nodes, config.cable_nodes, _nid, tol, "all_bars")
        config.cable_elements.append((_eid[0], 2, [sid_a, sid_b]))
        _eid[0] += 1
        n_added += 1

    print(f"  Cable all_bars: {len(cable_elems)} bars → {n_added} spring(s)")


def _add_cable_from_points(
    pts:         list[tuple],
    tol:         float,
    label:       str,
    surrogate:   "Surrogate",
    config:      ModelConfig,
    next_nid:    int,
    next_eid:    int,
) -> None:
    """Create ONE tension-only spring between the first and last point in the list."""
    if len(pts) < 2:
        print(f"  WARNING: cable '{label}' needs at least 2 points — skipped")
        return

    _nid, _eid = _counters(config, next_nid, next_eid)
    sid_a = _find_or_add_node(pts[0],  surrogate.nodes, config.cable_nodes, _nid, tol, label)
    sid_b = _find_or_add_node(pts[-1], surrogate.nodes, config.cable_nodes, _nid, tol, label)
    config.cable_elements.append((_eid[0], 2, [sid_a, sid_b]))

    print(f"  Cable '{label}': 1 spring  node {sid_a} → node {sid_b}"
          f"  ({len(pts)} intermediate points ignored)")
