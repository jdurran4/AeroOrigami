"""
surrogate.py — Step 4 of the AeroOrigami pipeline.

build_surrogate(coarse, creases, ...) takes the panel-mapped coarse mesh from
Step 3 and produces the AERO-S fold surrogate: duplicated crease nodes, driver
joint elements with hinge axes, and per-joint actuator parameters.

The returned Surrogate object is passed to write_aeros (Step 6) which writes
the actual AERO-S include files.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mesh import Mesh
    from .crease import CreasePattern

# rounding decimals for coordinate key matching
_COORD_DIGITS = 4


@dataclass
class JointInfo:
    """One joint element connecting two copies of a duplicated crease node."""
    eid:          int
    node_a:       int           # lower-panel-id side
    node_b:       int           # higher-panel-id side (or dup)
    jtype:        int           # 120 = spherical, 126 = revolute driver
    axis:         tuple | None  # unit crease axis (revolute only)
    target_angle: float         # signed radians (+mountain, -valley)
    start_time:   float         # actuator ramp start
    end_time:     float         # actuator ramp end
    penalty:      float         # CONMAT penalty stiffness


@dataclass
class Surrogate:
    """
    Step 4 output — the AERO-S fold surrogate.

    nodes     : all mesh nodes, including duplicated crease nodes
    elements  : shell element connectivity after crease-node duplication
    joints    : list of JointInfo (revolute and spherical)
    panel_map : element → panel_id (shell elements only; joints not included)
    penalty_stiffness  : global default penalty stiffness
    actuator_ramp_time : global default actuator ramp end time
    """
    nodes:             dict[int, tuple[float, float, float]]
    elements:          dict[int, tuple[int, list[int]]]
    joints:            list[JointInfo]
    panel_map:         dict[int, int]
    penalty_stiffness: float
    actuator_ramp_time: float

    @property
    def revolute_joints(self) -> list[JointInfo]:
        return [j for j in self.joints if j.jtype == 126]

    @property
    def spherical_joints(self) -> list[JointInfo]:
        return [j for j in self.joints if j.jtype == 120]


# ── Public API ────────────────────────────────────────────────────────────────

def build_surrogate(
    coarse:             "Mesh",
    creases:            "CreasePattern",
    penalty_stiffness:  float = 8e9,
    actuator_ramp_time: float = 3.0,
    vertex_joint_type:  int | None = None,
) -> Surrogate:
    """
    Step 4: duplicate crease nodes, build driver joints, compute hinge axes.

    Parameters
    ----------
    coarse             : Mesh with panel_map populated (output of remesh())
    creases            : CreasePattern with fold angles and optional timing
    penalty_stiffness  : CONMAT penalty stiffness for all joints
    actuator_ramp_time : default actuator end time (used when CSV omits end_time)
    vertex_joint_type  : joint type for crease-segment endpoint nodes
                         120 = spherical (no actuation, free rotation)
                         126 = revolute driver (actuated)
                         None (default) = auto: 126 for Path B (crease-as-mesh),
                           120 for Path A (Gmsh remesh).
                         In Path A, Gmsh-added interior crease nodes always get
                         126 regardless of this setting — only the segment
                         endpoint/junction nodes are affected.

    Returns
    -------
    Surrogate
    """
    if not coarse.panel_map:
        raise ValueError(
            "coarse.panel_map is empty — run remesh() before build_surrogate()"
        )

    # ── 1. node → panels ─────────────────────────────────────────────────────
    n2p: dict[int, set[int]] = defaultdict(set)
    for eid, (etype, nids) in coarse.elements.items():
        if len(nids) < 3:
            continue
        pid = coarse.panel_map.get(eid)
        if pid is None:
            continue
        for nid in nids:
            n2p[nid].add(pid)

    crease_nids: set[int] = {v for v, ps in n2p.items() if len(ps) >= 2}

    # ── 2. Resolve vertex joint type ─────────────────────────────────────────
    # Path B default: 126 (revolute) — every node is a CSV endpoint, axes known.
    # Path A default: 120 (spherical) for endpoint/junction nodes; Gmsh-added
    #   interior nodes always get 126 regardless of this setting.
    path = getattr(coarse, "path", "")
    if vertex_joint_type is None:
        _vtx_jtype = 126 if path == "B" else 120
    else:
        _vtx_jtype = vertex_joint_type

    # Set of crease-CSV endpoint coords (rounded) for Path A node classification.
    _csv_eps: set[tuple] = set()
    for entry in creases.mountain + creases.valley:
        _csv_eps.add(_rp(entry[0]))
        _csv_eps.add(_rp(entry[1]))
    for p1, p2 in creases.boundary:
        _csv_eps.add(_rp(p1))
        _csv_eps.add(_rp(p2))

    # ── 3. Crease edges: mesh edges shared by two different panels ────────────
    edge_panels: dict[tuple[int, int], set[int]] = defaultdict(set)
    for eid, (etype, nids) in coarse.elements.items():
        if len(nids) < 3:
            continue
        pid = coarse.panel_map.get(eid)
        if pid is None:
            continue
        n = len(nids)
        for i in range(n):
            e = _edge(nids[i], nids[(i + 1) % n])
            edge_panels[e].add(pid)

    crease_edges: dict[tuple[int, int], frozenset[int]] = {
        e: frozenset(ps) for e, ps in edge_panels.items() if len(ps) == 2
    }

    # node → [(other_node, panel_pair)]
    node_cedges: dict[int, list[tuple[int, frozenset[int]]]] = defaultdict(list)
    for (va, vb), ps in crease_edges.items():
        node_cedges[va].append((vb, ps))
        node_cedges[vb].append((va, ps))

    # ── 4. Duplicate crease nodes ─────────────────────────────────────────────
    # dup_map[(orig_nid, panel_id)] = node_id to use for that panel
    # Owner panel (lowest pid at that node) keeps the original node.
    dup_map: dict[tuple[int, int], int] = {}
    new_nodes: dict[int, tuple[float, float, float]] = dict(coarse.nodes)
    next_nid = max(coarse.nodes) + 1

    for v in crease_nids:
        owner = min(n2p[v])
        for pid in n2p[v]:
            if pid == owner:
                dup_map[(v, pid)] = v
            else:
                dup_map[(v, pid)] = next_nid
                new_nodes[next_nid] = coarse.nodes[v]
                next_nid += 1

    # ── 5. Update element connectivity ───────────────────────────────────────
    new_elements: dict[int, tuple[int, list[int]]] = {}
    for eid, (etype, nids) in coarse.elements.items():
        pid = coarse.panel_map.get(eid)
        if pid is None:
            new_elements[eid] = (etype, list(nids))
            continue
        new_nids = [dup_map.get((nid, pid), nid) for nid in nids]
        new_elements[eid] = (etype, new_nids)

    # ── 6. Crease-segment angle lookup ───────────────────────────────────────
    # Primary: key (rounded_p1, rounded_p2) → (angle, start_t, end_t)
    # Both directions stored.  Used when both endpoints match CSV endpoints.
    #
    # Fallback: seg_list of (p1_arr, p2_arr, angle, start_t, end_t) for nodes
    # that lie on the INTERIOR of a through-crease (T-junction split points).
    seg_angle: dict[tuple, tuple[float, float, float]] = {}
    seg_list: list[tuple] = []
    for entry in creases.mountain + creases.valley:
        p1, p2, angle = entry[0], entry[1], entry[2]
        start_t = entry[3] if (len(entry) > 3 and entry[3] is not None) else 0.0
        end_t   = entry[4] if (len(entry) > 4 and entry[4] is not None) else actuator_ramp_time
        k1, k2 = _rp(p1), _rp(p2)
        seg_angle[(k1, k2)] = (angle, start_t, end_t)
        seg_angle[(k2, k1)] = (angle, start_t, end_t)
        seg_list.append((np.asarray(p1, float), np.asarray(p2, float),
                         angle, start_t, end_t))

    # ── 7. Build joint elements ───────────────────────────────────────────────
    joints: list[JointInfo] = []
    next_eid = max(new_elements) + 1
    seen_node_pairs: set[tuple[int, int]] = set()

    for v in sorted(crease_nids):
        v_xyz = np.array(coarse.nodes[v])

        # Group incident crease edges by panel pair so that multiple collinear
        # segments between the same two panels at v produce ONE joint.
        pair_others: dict[frozenset[int], list[int]] = defaultdict(list)
        for other, ps in node_cedges[v]:
            pair_others[ps].append(other)

        for ps, others in pair_others.items():
            PA, PB = sorted(ps)
            node_a = dup_map.get((v, PA), v)
            node_b = dup_map.get((v, PB), v)

            # Skip if this node pair already got a joint (e.g. both ends of a
            # collinear crease produced the same physical node pair).
            pair_key = (min(node_a, node_b), max(node_a, node_b))
            if pair_key in seen_node_pairs:
                continue
            seen_node_pairs.add(pair_key)

            # Joint type:
            #   Path B — always use _vtx_jtype (default 126, revolute)
            #   Path A — Gmsh interior nodes (not a CSV endpoint) always get 126;
            #             CSV endpoint/junction nodes get _vtx_jtype (default 120)
            v_is_endpoint = _rp(coarse.nodes[v]) in _csv_eps
            if path == "A" and not v_is_endpoint:
                jtype = 126
            else:
                jtype = _vtx_jtype

            # Crease axis: direction from v toward the first other endpoint.
            # For collinear segments, pick the longest one for a stable axis.
            best_other = max(
                others,
                key=lambda w: np.linalg.norm(np.array(coarse.nodes[w]) - v_xyz),
            )
            d = np.array(coarse.nodes[best_other]) - v_xyz
            norm = float(np.linalg.norm(d))
            axis = tuple(float(x) for x in d / norm) if norm > 1e-12 else (1.0, 0.0, 0.0)

            # Look up target angle from the crease pattern.
            other_xyz = np.array(coarse.nodes[best_other])
            v_key    = _rp(coarse.nodes[v])
            other_key = _rp(coarse.nodes[best_other])
            angle_info = (seg_angle.get((v_key, other_key))
                          or seg_angle.get((other_key, v_key))
                          or _seg_fallback(v_xyz, other_xyz, seg_list))
            if angle_info is None:
                angle_rad, start_t, end_t = 0.0, 0.0, actuator_ramp_time
                if jtype == 126:
                    _warn_no_angle(v, coarse.nodes[v])
            else:
                angle_rad, start_t, end_t = angle_info

            joints.append(JointInfo(
                eid=next_eid,
                node_a=node_a,
                node_b=node_b,
                jtype=jtype,
                axis=axis if jtype == 126 else None,
                target_angle=angle_rad,
                start_time=start_t,
                end_time=end_t,
                penalty=penalty_stiffness,
            ))
            next_eid += 1

    n_rev = sum(1 for j in joints if j.jtype == 126)
    n_sph = sum(1 for j in joints if j.jtype == 120)
    print(f"  Surrogate : {len(new_nodes)} nodes (+{len(new_nodes)-len(coarse.nodes)} dups), "
          f"{len(new_elements)} shell elements, "
          f"{n_rev} revolute joints, {n_sph} spherical joints")

    return Surrogate(
        nodes=new_nodes,
        elements=new_elements,
        joints=joints,
        panel_map=dict(coarse.panel_map),
        penalty_stiffness=penalty_stiffness,
        actuator_ramp_time=actuator_ramp_time,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _rp(pt) -> tuple:
    return (round(pt[0], _COORD_DIGITS),
            round(pt[1], _COORD_DIGITS),
            round(pt[2], _COORD_DIGITS))


def _boundary_nodes(coarse: "Mesh") -> set[int]:
    """Nodes on mesh-perimeter edges (edges belonging to exactly 1 element)."""
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for eid, (etype, nids) in coarse.elements.items():
        if len(nids) < 3:
            continue
        n = len(nids)
        for i in range(n):
            edge_count[_edge(nids[i], nids[(i + 1) % n])] += 1
    bnd: set[int] = set()
    for (va, vb), cnt in edge_count.items():
        if cnt == 1:
            bnd.add(va)
            bnd.add(vb)
    return bnd


def _seg_fallback(
    v_xyz:    "np.ndarray",
    other_xyz: "np.ndarray",
    seg_list: list,
    snap_tol: float = 1e-3,
) -> tuple | None:
    """
    Fallback angle lookup for T-junction split nodes.

    When a node lies on the interior of a through-crease (not at a CSV
    endpoint), the exact key lookup fails because `other` is a split point.
    This searches every crease segment for one that:
      1. Passes through v (within snap_tol)
      2. Has direction approximately aligned with (other - v)
    """
    d = other_xyz - v_xyz
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-12:
        return None
    d_norm = d / d_len

    for p1, p2, angle, start_t, end_t in seg_list:
        seg = p2 - p1
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-12:
            continue
        seg_dir = seg / seg_len
        if abs(float(np.dot(seg_dir, d_norm))) < 0.9:
            continue
        # Check that v lies on this segment
        t = float(np.dot(v_xyz - p1, seg)) / (seg_len ** 2)
        if t < -snap_tol / seg_len or t > 1.0 + snap_tol / seg_len:
            continue
        residual = float(np.linalg.norm(v_xyz - (p1 + t * seg)))
        if residual < snap_tol:
            return (angle, start_t, end_t)

    return None


_warned_nids: set[int] = set()


def _warn_no_angle(nid: int, coords) -> None:
    if nid not in _warned_nids:
        _warned_nids.add(nid)
        print(f"  WARNING: no crease angle found for joint at node {nid} "
              f"({coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}) — "
              f"defaulting to 0.0 rad")
