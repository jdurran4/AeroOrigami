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
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mesh import Mesh
    from .crease import CreasePattern

# rounding decimals for coordinate key matching
_COORD_DIGITS = 4

# When split_quads=True, skip splitting any quad whose best diagonal produces a
# triangle with min angle below this (keep as type-1515 quad instead).
_MIN_SPLIT_ANGLE_DEG = 5.0


def _tri_min_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Minimum interior angle (radians) of a triangle given 3-D vertex arrays."""
    s = np.array([np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)])
    if s.min() < 1e-12:
        return 0.0
    cos_a = (s[0]**2 + s[2]**2 - s[1]**2) / (2 * s[0] * s[2])
    cos_b = (s[0]**2 + s[1]**2 - s[2]**2) / (2 * s[0] * s[1])
    ang_a = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    ang_b = float(np.arccos(np.clip(cos_b, -1.0, 1.0)))
    return min(ang_a, ang_b, np.pi - ang_a - ang_b)


@dataclass
class JointInfo:
    """One joint element connecting two copies of a duplicated crease node."""
    eid:          int
    node_a:       int           # color-0 side (original node)
    node_b:       int           # color-1 side (duplicate node)
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
    panel_colors:      dict[int, int]   # panel_id → 0 or 1 (BFS 2-coloring)
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
    split_quads:        bool = True,
    fold_fraction:      float = 1.0,
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
    split_quads        : if True (default), split every 4-node quad element into
                         two triangles (fan from vertex 0) before joint
                         construction.  Both triangles share the same panel_id so
                         no joint is placed between them and the panel stays rigid.
                         Set to False to keep type-1515 quad shell elements.
    fold_fraction      : scale factor applied to every target fold angle from the
                         CSV (default 1.0 = full fold).  Set to e.g. 0.9 for 90 %
                         of the prescribed angles.

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

    # ── 4. Build panel adjacency graph + BFS 2-color ─────────────────────────
    # Panels are adjacent when they share a crease edge.
    panel_adj: dict[int, set[int]] = defaultdict(set)
    for ps in crease_edges.values():
        pid_list = sorted(ps)
        if len(pid_list) == 2:
            panel_adj[pid_list[0]].add(pid_list[1])
            panel_adj[pid_list[1]].add(pid_list[0])

    all_pids = sorted(set(coarse.panel_map.values()))
    panel_colors = _bfs_color_panels(all_pids, panel_adj)

    # ── 5. Duplicate crease nodes ─────────────────────────────────────────────
    # One duplicate per (vertex, non-owner panel).  Owner = min panel ID at v.
    # This gives each panel its own unique node at every crease vertex, so
    # all N crease edges at a degree-N vertex produce N distinct node pairs
    # and N independent joint elements.
    # 2-coloring is used only for axis direction and node_a/node_b ordering.
    # dup_map[(orig_nid, panel_id)] = node_id to use for that panel.
    dup_map: dict[tuple[int, int], int] = {}
    new_nodes: dict[int, tuple[float, float, float]] = dict(coarse.nodes)
    next_nid = max(coarse.nodes) + 1

    for v in sorted(crease_nids):
        owner = min(n2p[v])
        for pid in n2p[v]:
            if pid == owner:
                dup_map[(v, pid)] = v
            else:
                dup_map[(v, pid)] = next_nid
                new_nodes[next_nid] = coarse.nodes[v]
                next_nid += 1

    # ── 5b. Update element connectivity ──────────────────────────────────────
    new_elements: dict[int, tuple[int, list[int]]] = {}
    for eid, (etype, nids) in coarse.elements.items():
        pid = coarse.panel_map.get(eid)
        if pid is None:
            new_elements[eid] = (etype, list(nids))
            continue
        new_nids = [dup_map.get((nid, pid), nid) for nid in nids]
        new_elements[eid] = (etype, new_nids)

    # ── 5b. Split quad elements into triangles ───────────────────────────────
    n_quad_splits  = 0
    n_quad_kept    = 0
    _min_split_rad = np.radians(_MIN_SPLIT_ANGLE_DEG)
    if split_quads:
        split_elems: dict[int, tuple[int, list[int]]] = {}
        split_pmap:  dict[int, int] = {}
        _next_split_eid = max(new_elements) + 1
        for eid, (etype, nids) in new_elements.items():
            pid = coarse.panel_map.get(eid)
            if len(nids) == 4:
                n0, n1, n2, n3 = nids
                p0 = np.array(new_nodes[n0], dtype=float)
                p1 = np.array(new_nodes[n1], dtype=float)
                p2 = np.array(new_nodes[n2], dtype=float)
                p3 = np.array(new_nodes[n3], dtype=float)
                # Both candidate splits; pick the one with the better min angle.
                min02 = min(_tri_min_angle(p0, p1, p2), _tri_min_angle(p0, p2, p3))
                min13 = min(_tri_min_angle(p0, p1, p3), _tri_min_angle(p1, p2, p3))
                best  = max(min02, min13)
                if best < _min_split_rad:
                    # Both splits produce a degenerate triangle — keep as quad.
                    split_elems[eid] = (etype, list(nids))
                    if pid is not None:
                        split_pmap[eid] = pid
                    n_quad_kept += 1
                elif min13 > min02:
                    split_elems[eid]             = (etype, [n0, n1, n3])
                    split_elems[_next_split_eid] = (etype, [n1, n2, n3])
                    if pid is not None:
                        split_pmap[eid]             = pid
                        split_pmap[_next_split_eid] = pid
                    _next_split_eid += 1
                    n_quad_splits += 1
                else:
                    split_elems[eid]             = (etype, [n0, n1, n2])
                    split_elems[_next_split_eid] = (etype, [n0, n2, n3])
                    if pid is not None:
                        split_pmap[eid]             = pid
                        split_pmap[_next_split_eid] = pid
                    _next_split_eid += 1
                    n_quad_splits += 1
            else:
                split_elems[eid] = (etype, nids)
                if pid is not None:
                    split_pmap[eid] = pid
        new_elements = split_elems
        panel_map_out = split_pmap
    else:
        panel_map_out = {eid: pid for eid, pid in coarse.panel_map.items()
                         if eid in new_elements}

    # ── 6. Panel normals + centroids for consistent axis computation ─────────
    panel_normals, panel_centroids = _compute_panel_normals_centroids(
        coarse.elements, coarse.nodes, coarse.panel_map, all_pids,
        getattr(coarse, "panel_outward_hints", {}),
    )

    # ── 7. Crease-segment angle lookup ───────────────────────────────────────
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

    # ── 8. Build joint elements ───────────────────────────────────────────────
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

            # Identify color-0 (original) and color-1 (duplicate) sides.
            PA_color = panel_colors.get(PA, 0)
            pid0 = PA if PA_color == 0 else PB   # color-0 panel
            pid1 = PB if PA_color == 0 else PA   # color-1 panel

            node_a = dup_map.get((v, pid0), v)   # original node (color-0)
            node_b = dup_map.get((v, pid1), v)   # duplicate node (color-1)

            # Skip if this node pair already got a joint (collinear segments).
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

            # Crease axis — consistent convention via 2-coloring:
            #   axis points along the crease such that the color-0 panel is on
            #   the RIGHT when viewed from the outward surface normal.
            #
            #   x_raw = cross(d_01, n_avg)  where d_01 = centroid_1 - centroid_0
            #   axis  = crease_tangent aligned with x_raw
            best_other = max(
                others,
                key=lambda w: np.linalg.norm(np.array(coarse.nodes[w]) - v_xyz),
            )
            other_xyz = np.array(coarse.nodes[best_other])
            tang = other_xyz - v_xyz
            tang_len = float(np.linalg.norm(tang))
            tang = tang / tang_len if tang_len > 1e-12 else np.array([1., 0., 0.])

            cA = panel_centroids.get(pid0, np.zeros(3))
            cB = panel_centroids.get(pid1, np.zeros(3))
            nA = panel_normals.get(pid0, np.array([0., 0., 1.]))
            nB = panel_normals.get(pid1, np.array([0., 0., 1.]))

            d01 = cB - cA
            d01_len = float(np.linalg.norm(d01))
            if d01_len > 1e-12:
                d01 = d01 / d01_len
                n_avg = nA + nB
                n_avg_len = float(np.linalg.norm(n_avg))
                if n_avg_len > 1e-12:
                    n_avg = n_avg / n_avg_len
                    x_raw = np.cross(d01, n_avg)
                    x_raw_len = float(np.linalg.norm(x_raw))
                    if x_raw_len > 1e-12:
                        x_raw = x_raw / x_raw_len
                        axis_arr = tang if float(np.dot(tang, x_raw)) > 0 else -tang
                    else:
                        axis_arr = tang
                else:
                    axis_arr = tang
            else:
                axis_arr = tang

            axis = tuple(float(x) for x in axis_arr)

            # Look up target angle from the crease pattern.
            v_key     = _rp(coarse.nodes[v])
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
                angle_rad *= fold_fraction

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
    split_parts = []
    if n_quad_splits: split_parts.append(f"{n_quad_splits} quads → 2 tris")
    if n_quad_kept:   split_parts.append(f"{n_quad_kept} quads kept (thin split avoided)")
    split_note = f" ({', '.join(split_parts)})" if split_parts else ""
    print(f"  Surrogate : {len(new_nodes)} nodes (+{len(new_nodes)-len(coarse.nodes)} dups), "
          f"{len(new_elements)} shell elements{split_note}, "
          f"{n_rev} revolute joints, {n_sph} spherical joints")

    return Surrogate(
        nodes=new_nodes,
        elements=new_elements,
        joints=joints,
        panel_map=panel_map_out,
        panel_colors=panel_colors,
        penalty_stiffness=penalty_stiffness,
        actuator_ramp_time=actuator_ramp_time,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bfs_color_panels(
    panel_ids: list[int],
    panel_adj: dict[int, set[int]],
) -> dict[int, int]:
    """BFS 2-color the panel adjacency graph. Origami panels are always bipartite."""
    colors: dict[int, int] = {}
    for start in panel_ids:
        if start in colors:
            continue
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        while queue:
            pid, color = queue.popleft()
            if pid in colors:
                if colors[pid] != color:
                    raise ValueError(
                        f"Panel graph is not bipartite at panel {pid} — "
                        "crease pattern may be invalid"
                    )
                continue
            colors[pid] = color
            for nbr in panel_adj.get(pid, set()):
                if nbr not in colors:
                    queue.append((nbr, 1 - color))
    return colors


def _compute_panel_normals_centroids(
    coarse_elems: dict,
    coarse_nodes: dict,
    panel_map: dict,
    panel_ids: list[int],
    outward_hints: dict,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Area-weighted panel normals + centroids oriented outward.

    If outward_hints[pid] is set, that vector determines the outward side.
    Otherwise orientation is determined by the sign of dot(n, centroid - center)
    where center is the centroid of all un-hinted panel centroids.
    """
    panel_elems: dict[int, list] = defaultdict(list)
    for eid, pid in panel_map.items():
        if eid in coarse_elems:
            panel_elems[pid].append(eid)

    normals:   dict[int, np.ndarray] = {}
    centroids: dict[int, np.ndarray] = {}

    for pid in panel_ids:
        n_sum = np.zeros(3)
        c_sum = np.zeros(3)
        a_sum = 0.0
        for eid in panel_elems.get(pid, []):
            nids = coarse_elems[eid][1]
            if len(nids) < 3:
                continue
            pts = [np.asarray(coarse_nodes[n], float) for n in nids[:3]]
            tri_n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            tri_a = 0.5 * float(np.linalg.norm(tri_n))
            n_sum += tri_n
            c_sum += ((pts[0] + pts[1] + pts[2]) / 3.0) * tri_a
            a_sum += tri_a

        nlen = float(np.linalg.norm(n_sum))
        normals[pid]   = n_sum / nlen if nlen > 1e-12 else np.array([0., 0., 1.])
        centroids[pid] = c_sum / a_sum if a_sum > 1e-12 else np.zeros(3)

    # Orient outward: hinted panels first, then auto-detect remainder
    hinted   = {pid for pid in panel_ids if pid in outward_hints}
    unhinted = [pid for pid in panel_ids if pid not in outward_hints]

    for pid in hinted:
        hint = np.asarray(outward_hints[pid], float)
        if float(np.dot(normals[pid], hint)) < 0:
            normals[pid] = -normals[pid]

    if unhinted:
        center = np.mean([centroids[pid] for pid in unhinted], axis=0)
        for pid in unhinted:
            if float(np.dot(normals[pid], centroids[pid] - center)) < 0:
                normals[pid] = -normals[pid]

    return normals, centroids


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
