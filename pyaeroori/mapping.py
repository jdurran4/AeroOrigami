"""
mapping.py — Step 7 of the AeroOrigami pipeline.

Reads AERO-S fold simulation output (gdisplac6.xpost) and maps the
coarse surrogate displacements back onto the original fine mesh, then
writes IDISP6.include for downstream FSI simulations.

Public API
----------
read_xpost(disp_file, step=-1)
    Parse gdisplac6.xpost displacement file.

map_displacements(surrogate, fine_mesh, disp_file, config, step, ...)
    Membrane interpolation + cable arc-length reconstruction.
    method="rbf"         — multiquadric RBF (smooth, default)
    method="panel_rigid" — per-panel Procrustes rigid-body (best near
                           vent/high-curvature regions with large angles)

write_idisp6(displacements, fine_mesh, output_path, amp=1.0)
    Write AERO-S IDISP6.include.

write_folded_vtk(fine_mesh, displacements, output_path)
    Write VTK for ParaView visualization of the deformed fine mesh.

Notes on kernel choice for RBF method
--------------------------------------
multiquadric (default): global, smooth — works well for moderate folds.
    Increasing rbf_neighbors (e.g. 200-500) helps more than decreasing it;
    fewer neighbors makes the field jagged.

thin_plate_spline: requires degree >= 2 in 3D (scipy RBFInterpolator), otherwise
    the polynomial basis is under-determined → singular matrix.  Use:
        RBFInterpolator(..., kernel="thin_plate_spline", degree=2)

For large-angle folds near the vent where many panels converge, the RBF
approach struggles because the displacement field is discontinuous across
fold lines.  Use method="panel_rigid" in those cases.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# xpost reader
# ─────────────────────────────────────────────────────────────────────────────

def read_xpost(
    disp_file: str | Path,
    step: int = -1,
    node_ids: list[int] | None = None,
) -> dict[int, tuple]:
    """
    Parse an AERO-S gdisplac6.xpost displacement file.

    AERO-S writes two formats depending on solver version:
        7-column:  node_id  dx  dy  dz  rx  ry  rz   (node ID explicit)
        6-column:  dx  dy  dz  rx  ry  rz             (positional, no ID)

    The format is detected automatically from the first data row.

    Parameters
    ----------
    disp_file : path to xpost file
    step      : which time step to read; -1 = last (default)
    node_ids  : ordered list of node IDs to use when the file is in 6-column
                (positional) format.  The i-th displacement row is assigned to
                node_ids[i].  Pass the coarse node list from map_displacements
                so the ordering matches the ORIGAMI_MESH.include output.
                If None and the file is 6-column, sequential 1-based IDs are
                used as a fallback (only correct when nodes are numbered 1..N).

    Returns
    -------
    dict[int, tuple[float,float,float,float,float,float]]
        {node_id: (dx, dy, dz, rx, ry, rz)}
    """
    with open(disp_file, "r") as f:
        lines = f.readlines()

    blocks: list[tuple[int, int]] = []   # (data_start_line, n_nodes)
    for i, line in enumerate(lines):
        if line.strip().startswith("Vector DISP"):
            n_nodes = int(lines[i + 1].strip())
            blocks.append((i + 2, n_nodes))

    if not blocks:
        raise RuntimeError(f"No 'Vector DISP' block found in {disp_file}")

    start_idx, n_nodes = blocks[step]

    # Detect format from the first non-empty data row
    fmt = None
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) == 7:
            fmt = "id+6"
            break
        if len(parts) == 6:
            fmt = "6only"
            break

    if fmt is None:
        raise RuntimeError(f"Could not detect xpost row format in {disp_file}")

    if fmt == "6only":
        if node_ids is None:
            print(f"  WARNING: xpost is 6-column (no node IDs); using sequential "
                  f"1-based IDs. Pass node_ids= for correct mapping.")
        pos = 0

    result: dict[int, tuple] = {}
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if fmt == "id+6" and len(parts) == 7:
            nid = int(parts[0])
            result[nid] = tuple(float(v) for v in parts[1:])
        elif fmt == "6only" and len(parts) == 6:
            if node_ids is not None:
                nid = node_ids[pos]
            else:
                nid = pos + 1
            result[nid] = tuple(float(v) for v in parts)
            pos += 1
        if len(result) == n_nodes:
            break

    if len(result) != n_nodes:
        raise RuntimeError(
            f"Expected {n_nodes} displacement rows, got {len(result)} in {disp_file}"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main mapping function
# ─────────────────────────────────────────────────────────────────────────────

def map_displacements(
    surrogate,
    fine_mesh,
    disp_file: str | Path,
    config=None,
    step: int = -1,
    method: str = "rbf",
    rbf_neighbors: int = 100,
    rbf_smoothing: float = 1e-7,
) -> dict[int, tuple]:
    """
    Map fold displacements from the coarse surrogate onto the original fine mesh.

    Cable intermediate nodes are always reconstructed by topology-driven BFS
    arc-length interpolation regardless of method.

    Parameters
    ----------
    surrogate     : Surrogate (output of build_surrogate)
    fine_mesh     : Mesh     (output of load_mesh on the original fine .fem)
    disp_file     : path to gdisplac6.xpost (or .xpost.N) from the fold sim
    config        : ModelConfig (optional) — adds cable endpoint nodes and their
                    DISP BCs as additional source / anchor points
    step          : time step index; -1 = last
    method        : "rbf" (default) or "panel_rigid"
                    "rbf"         — multiquadric RBF from all coarse nodes;
                                    smooth but can average across fold lines.
                    "panel_rigid" — per-panel Procrustes rigid-body transform;
                                    correct for large-angle folds, no cross-panel
                                    averaging.  Best near vent / high-curvature
                                    regions.  Requires surrogate.panel_map to be
                                    populated (always true after build_surrogate).
    rbf_neighbors : nearest-neighbour count for RBF (ignored for panel_rigid)
    rbf_smoothing : RBF smoothing factor; 0 = exact interpolation

    Returns
    -------
    dict[int, tuple[float,float,float,float,float,float]]
        {node_id: (dx, dy, dz, rx, ry, rz)} for every node in fine_mesh.nodes.
        Cable intermediate nodes not reachable via BFS are set to zero.
    """
    # ── 1. Build coarse source: surrogate nodes + any extra cable endpoint nodes
    coarse_nodes: dict[int, tuple] = dict(surrogate.nodes)
    if config is not None and config.cable_nodes:
        coarse_nodes.update(config.cable_nodes)

    # Ordered list used to assign node IDs when xpost has no node column (6-col fmt)
    coarse_ids = list(coarse_nodes.keys())

    # ── 2. Read xpost displacements (keyed by coarse node ID) ────────────────
    raw_disp = read_xpost(disp_file, step=step, node_ids=coarse_ids)

    missing = [nid for nid in coarse_nodes if nid not in raw_disp]
    if missing:
        print(f"  WARNING: {len(missing)} coarse node(s) not found in xpost "
              f"(first few: {missing[:5]}). Their displacement will be zero.")

    # ── 3. Interpolate membrane (canopy) nodes ────────────────────────────────
    fine_membrane_ids    = sorted(fine_mesh.membrane_nodes)
    fine_cable_only_ids  = sorted(fine_mesh.cable_nodes)

    print(f"  Mapping [{method}]: {len(coarse_nodes)} coarse nodes → "
          f"{len(fine_membrane_ids)} fine membrane nodes, "
          f"{len(fine_cable_only_ids)} cable nodes via arc-length")

    if method == "panel_rigid":
        resolved = _panel_rigid_body_map(surrogate, fine_mesh, raw_disp,
                                         fine_membrane_ids)
    elif method == "rbf":
        resolved = _rbf_map(coarse_nodes, raw_disp, fine_mesh,
                            fine_membrane_ids, rbf_neighbors, rbf_smoothing)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'rbf' or 'panel_rigid'.")

    # ── 4. Seed any DISP BC nodes with zero displacement ─────────────────────
    if config is not None and config.disp_bcs:
        for nid, _ in config.disp_bcs:
            if nid not in resolved:
                resolved[nid] = (0., 0., 0., 0., 0., 0.)

    # ── 4.5. Seed cable anchor nodes via nearest coarse node ─────────────────
    # Cable chain endpoints (degree 1) and junctions (degree ≥ 3) that aren't
    # membrane nodes won't be touched by the membrane interpolation above.
    # BFS needs both ends of each chain resolved before it can fill interiors,
    # so seed any still-unresolved anchors from the nearest coarse surrogate
    # node.  This covers suspension-line top attachments (not shared with the
    # canopy shell mesh) and bottom bridle/payload attachment nodes.
    if fine_cable_only_ids:
        _seed_cable_anchors(fine_mesh, resolved, coarse_ids, coarse_nodes, raw_disp)

    # ── 5. Cable arc-length reconstruction via BFS from resolved boundary ─────
    if fine_cable_only_ids:
        _reconstruct_cables(fine_mesh, resolved)

    # ── 6. Pack result ────────────────────────────────────────────────────────
    zero6 = (0., 0., 0., 0., 0., 0.)
    result = {nid: resolved.get(nid, zero6) for nid in fine_mesh.nodes}

    n_zero = sum(1 for nid in fine_cable_only_ids if result[nid] == zero6)
    if n_zero:
        print(f"  WARNING: {n_zero} cable node(s) could not be resolved "
              "(not reachable from any membrane anchor). Set to zero.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Method A: RBF
# ─────────────────────────────────────────────────────────────────────────────

def _rbf_map(
    coarse_nodes: dict,
    raw_disp: dict,
    fine_mesh,
    fine_membrane_ids: list[int],
    rbf_neighbors: int,
    rbf_smoothing: float,
) -> dict[int, tuple]:
    from scipy.interpolate import RBFInterpolator

    coarse_ids    = list(coarse_nodes.keys())
    coarse_coords = np.array([coarse_nodes[nid] for nid in coarse_ids], dtype=float)
    coarse_disp6  = np.array(
        [raw_disp.get(nid, (0., 0., 0., 0., 0., 0.)) for nid in coarse_ids],
        dtype=float,
    )

    fine_coords = np.array(
        [fine_mesh.nodes[nid] for nid in fine_membrane_ids], dtype=float
    )

    rbf_trans = RBFInterpolator(
        coarse_coords, coarse_disp6[:, :3],
        kernel="multiquadric", epsilon=1.0, degree=0,
        neighbors=rbf_neighbors, smoothing=rbf_smoothing,
    )
    rbf_rots = RBFInterpolator(
        coarse_coords, coarse_disp6[:, 3:],
        kernel="multiquadric", epsilon=1.0, degree=0,
        neighbors=rbf_neighbors, smoothing=rbf_smoothing,
    )

    fine_trans = rbf_trans(fine_coords)
    fine_rots  = rbf_rots(fine_coords)

    return {
        nid: (float(t[0]), float(t[1]), float(t[2]),
              float(r[0]), float(r[1]), float(r[2]))
        for nid, t, r in zip(fine_membrane_ids, fine_trans, fine_rots)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method B: per-panel Procrustes rigid-body
# ─────────────────────────────────────────────────────────────────────────────

def _panel_rigid_body_map(
    surrogate,
    fine_mesh,
    raw_disp: dict,
    fine_membrane_ids: list[int],
) -> dict[int, tuple]:
    """
    For each surrogate panel, find the best-fit rigid-body rotation R and
    translation t (Procrustes / Kabsch algorithm) mapping undeformed → deformed
    panel node positions.  Each fine membrane node is assigned to its nearest
    panel centroid (in undeformed space) and displaced by that panel's R and t.

    Rotation DOFs (rx, ry, rz) are the rotation-vector components of R,
    consistent with AERO-S large-rotation shell output.

    This avoids cross-panel averaging that makes RBF inaccurate near the vent
    where many panels converge at large fold angles.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    # ── Build panel → node set from surrogate topology ───────────────────────
    # surrogate.elements already has the correct node IDs for each panel's
    # 2-coloring (original nodes for color-0, duplicates for color-1), so
    # each panel's Procrustes sees exactly its own side of every crease.
    panel_to_nodes: dict[int, set[int]] = defaultdict(set)
    for eid, (_, conn) in surrogate.elements.items():
        if len(conn) >= 3:
            pid = surrogate.panel_map.get(eid)
            if pid is not None:
                panel_to_nodes[pid].update(conn)

    # ── Procrustes per panel ──────────────────────────────────────────────────
    panel_R:   dict[int, np.ndarray] = {}
    panel_t:   dict[int, np.ndarray] = {}
    panel_rv:  dict[int, np.ndarray] = {}   # rotation vector for DOFs 4-6
    panel_cen: dict[int, np.ndarray] = {}   # undeformed centroid for KD-tree

    skipped = 0
    for pid, node_set in panel_to_nodes.items():
        # Keep only nodes that appear in both the surrogate and the xpost
        nids = [n for n in node_set if n in surrogate.nodes and n in raw_disp]
        if len(nids) < 3:
            skipped += 1
            continue

        P = np.array([surrogate.nodes[n] for n in nids], dtype=float)   # undeformed
        Q = np.array(
            [(surrogate.nodes[n][0] + raw_disp[n][0],
              surrogate.nodes[n][1] + raw_disp[n][1],
              surrogate.nodes[n][2] + raw_disp[n][2])
             for n in nids],
            dtype=float,
        )  # deformed

        c_p = P.mean(axis=0)
        c_q = Q.mean(axis=0)

        # Kabsch algorithm: SVD of the cross-covariance matrix
        H = (P - c_p).T @ (Q - c_q)
        U, _, Vt = np.linalg.svd(H)

        # Enforce proper rotation (det = +1, not a reflection)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        panel_R[pid]   = R
        panel_t[pid]   = c_q - R @ c_p
        panel_cen[pid] = c_p
        panel_rv[pid]  = Rotation.from_matrix(R).as_rotvec()

    if skipped:
        print(f"  panel_rigid: {skipped} panel(s) skipped (< 3 nodes with xpost data)")
    print(f"  panel_rigid: {len(panel_R)} panels with Procrustes transforms")

    if not panel_R:
        raise RuntimeError(
            "panel_rigid: no panels could be solved. "
            "Ensure surrogate.panel_map is populated and xpost node IDs match."
        )

    # ── KD-tree on undeformed panel centroids → nearest-panel assignment ──────
    pid_list  = list(panel_R.keys())
    centroids = np.array([panel_cen[p] for p in pid_list], dtype=float)
    tree      = cKDTree(centroids)

    fine_coords = np.array(
        [fine_mesh.nodes[nid] for nid in fine_membrane_ids], dtype=float
    )
    _, idx = tree.query(fine_coords)

    # ── Apply panel transform to each fine node ───────────────────────────────
    resolved: dict[int, tuple] = {}
    for i, nid in enumerate(fine_membrane_ids):
        pid = pid_list[idx[i]]
        R   = panel_R[pid]
        t   = panel_t[pid]
        rv  = panel_rv[pid]
        x   = fine_coords[i]
        d   = R @ x + t - x          # displacement = deformed_pos - original_pos
        resolved[nid] = (
            float(d[0]),  float(d[1]),  float(d[2]),
            float(rv[0]), float(rv[1]), float(rv[2]),
        )

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Cable anchor seeding (shared by both methods)
# ─────────────────────────────────────────────────────────────────────────────

def _seed_cable_anchors(
    fine_mesh,
    resolved: dict[int, tuple],
    coarse_ids: list[int],
    coarse_nodes: dict[int, tuple],
    raw_disp: dict[int, tuple],
) -> None:
    """
    Seed cable chain endpoints and junction nodes that aren't already resolved.

    Uses a KD-tree nearest-coarse-node lookup so that:
    - Suspension-line top nodes (not shared with the canopy shell mesh) get
      the displacement of the nearest surrogate node on the canopy rim.
    - Bottom bridle/payload attachment nodes get the displacement of the
      nearest surrogate DISP BC node (~zero).
    - Junction nodes get the displacement of their nearest surrogate neighbour.

    After seeding, BFS arc-length fill in _reconstruct_cables has an anchor
    at both ends of every chain.
    """
    from scipy.spatial import cKDTree

    adj    = _build_cable_graph(fine_mesh)
    degree = {n: len(nbrs) for n, nbrs in adj.items()}
    anchors = [n for n, d in degree.items()
               if (d == 1 or d >= 3) and n not in resolved]

    if not anchors:
        return

    coarse_coords = np.array([coarse_nodes[n] for n in coarse_ids], dtype=float)
    coarse_disp6  = np.array(
        [raw_disp.get(n, (0., 0., 0., 0., 0., 0.)) for n in coarse_ids], dtype=float
    )
    anchor_coords = np.array([fine_mesh.nodes[n] for n in anchors], dtype=float)

    tree = cKDTree(coarse_coords)
    _, idx = tree.query(anchor_coords)

    for i, nid in enumerate(anchors):
        resolved[nid] = tuple(float(v) for v in coarse_disp6[idx[i]])

    print(f"  Seeded {len(anchors)} cable anchor node(s) via nearest coarse node")


# ─────────────────────────────────────────────────────────────────────────────
# Cable arc-length reconstruction (shared by both methods)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cable_graph(fine_mesh) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for _, (_, conn) in fine_mesh.cable_elements.items():
        if len(conn) == 2:
            a, b = conn
            adj[a].add(b)
            adj[b].add(a)
    return dict(adj)


def _extract_chains(adj: dict[int, set[int]]) -> list[list[int]]:
    """
    Extract maximal paths (chains) from a cable adjacency graph.

    Endpoints are nodes of degree 1 (free ends) or degree >= 3 (junctions).
    """
    degree = {n: len(nbrs) for n, nbrs in adj.items()}
    visited_edges: set[frozenset] = set()
    chains: list[list[int]] = []

    def walk(start: int, first_step: int) -> list[int]:
        path = [start, first_step]
        visited_edges.add(frozenset((start, first_step)))
        prev, cur = start, first_step
        while degree.get(cur, 0) == 2:
            nxt_options = adj[cur] - {prev}
            if not nxt_options:
                break
            nxt = next(iter(nxt_options))
            edge = frozenset((cur, nxt))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for node in sorted(adj.keys()):
        if degree.get(node, 0) in (1,) or degree.get(node, 0) >= 3:
            for nbr in sorted(adj[node]):
                edge = frozenset((node, nbr))
                if edge not in visited_edges:
                    chains.append(walk(node, nbr))

    # Catch unvisited edges (pure cycles)
    for node in sorted(adj.keys()):
        for nbr in sorted(adj[node]):
            edge = frozenset((node, nbr))
            if edge not in visited_edges:
                chains.append(walk(node, nbr))

    return chains


def _arc_length_fill(
    chain: list[int],
    fine_nodes: dict[int, tuple],
    resolved: dict[int, tuple],
    d0: tuple,
    d1: tuple,
) -> None:
    """
    Fill interior cable nodes by arc-length interpolation.

    s ∈ [0,1] is the normalised arc-length in the UNDEFORMED geometry;
    both translation and rotation are linearly blended.
    """
    coords = [np.array(fine_nodes[n], dtype=float) for n in chain]
    cumlen = [0.0]
    for i in range(1, len(coords)):
        cumlen.append(cumlen[-1] + float(np.linalg.norm(coords[i] - coords[i - 1])))
    total = cumlen[-1]
    if total < 1e-15:
        return

    d0_arr = np.array(d0, dtype=float)
    d1_arr = np.array(d1, dtype=float)

    for nid, s_raw in zip(chain, cumlen):
        if nid in resolved:
            continue
        s     = s_raw / total
        trans = (1.0 - s) * d0_arr[:3] + s * d1_arr[:3]
        rots  = (1.0 - s) * d0_arr[3:] + s * d1_arr[3:]
        resolved[nid] = (
            float(trans[0]), float(trans[1]), float(trans[2]),
            float(rots[0]),  float(rots[1]),  float(rots[2]),
        )


def _reconstruct_cables(fine_mesh, resolved: dict[int, tuple]) -> None:
    """
    Fill all cable-only nodes via BFS arc-length reconstruction.

    Iterates until stable:
    1. Any chain with both endpoints resolved → arc-length fill interior.
    2. Any junction node (degree >= 3) with all resolved cable neighbours
       → resolve as displacement mean of those neighbours.
    """
    adj    = _build_cable_graph(fine_mesh)
    if not adj:
        return

    chains = _extract_chains(adj)
    degree = {n: len(nbrs) for n, nbrs in adj.items()}
    pending = list(chains)

    for _ in range(len(chains) + 1):
        if not pending:
            break

        still_pending = []
        made_progress = False

        for chain in pending:
            end0, end1 = chain[0], chain[-1]
            if end0 in resolved and end1 in resolved:
                _arc_length_fill(
                    chain, fine_mesh.nodes, resolved,
                    resolved[end0], resolved[end1],
                )
                made_progress = True
            else:
                still_pending.append(chain)

        # Resolve junction nodes from their already-resolved cable neighbours
        for node in sorted(adj.keys()):
            if node in resolved or degree.get(node, 0) < 3:
                continue
            resolved_nbrs = [n for n in adj[node] if n in resolved]
            if not resolved_nbrs:
                continue
            mean_d = np.mean(
                [np.array(resolved[n], dtype=float) for n in resolved_nbrs], axis=0
            )
            resolved[node] = tuple(float(v) for v in mean_d)
            made_progress = True

        pending = still_pending
        if not made_progress:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_idisp6(
    displacements: dict[int, tuple],
    fine_mesh,
    output_path: str | Path,
    amp: float = 1.0,
) -> None:
    """
    Write AERO-S IDISP6.include with 6-DOF initial displacements.

    Format (per node):
        {nid:8d}{dx:16.8e}{dy:16.8e}{dz:16.8e}{rx:16.8e}{ry:16.8e}{rz:16.8e}

    Iterates over fine_mesh.nodes to guarantee consistent ordering.
    Nodes absent from displacements are written as zeros.
    """
    zero6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("IDISP6\n")
        for nid in fine_mesh.nodes:
            dx, dy, dz, rx, ry, rz = displacements.get(nid, zero6)
            f.write(
                f"{nid:8d}"
                f"{dx * amp:16.8e}{dy * amp:16.8e}{dz * amp:16.8e}"
                f"{rx * amp:16.8e}{ry * amp:16.8e}{rz * amp:16.8e}\n"
            )
        f.write("*\n")

    print(f"  Wrote {len(fine_mesh.nodes)} nodes to {output_path}")


def write_folded_vtk(
    fine_mesh,
    displacements: dict[int, tuple],
    output_path: str | Path,
) -> None:
    """
    Write a VTK unstructured grid of the deformed fine mesh for ParaView.

    Node positions are shifted by the translation component of displacements.
    The displacement vector is stored as a VECTORS point-data field.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zero6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nid_list   = list(fine_mesh.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}

    coords = []
    for nid in nid_list:
        x, y, z = fine_mesh.nodes[nid]
        dx, dy, dz, *_ = displacements.get(nid, zero6)
        coords.append((x + dx, y + dy, z + dz))

    valid_elems = [
        conn for _, (_, conn) in fine_mesh.elements.items()
        if all(n in nid_to_idx for n in conn) and len(conn) in (2, 3, 4)
    ]

    vtk_type   = {2: 3, 3: 5, 4: 9}
    total_size = sum(len(e) + 1 for e in valid_elems)

    with open(output_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("AeroOrigami folded fine mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"\nPOINTS {len(coords)} float\n")
        for x, y, z in coords:
            f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"\nCELLS {len(valid_elems)} {total_size}\n")
        for conn in valid_elems:
            idxs = [nid_to_idx[n] for n in conn]
            f.write(f"{len(conn)} " + " ".join(map(str, idxs)) + "\n")

        f.write(f"\nCELL_TYPES {len(valid_elems)}\n")
        for conn in valid_elems:
            f.write(f"{vtk_type[len(conn)]}\n")

        f.write(f"\nPOINT_DATA {len(nid_list)}\n")
        f.write("VECTORS displacement float\n")
        for nid in nid_list:
            dx, dy, dz, *_ = displacements.get(nid, zero6)
            f.write(f"{dx:.8e} {dy:.8e} {dz:.8e}\n")

    print(f"  Wrote {len(valid_elems)} elements to {output_path}")
