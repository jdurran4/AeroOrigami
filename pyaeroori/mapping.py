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
    Membrane interpolation + cable arc-length reconstruction, reading coarse
    node displacements from an AERO-S xpost file. Thin wrapper around
    map_displacements_from_coarse().
    method="rbf"         — multiquadric RBF (smooth, default)
    method="panel_rigid" — per-panel Procrustes rigid-body, single-panel
                           assignment per fine node (no blending at creases —
                           see mapping.py module notes below). Best for
                           large-angle / tightly-folded regions where RBF's
                           smoothing would self-penetrate.

map_displacements_from_coarse(coarse_nodes, raw_disp, fine_mesh, ...)
    Same mapping, but takes already-known coarse node coordinates +
    displacements directly (no xpost file / surrogate needed) — for datasets
    where the fold displacement was solved outside AERO-S (e.g. a kinematic
    rigid-origami model) and Steps 1-6 are skipped entirely.

assign_fine_nodes_to_panels(panel_source, fine_mesh, ...)
    Topology-based (point-in-triangle) fine-node → panel assignment used by
    method="panel_rigid". Exposed publicly so callers can reuse the same
    assignment for check_panel_boundary_strain / check_inverted_elements
    without recomputing it differently.

check_panel_boundary_strain(fine_mesh, displacements, node_to_panel, tol=0.05)
    General diagnostic (works with displacements from ANY method): strain on
    every fine-mesh edge whose endpoints land in different panels.

check_inverted_elements(fine_mesh, displacements, node_to_panel)
    General diagnostic (works with displacements from ANY method): flags
    multi-panel elements (panel junctions, or fine elements straddling a
    fold line the fine mesh doesn't track) whose normal flips after folding.

write_idisp6(displacements, fine_mesh, output_path, amp=1.0)
    Write AERO-S IDISP6.include.

write_folded_vtk(fine_mesh, displacements, output_path)
    Write VTK for ParaView visualization of the deformed fine mesh.

write_wireframe_vtk(nodes, displacements, edges, output_path)
    Write a VTK wireframe (line segments) for a node+edge set with no shell-
    element connectivity — e.g. a coarse crease pattern's own edges.csv
    topology — for comparing a pre-interpolation ground truth against
    write_folded_vtk()'s fine-mesh surface in the same viewer.

Notes on kernel choice for RBF method
--------------------------------------
multiquadric (default): global, smooth — works well for moderate folds.
    Increasing rbf_neighbors (e.g. 200-500) helps more than decreasing it;
    fewer neighbors makes the field jagged.

    rbf_smoothing matters more than rbf_neighbors for suppressing large,
    "makes no sense" spikes at individual query points: rbf_smoothing≈0
    forces RBFInterpolator to fit every coarse point exactly, which is a
    near-singular linear solve whenever coarse points are unevenly spaced
    or nearly collinear/coincident locally (common for coarse, few-node
    crease-pattern data) — the fit still succeeds but is poorly conditioned,
    so evaluating it at a nearby fine node can massively overshoot. Fewer
    neighbors makes this worse (smaller, more locally-conditioned linear
    system per query point) but is not the root cause — increasing
    rbf_smoothing (e.g. 1e-3 to 1e-2) regularizes the solve directly and is
    the more effective fix. Confirmed empirically on the flasher dataset:
    smoothing=1e-2 removed spikes that persisting at any neighbor count
    with smoothing=1e-7 did not.

thin_plate_spline: requires degree >= 2 in 3D (scipy RBFInterpolator), otherwise
    the polynomial basis is under-determined → singular matrix.  Use:
        RBFInterpolator(..., kernel="thin_plate_spline", degree=2)

For large-angle folds near the vent where many panels converge, the RBF
approach struggles because the displacement field is discontinuous across
fold lines.  method="panel_rigid" (below) removes that particular problem,
but introduces a different one — see its own notes — so it isn't a strict
upgrade; try tuning rbf_smoothing first.

Notes on method="panel_rigid"
------------------------------
Each fine node is assigned to exactly ONE panel (via topology — point-in-panel
containment against that panel's actual triangle(s), not nearest-centroid) and
moved by that panel's exact rigid Procrustes transform. There is no blending
at crease boundaries: this assumes the true geometric gap/overlap between
adjacent panels is smaller than the fine mesh's local edge length, so the
small strain induced at crease-boundary edges is acceptable. Use
check_panel_boundary_strain() / check_inverted_elements() to confirm that
assumption holds rather than assuming it — on the flasher dataset it did NOT
hold everywhere: check_crease_coverage() (pyaeroori/plot.py) showed crease
*endpoints* are well covered by the fine mesh, but that says nothing about
whether the fine mesh's own triangulation tracks the *interior* of each
crease segment — it doesn't, since dgb_mesh.fem was gore-triangulated once,
shared across every crease pattern, independent of any of their fold lines.
Fine elements straddling a real fold line the mesh never tracked get a hard,
undiluted wrong-panel assignment under panel_rigid (a spike), vs. a diluted
smooth error under RBF (a bulge) — same root mismatch, different failure
shape. Neither method fixes fine-mesh/crease-line misalignment by itself.

check_panel_boundary_strain() and check_inverted_elements() are general
geometric diagnostics — they take a `displacements` dict from ANY method
(not just panel_rigid) plus a `node_to_panel` assignment from
assign_fine_nodes_to_panels(), and report edges/elements that are
inconsistent with a single rigid panel motion. A nonzero count under
method="rbf" is meaningful too: it means RBF produced real local distortion
there, not that the check is somehow specific to panel_rigid's own
mechanism.

`surrogate` (the `panel_rigid`-only parameter of map_displacements_from_coarse)
accepts anything duck-typing `.nodes` / `.elements` / `.panel_map` — either a
Surrogate from build_surrogate(), or a plain Mesh straight from remesh()
(Path B, use_crease_mesh=True) when Steps 4-6 are skipped entirely, e.g. for
datasets with pre-solved (not AERO-S) fold displacements.
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

    header_idxs = [i for i, line in enumerate(lines) if line.strip().startswith("Vector DISP")]
    if not header_idxs:
        raise RuntimeError(f"No 'Vector DISP' block found in {disp_file}")

    blocks: list[tuple[int, int]] = []   # (data_start_line, n_nodes)
    if len(header_idxs) > 1:
        # One full "Vector DISP ..." + node-count header repeated before
        # every timestep block.
        for i in header_idxs:
            n_nodes = int(lines[i + 1].strip())
            blocks.append((i + 2, n_nodes))
    else:
        # Single header + node-count at the very top of the file, followed
        # by N repeating (optional 1-value time-stamp line + n_nodes data
        # rows) chunks with NO repeated header per timestep — seen on
        # gdisplac6.xpost under NLDynamic with LMPCs active. The
        # repeated-header assumption above silently collapses every
        # timestep into a single block in that case, so step=-1 (or any
        # step != 0) returned block 0 — t=0, everything still at its
        # undeformed position — instead of the requested step, with no
        # error raised. Walk the rest of the file in fixed-size chunks
        # instead, using the declared n_nodes as the stride.
        i0 = header_idxs[0]
        n_nodes = int(lines[i0 + 1].strip())
        data_lines = lines[i0 + 2:]
        first_parts = data_lines[0].strip().split() if data_lines else []
        has_time_line = len(first_parts) not in (6, 7)
        stride = n_nodes + (1 if has_time_line else 0)
        if stride <= 0 or len(data_lines) < stride:
            raise RuntimeError(
                f"Could not determine timestep block size in {disp_file} "
                f"(single header, declared n_nodes={n_nodes})."
            )
        n_blocks = len(data_lines) // stride
        for b in range(n_blocks):
            block_start = i0 + 2 + b * stride + (1 if has_time_line else 0)
            blocks.append((block_start, n_nodes))
        print(f"  Note: {disp_file} has one header covering {n_blocks} "
              f"timestep(s) of {n_nodes} nodes each (no per-timestep header) "
              f"— reading step {step} of that range.")

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
# Zero-displacement output repair (AERO-S/LMPC artifact)
# ─────────────────────────────────────────────────────────────────────────────

def _repair_zero_output_nodes(
    raw_disp: dict[int, tuple],
    coarse_nodes: dict[int, tuple],
    surrogate,
    skip_ids: set[int],
    tol: float = 1e-9,
) -> int:
    """
    Repair coarse nodes whose raw_disp entry is exact-zero across all 6 DOF
    but clearly shouldn't be, by averaging topological neighbors' values.
    Mutates raw_disp in place. Returns the number of nodes repaired.

    See map_displacements()'s repair_zero_displacements docstring for why
    this exists. A node is only repaired if:
      1. It's present in raw_disp (a node missing from xpost entirely is a
         different, already-warned-about case — left alone here).
      2. It's not in skip_ids (nodes seeded zero via config.disp_bcs are
         legitimately fixed, e.g. riser/bridle attachments).
      3. At least one topological neighbor has a non-zero raw_disp value —
         if every neighbor is also zero, this may be a genuinely-static
         region this early in the fold, not a solver artifact, so it's left
         as-is (with a separate warning) rather than guessed at.

    Neighbors come from real mesh connectivity — shared shell-element edges
    (surrogate.elements) plus joint pairs (surrogate.joints, i.e. the two
    duplicated copies of a crease node either side of a hinge) — never 3D
    proximity, so a repair can't accidentally blend across a fold gap where
    two panels are geometrically close but topologically unrelated.
    """
    if surrogate is None:
        return 0   # no connectivity available to find safe neighbors

    def _is_zero(disp: tuple) -> bool:
        return all(abs(v) < tol for v in disp)

    zero_ids = [
        nid for nid in coarse_nodes
        if nid not in skip_ids and nid in raw_disp and _is_zero(raw_disp[nid])
    ]
    if not zero_ids:
        return 0

    # Build undirected adjacency from shell/quad element edges + joints.
    adj: dict[int, set[int]] = defaultdict(set)
    for etype, nids in surrogate.elements.values():
        n = len(nids)
        if n < 2:
            continue
        for i in range(n):
            a, b = nids[i], nids[(i + 1) % n]
            adj[a].add(b)
            adj[b].add(a)
    for j in surrogate.joints:
        adj[j.node_a].add(j.node_b)
        adj[j.node_b].add(j.node_a)

    repaired, unrepaired = [], []
    for nid in zero_ids:
        neighbor_disps = [
            raw_disp[n] for n in adj.get(nid, ())
            if n in raw_disp and not _is_zero(raw_disp[n])
        ]
        if not neighbor_disps:
            unrepaired.append(nid)
            continue
        raw_disp[nid] = tuple(
            sum(d[k] for d in neighbor_disps) / len(neighbor_disps)
            for k in range(6)
        )
        repaired.append(nid)

    if repaired:
        print(f"  WARNING: {len(repaired)} coarse node(s) had an exact-zero "
              f"xpost displacement with non-zero neighbors (likely an AERO-S/"
              f"LMPC output artifact — see repair_zero_displacements "
              f"docstring). Repaired via neighbor averaging: {repaired[:10]}"
              + (" ..." if len(repaired) > 10 else ""))
    if unrepaired:
        print(f"  WARNING: {len(unrepaired)} coarse node(s) had an exact-zero "
              f"xpost displacement but no non-zero neighbor to repair from — "
              f"left as zero: {unrepaired[:10]}"
              + (" ..." if len(unrepaired) > 10 else ""))

    return len(repaired)


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
    rbf_kernel: str = "multiquadric",
    rbf_epsilon: float = 1.0,
    rbf_degree: int = 0,
    repair_zero_displacements: bool = True,
    repair_zero_tol: float = 1e-9,
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
                    "rbf"         — RBF from all coarse nodes; smooth but can
                                    average across fold lines. See
                                    rbf_kernel/rbf_epsilon/rbf_degree below and
                                    the module "Notes on kernel choice" docstring.
                    "panel_rigid" — per-panel Procrustes rigid-body transform;
                                    correct for large-angle folds, no cross-panel
                                    averaging.  Best near vent / high-curvature
                                    regions.  Requires surrogate.panel_map to be
                                    populated (always true after build_surrogate).
    rbf_neighbors : nearest-neighbour count for RBF (ignored for panel_rigid)
    rbf_smoothing : RBF smoothing factor; 0 = exact interpolation. The single
                    biggest lever against large, ill-conditioning-driven
                    spikes — see module docstring notes.
    rbf_kernel    : scipy.interpolate.RBFInterpolator kernel name. Kernels
                    needing rbf_epsilon: 'multiquadric' (default),
                    'inverse_multiquadric', 'inverse_quadratic', 'gaussian'
                    (all locally/globally decaying except multiquadric).
                    Kernels ignoring rbf_epsilon: 'linear', 'cubic',
                    'quintic', 'thin_plate_spline' (all need rbf_degree
                    high enough to stay full-rank — thin_plate_spline needs
                    rbf_degree >= 2 in 3D).
    rbf_epsilon   : shape parameter (kernel "width", same units as
                    coordinates) for the kernels listed above. Mismatched
                    epsilon vs. local coarse-point spacing is another common
                    cause of the same ill-conditioning/overshoot symptom as
                    rbf_smoothing=0 — tune both together.
    rbf_degree    : degree of the polynomial trend added to the RBF sum.
                    0 = constant only (default, always full-rank for
                    multiquadric). Higher degree can reduce oscillation for
                    fields that are locally close to affine (e.g. rigid
                    panel rotations) but needs enough coarse points to stay
                    full-rank.
    repair_zero_displacements : AERO-S has been observed to write an exact-
                    zero 6-DOF row in gdisplac6.xpost for a handful of coarse
                    nodes even when the fold clearly moved them — seen
                    specifically with LMPCs active (disabling them removes
                    the effect), likely an LMPC-related gap in AERO-S's own
                    output-vector bookkeeping rather than anything in this
                    pipeline. Exact zero (all 6 components, within
                    repair_zero_tol) from a converged nonlinear dynamic solve
                    is otherwise essentially impossible by chance, so it's a
                    reliable "this row is a placeholder, not a real result"
                    signal. When True (default), such a node is repaired by
                    averaging its topological neighbors' (shared shell
                    element edges + joints — real mesh connectivity, not 3D
                    nearest-neighbor, so a repair never bridges across a
                    fold gap) displacements, provided at least one neighbor
                    has a genuinely nonzero value; nodes seeded via
                    config.disp_bcs (legitimately zero) are never touched,
                    and a node with no usable neighbor is left as-is with a
                    warning. See _repair_zero_output_nodes().
    repair_zero_tol : magnitude below which a raw_disp entry (and a
                    candidate repair neighbor) is treated as "exactly zero".
                    Only used when repair_zero_displacements=True.

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

    extra_zero_seed = None
    if config is not None and config.disp_bcs:
        extra_zero_seed = [nid for nid, _ in config.disp_bcs]

    if repair_zero_displacements:
        _repair_zero_output_nodes(
            raw_disp, coarse_nodes, surrogate,
            skip_ids=set(extra_zero_seed or ()),
            tol=repair_zero_tol,
        )

    return map_displacements_from_coarse(
        coarse_nodes, raw_disp, fine_mesh,
        method=method, rbf_neighbors=rbf_neighbors, rbf_smoothing=rbf_smoothing,
        rbf_kernel=rbf_kernel, rbf_epsilon=rbf_epsilon, rbf_degree=rbf_degree,
        surrogate=surrogate, extra_zero_seed=extra_zero_seed,
    )


def map_displacements_from_coarse(
    coarse_nodes: dict[int, tuple],
    raw_disp: dict[int, tuple],
    fine_mesh,
    method: str = "rbf",
    rbf_neighbors: int = 100,
    rbf_smoothing: float = 1e-7,
    rbf_kernel: str = "multiquadric",
    rbf_epsilon: float = 1.0,
    rbf_degree: int = 0,
    surrogate=None,
    extra_zero_seed: list[int] | None = None,
) -> dict[int, tuple]:
    """
    Map already-known coarse node displacements onto the original fine mesh.

    Use this directly (instead of map_displacements) when the fold
    displacement was solved outside AERO-S — e.g. a kinematic rigid-origami
    model that hands you coarse node coordinates + displacements for each
    node, with no xpost file and no surrogate/config from Steps 1-6.

    Parameters
    ----------
    coarse_nodes     : {node_id: (x, y, z)} — undeformed coarse node positions
    raw_disp         : {node_id: (dx, dy, dz, rx, ry, rz)} — coarse node
                       displacements, same keys as coarse_nodes (a node
                       missing here is treated as zero displacement)
    fine_mesh        : Mesh (output of load_mesh on the original fine .fem)
    method           : "rbf" (default) or "panel_rigid". "panel_rigid"
                       requires `surrogate` (with panel_map/elements) — it
                       is not available for coarse node sets that didn't
                       come from build_surrogate().
    rbf_neighbors    : nearest-neighbour count for RBF (ignored for panel_rigid)
    rbf_smoothing    : RBF smoothing factor; 0 = exact interpolation. See
                       map_displacements()'s docstring — the biggest lever
                       against large, ill-conditioning-driven spikes.
    rbf_kernel       : RBFInterpolator kernel name — see map_displacements().
    rbf_epsilon      : RBF shape parameter — see map_displacements().
    rbf_degree       : RBF polynomial trend degree — see map_displacements().
    surrogate        : Surrogate — only needed for method="panel_rigid"
    extra_zero_seed  : optional list of fine-mesh node IDs to force to zero
                       displacement (e.g. known-fixed BC nodes such as riser/
                       bridle attachments) if not already resolved

    Returns
    -------
    dict[int, tuple[float,float,float,float,float,float]]
        {node_id: (dx, dy, dz, rx, ry, rz)} for every node in fine_mesh.nodes.
        Cable intermediate nodes not reachable via BFS are set to zero.
    """
    coarse_ids = list(coarse_nodes.keys())

    # ── 3. Interpolate membrane (canopy) nodes ────────────────────────────────
    fine_membrane_ids    = sorted(fine_mesh.membrane_nodes)
    fine_cable_only_ids  = sorted(fine_mesh.cable_nodes)

    print(f"  Mapping [{method}]: {len(coarse_nodes)} coarse nodes → "
          f"{len(fine_membrane_ids)} fine membrane nodes, "
          f"{len(fine_cable_only_ids)} cable nodes via arc-length")

    if method == "panel_rigid":
        if surrogate is None:
            raise ValueError(
                "method='panel_rigid' requires a surrogate (with panel_map/"
                "elements) — use method='rbf' when calling "
                "map_displacements_from_coarse() without one."
            )
        resolved = _panel_rigid_body_map(surrogate, fine_mesh, raw_disp,
                                         fine_membrane_ids)
    elif method == "rbf":
        resolved = _rbf_map(coarse_nodes, raw_disp, fine_mesh, fine_membrane_ids,
                            rbf_neighbors, rbf_smoothing, rbf_kernel,
                            rbf_epsilon, rbf_degree)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'rbf' or 'panel_rigid'.")

    # ── 4. Seed any known-fixed BC nodes with zero displacement ──────────────
    if extra_zero_seed:
        for nid in extra_zero_seed:
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
    rbf_kernel: str = "multiquadric",
    rbf_epsilon: float = 1.0,
    rbf_degree: int = 0,
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
        kernel=rbf_kernel, epsilon=rbf_epsilon, degree=rbf_degree,
        neighbors=rbf_neighbors, smoothing=rbf_smoothing,
    )
    rbf_rots = RBFInterpolator(
        coarse_coords, coarse_disp6[:, 3:],
        kernel=rbf_kernel, epsilon=rbf_epsilon, degree=rbf_degree,
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

def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Best-fit rigid rotation R and translation t (Kabsch algorithm) such that
    R @ P[i] + t ≈ Q[i] for corresponding point sets P (undeformed) and
    Q (deformed).
    """
    c_p = P.mean(axis=0)
    c_q = Q.mean(axis=0)

    H = (P - c_p).T @ (Q - c_q)
    U, _, Vt = np.linalg.svd(H)

    # Enforce proper rotation (det = +1, not a reflection)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = c_q - R @ c_p
    return R, t


def assign_fine_nodes_to_panels(
    panel_source,
    fine_mesh,
    fine_ids: list[int] | None = None,
    k_candidates: int = 8,
    tol: float = 1e-3,
) -> tuple[dict[int, int], int]:
    """
    Assign each fine mesh node to exactly one panel from panel_source, via
    topology — point-in-triangle containment against that panel's own
    triangle(s) in undeformed space — rather than nearest-panel-centroid.

    Nearest-centroid can misassign a fine node to the wrong panel whenever a
    neighbouring panel's centroid happens to be numerically closer than the
    correct panel's own far edge — a real risk for coarse, few-node,
    irregularly-shaped panels (exactly the case for a directly-supplied
    coarse pattern with no Gmsh refinement). Quad panel elements are tested
    as two triangles (0-1-2 / 0-2-3 fan split, matching the split_quads
    convention used elsewhere) — this only affects which panel a query point
    falls in, not the panel's own geometry/DOFs.

    Parameters
    ----------
    panel_source : Surrogate or plain Mesh (e.g. from remesh()) — duck-typed,
                   needs .nodes / .elements / .panel_map.
    fine_ids      : fine mesh node IDs to assign; defaults to all membrane
                    nodes.
    k_candidates  : nearest panel-triangle centroids considered per fine node
                    (efficiency only — the real test is exact containment).
    tol           : barycentric-coordinate slack for containment.

    Returns
    -------
    node_to_panel : {fine_node_id: panel_id}
    n_fallback    : count of fine nodes not exactly contained by any
                    candidate triangle (assigned to the nearest triangle's
                    panel instead) — large values mean the fine mesh doesn't
                    tightly track this panel_source's crease lines.
    """
    from scipy.spatial import cKDTree

    if fine_ids is None:
        fine_ids = sorted(fine_mesh.membrane_nodes)

    # ── Build (triangle, panel_id) list from panel_source topology ──────────
    tri_v0: list = []
    tri_v1: list = []
    tri_v2: list = []
    tri_pid: list = []
    for eid, (_, conn) in panel_source.elements.items():
        if len(conn) < 3:
            continue
        pid = panel_source.panel_map.get(eid)
        if pid is None or not all(n in panel_source.nodes for n in conn):
            continue
        pts = [panel_source.nodes[n] for n in conn]
        if len(conn) == 3:
            fans = [(pts[0], pts[1], pts[2])]
        elif len(conn) == 4:
            fans = [(pts[0], pts[1], pts[2]), (pts[0], pts[2], pts[3])]
        else:
            fans = [(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]
        for a, b, c in fans:
            tri_v0.append(a)
            tri_v1.append(b)
            tri_v2.append(c)
            tri_pid.append(pid)

    if not tri_pid:
        raise RuntimeError(
            "assign_fine_nodes_to_panels: panel_source has no usable panel "
            "triangles — check .elements / .panel_map are populated."
        )

    v0 = np.array(tri_v0, dtype=float)
    v1 = np.array(tri_v1, dtype=float)
    v2 = np.array(tri_v2, dtype=float)
    tri_pid_arr = np.array(tri_pid, dtype=int)
    centroids   = (v0 + v1 + v2) / 3.0

    tree = cKDTree(centroids)
    fine_coords = np.array([fine_mesh.nodes[nid] for nid in fine_ids], dtype=float)
    k = min(k_candidates, len(centroids))
    _, cand = tree.query(fine_coords, k=k)
    cand = cand.reshape(len(fine_ids), k)     # cKDTree squeezes the axis when k==1

    # ── Vectorised barycentric containment test against the k candidates ─────
    cv0, cv1, cv2 = v0[cand], v1[cand], v2[cand]      # (n_fine, k, 3)
    e1 = cv1 - cv0
    e2 = cv2 - cv0
    ep = fine_coords[:, None, :] - cv0

    d11 = np.einsum('nkj,nkj->nk', e1, e1)
    d12 = np.einsum('nkj,nkj->nk', e1, e2)
    d22 = np.einsum('nkj,nkj->nk', e2, e2)
    d1p = np.einsum('nkj,nkj->nk', ep, e1)
    d2p = np.einsum('nkj,nkj->nk', ep, e2)

    denom = d11 * d22 - d12 * d12
    denom = np.where(denom == 0, 1e-30, denom)
    u = (d22 * d1p - d12 * d2p) / denom
    v = (d11 * d2p - d12 * d1p) / denom
    inside = (u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)   # (n_fine, k)

    any_inside = inside.any(axis=1)
    first_hit  = np.argmax(inside, axis=1)             # 0 if none inside
    chosen_col = np.where(any_inside, first_hit, 0)     # fallback: nearest (col 0)

    rows        = np.arange(len(fine_ids))
    chosen_tri  = cand[rows, chosen_col]
    chosen_pid  = tri_pid_arr[chosen_tri]
    n_fallback  = int((~any_inside).sum())

    node_to_panel = {nid: int(pid) for nid, pid in zip(fine_ids, chosen_pid)}

    if n_fallback:
        print(f"  panel assignment: {n_fallback}/{len(fine_ids)} fine node(s) "
              f"not exactly contained by any candidate panel triangle "
              f"(assigned to nearest instead) — large counts suggest the "
              f"fine mesh doesn't tightly track this pattern's crease lines.")

    return node_to_panel, n_fallback


def _panel_rigid_body_map(
    panel_source,
    fine_mesh,
    raw_disp: dict,
    fine_membrane_ids: list[int],
) -> dict[int, tuple]:
    """
    For each panel, find the best-fit rigid-body rotation R and translation t
    (Procrustes / Kabsch algorithm) mapping undeformed → deformed panel node
    positions.  Each fine membrane node is assigned to exactly one panel via
    topology (assign_fine_nodes_to_panels — point-in-triangle containment,
    not nearest-centroid) and displaced by that panel's R and t.  There is no
    blending at crease boundaries — see the module-level "Notes on
    method=panel_rigid" docstring for the assumption this relies on.

    Rotation DOFs (rx, ry, rz) are the rotation-vector components of R,
    consistent with AERO-S large-rotation shell output.

    panel_source : Surrogate or plain Mesh (e.g. from remesh()) — duck-typed,
                   needs .nodes / .elements / .panel_map.
    """
    from scipy.spatial.transform import Rotation

    # ── Build panel → node set from panel_source topology ────────────────────
    # panel_source.elements already has the correct node IDs for each panel's
    # own side of every crease (duplicated per-panel for a Surrogate;
    # shared/undeformed for a plain remesh() Mesh — either way each panel's
    # node set is exactly its own boundary+interior nodes).
    panel_to_nodes: dict[int, set[int]] = defaultdict(set)
    for eid, (_, conn) in panel_source.elements.items():
        if len(conn) >= 3:
            pid = panel_source.panel_map.get(eid)
            if pid is not None:
                panel_to_nodes[pid].update(conn)

    # ── Procrustes per panel ──────────────────────────────────────────────────
    panel_R:  dict[int, np.ndarray] = {}
    panel_t:  dict[int, np.ndarray] = {}
    panel_rv: dict[int, np.ndarray] = {}   # rotation vector for DOFs 4-6

    skipped = 0
    for pid, node_set in panel_to_nodes.items():
        # Keep only nodes that appear in both panel_source and raw_disp
        nids = [n for n in node_set if n in panel_source.nodes and n in raw_disp]
        if len(nids) < 3:
            skipped += 1
            continue

        P = np.array([panel_source.nodes[n] for n in nids], dtype=float)   # undeformed
        Q = np.array(
            [(panel_source.nodes[n][0] + raw_disp[n][0],
              panel_source.nodes[n][1] + raw_disp[n][1],
              panel_source.nodes[n][2] + raw_disp[n][2])
             for n in nids],
            dtype=float,
        )  # deformed

        R, t = _kabsch(P, Q)
        panel_R[pid]  = R
        panel_t[pid]  = t
        panel_rv[pid] = Rotation.from_matrix(R).as_rotvec()

    if skipped:
        print(f"  panel_rigid: {skipped} panel(s) skipped (< 3 nodes with displacement data)")
    print(f"  panel_rigid: {len(panel_R)} panels with Procrustes transforms")

    if not panel_R:
        raise RuntimeError(
            "panel_rigid: no panels could be solved. "
            "Ensure panel_source.panel_map is populated and node IDs match raw_disp."
        )

    # ── Topology-based fine-node → panel assignment ───────────────────────────
    node_to_panel, _ = assign_fine_nodes_to_panels(panel_source, fine_mesh, fine_membrane_ids)

    # ── Apply panel transform to each fine node ───────────────────────────────
    resolved: dict[int, tuple] = {}
    n_unassigned = 0
    for nid in fine_membrane_ids:
        pid = node_to_panel.get(nid)
        if pid not in panel_R:
            # Assigned panel had < 3 resolvable nodes (skipped above) — leave
            # unresolved; map_displacements_from_coarse packs this as zero.
            n_unassigned += 1
            continue
        R  = panel_R[pid]
        t  = panel_t[pid]
        rv = panel_rv[pid]
        x  = np.array(fine_mesh.nodes[nid], dtype=float)
        d  = R @ x + t - x          # displacement = deformed_pos - original_pos
        resolved[nid] = (
            float(d[0]),  float(d[1]),  float(d[2]),
            float(rv[0]), float(rv[1]), float(rv[2]),
        )

    if n_unassigned:
        print(f"  panel_rigid: {n_unassigned} fine node(s) assigned to a skipped "
              f"panel (< 3 resolvable nodes) — left unresolved.")

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


def write_wireframe_vtk(
    nodes: dict[int, tuple],
    displacements: dict[int, tuple],
    edges: list[tuple[int, int]],
    output_path: str | Path,
) -> None:
    """
    Write a VTK polydata wireframe (line segments only) for ParaView.

    For visualizing a node+edge set that has no shell-element connectivity —
    e.g. a coarse crease pattern's own edges.csv topology — alongside
    write_folded_vtk()'s fine-mesh surface, so the pre-interpolation ground
    truth and the interpolated result can be compared in the same viewer.
    Node positions are shifted by the translation component of displacements
    (a 3-tuple, or a 6-tuple as elsewhere in this module — only dx,dy,dz
    are used).

    Parameters
    ----------
    nodes         : {node_id: (x, y, z)} — undeformed positions
    displacements : {node_id: (dx, dy, dz, ...)} — a node missing here, or
                    whose id isn't in `nodes`, is dropped from output
    edges         : [(node_id_a, node_id_b), ...] — line segments to draw
    output_path   : .vtk file to write
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zero3 = (0.0, 0.0, 0.0)
    nid_list   = list(nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}

    coords = []
    for nid in nid_list:
        x, y, z = nodes[nid]
        dx, dy, dz = displacements.get(nid, zero3)[:3]
        coords.append((x + dx, y + dy, z + dz))

    valid_edges = [
        (a, b) for a, b in edges if a in nid_to_idx and b in nid_to_idx
    ]

    with open(output_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("AeroOrigami coarse wireframe\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"\nPOINTS {len(coords)} float\n")
        for x, y, z in coords:
            f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"\nCELLS {len(valid_edges)} {len(valid_edges) * 3}\n")
        for a, b in valid_edges:
            f.write(f"2 {nid_to_idx[a]} {nid_to_idx[b]}\n")

        f.write(f"\nCELL_TYPES {len(valid_edges)}\n")
        for _ in valid_edges:
            f.write("3\n")

        f.write(f"\nPOINT_DATA {len(nid_list)}\n")
        f.write("VECTORS displacement float\n")
        for nid in nid_list:
            dx, dy, dz = displacements.get(nid, zero3)[:3]
            f.write(f"{dx:.8e} {dy:.8e} {dz:.8e}\n")

    print(f"  Wrote {len(valid_edges)} edges to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics for method="panel_rigid" (no-blending assumption)
# ─────────────────────────────────────────────────────────────────────────────

def check_panel_boundary_strain(
    fine_mesh,
    displacements: dict[int, tuple],
    node_to_panel: dict[int, int],
    tol: float = 0.05,
) -> dict:
    """
    General-purpose geometric diagnostic, usable with displacements from ANY
    method (rbf or panel_rigid) — not specific to panel_rigid's own no-
    blending mechanism. For every fine-mesh membrane edge whose two endpoints
    are assigned to different panels (per node_to_panel, a purely
    topological/undeformed-space assignment independent of how
    `displacements` was computed), compute strain =
    deformed/undeformed length - 1. A nonzero result under method="rbf"
    means RBF produced real local distortion there — not that this check
    only means something for panel_rigid. Useful in general for turning
    "the crease gap looks smaller than the mesh" from an eyeballed
    assumption into a checked number, and for spotting fine-mesh/crease-line
    misalignment (see mapping.py's "Notes on method=panel_rigid").

    Parameters
    ----------
    fine_mesh      : Mesh (output of load_mesh)
    displacements  : {node_id: (dx,dy,dz,rx,ry,rz)} — output of
                      map_displacements_from_coarse, any method
    node_to_panel  : {node_id: panel_id} — from assign_fine_nodes_to_panels,
                      using any panel_source with real panel topology (does
                      not need to be the method that produced displacements)
    tol            : |strain| above this is reported in 'flagged'

    Returns
    -------
    dict with 'n_edges', 'max_strain', 'mean_strain', and 'flagged'
    (list of (nid_a, nid_b, strain), worst first).
    """
    zero6 = (0.0,) * 6
    seen: set[frozenset] = set()
    strains: list[tuple[int, int, float]] = []

    for _, (_, conn) in fine_mesh.membrane_elements.items():
        n = len(conn)
        for i in range(n):
            a, b = conn[i], conn[(i + 1) % n]
            pa, pb = node_to_panel.get(a), node_to_panel.get(b)
            if pa is None or pb is None or pa == pb:
                continue
            edge = frozenset((a, b))
            if edge in seen:
                continue
            seen.add(edge)

            xa = np.array(fine_mesh.nodes[a], dtype=float)
            xb = np.array(fine_mesh.nodes[b], dtype=float)
            l0 = float(np.linalg.norm(xb - xa))
            if l0 < 1e-12:
                continue
            da = np.array(displacements.get(a, zero6)[:3], dtype=float)
            db = np.array(displacements.get(b, zero6)[:3], dtype=float)
            l1 = float(np.linalg.norm((xb + db) - (xa + da)))
            strains.append((a, b, l1 / l0 - 1.0))

    if not strains:
        print("  check_panel_boundary_strain: no panel-boundary edges found.")
        return {"n_edges": 0, "max_strain": 0.0, "mean_strain": 0.0, "flagged": []}

    mags    = [abs(s) for _, _, s in strains]
    flagged = sorted((e for e in strains if abs(e[2]) > tol), key=lambda e: -abs(e[2]))

    print(f"  check_panel_boundary_strain: {len(strains)} panel-boundary edge(s), "
          f"strain max={max(mags):.4f} mean={sum(mags) / len(mags):.4f} "
          f"(tol={tol}) — {len(flagged)} over tolerance")
    if flagged:
        for a, b, s in flagged[:10]:
            print(f"    nodes {a},{b}: strain={s:+.4f}")
        if len(flagged) > 10:
            print(f"    ... and {len(flagged) - 10} more")

    return {
        "n_edges": len(strains),
        "max_strain": max(mags),
        "mean_strain": sum(mags) / len(mags),
        "flagged": flagged,
    }


def check_inverted_elements(
    fine_mesh,
    displacements: dict[int, tuple],
    node_to_panel: dict[int, int],
) -> list[int]:
    """
    General-purpose geometric diagnostic, usable with displacements from ANY
    method — not specific to panel_rigid's own mechanism. Flags fine-mesh
    membrane elements spanning >= 2 panels (per node_to_panel; concentrated
    at panel junctions, e.g. a vent hub, and at any fine element a real fold
    line crosses that the fine mesh's own triangulation doesn't track) whose
    normal flips sign after folding. Under panel_rigid this specifically
    means the element got crushed by a hard single-panel assignment; under
    rbf it means the smooth interpolant produced real local distortion at
    that element regardless — both are genuine inversions worth knowing
    about. Elements assigned entirely to one panel are not checked (under
    panel_rigid they provably cannot invert — one uniform rigid transform,
    det=+1 by construction — but under rbf a single-panel element could in
    principle still distort; this check only ever targets the
    junction/misaligned-boundary case, the failure mode most likely to
    actually matter).

    Parameters
    ----------
    fine_mesh      : Mesh (output of load_mesh)
    displacements  : {node_id: (dx,dy,dz,rx,ry,rz)} — any method
    node_to_panel  : {node_id: panel_id} — from assign_fine_nodes_to_panels

    Returns
    -------
    list of flagged element IDs.
    """
    zero6 = (0.0,) * 6
    flagged: list[int] = []

    for eid, (_, conn) in fine_mesh.membrane_elements.items():
        if len(conn) < 3:
            continue
        pids = {node_to_panel.get(n) for n in conn}
        if len(pids) < 2:
            continue   # single-panel element: cannot invert

        p0, p1, p2 = (np.array(fine_mesh.nodes[n], dtype=float) for n in conn[:3])
        n0 = np.cross(p1 - p0, p2 - p0)

        d0, d1, d2 = (np.array(displacements.get(n, zero6)[:3], dtype=float) for n in conn[:3])
        n1 = np.cross((p1 + d1) - (p0 + d0), (p2 + d2) - (p0 + d0))

        if float(np.dot(n0, n1)) <= 0.0:
            flagged.append(eid)

    if flagged:
        print(f"  check_inverted_elements: {len(flagged)} multi-panel element(s) "
              f"inverted after folding (first few: {flagged[:10]})")
    else:
        print("  check_inverted_elements: no inverted elements found.")

    return flagged
