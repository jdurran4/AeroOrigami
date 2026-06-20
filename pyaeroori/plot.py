"""
plot.py — visualization helpers for Steps 1 and 2.

Step 1:  mesh_stats(mesh)              — print node/element counts and extents
         plot_mesh(mesh)               — node scatter + element wireframe

Step 2:  crease_stats(cp)              — print fold counts and extents
         plot_creases(cp)              — mountain / valley / boundary lines
         plot_creases_on_mesh(mesh,cp) — crease lines overlaid on mesh
         check_crease_coverage(mesh,cp)— nearest-node proximity check

All plot_* functions return (fig, ax) so callers can add annotations.
Requires: matplotlib.  check_crease_coverage also requires scipy.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .crease import CreasePattern
    from .surrogate import Surrogate
    from .physics import ModelConfig


# ── Stats ─────────────────────────────────────────────────────────────────────

def mesh_stats(mesh: "Mesh") -> None:
    """Print a node/element summary and coordinate extents."""
    coords = list(mesh.nodes.values())
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    print("Mesh summary")
    print(f"  Nodes             : {len(mesh.nodes)}")
    print(f"  Membrane elements : {len(mesh.membrane_elements)}")
    print(f"  Cable elements    : {len(mesh.cable_elements)}")
    print(f"  x  [{min(xs):.4f},  {max(xs):.4f}]")
    print(f"  y  [{min(ys):.4f},  {max(ys):.4f}]")
    print(f"  z  [{min(zs):.4f},  {max(zs):.4f}]")


def crease_stats(cp: "CreasePattern") -> None:
    """Print a fold-count summary and coordinate extents."""
    print("Crease pattern summary")
    print(f"  Mountain folds : {len(cp.mountain)}")
    print(f"  Valley folds   : {len(cp.valley)}")
    print(f"  Boundary edges : {len(cp.boundary)}")

    all_pts: list = []
    for p1, p2, *_ in cp.mountain + cp.valley:
        all_pts += [p1, p2]
    for p1, p2 in cp.boundary:
        all_pts += [p1, p2]

    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        print(f"  x  [{min(xs):.4f},  {max(xs):.4f}]")
        print(f"  y  [{min(ys):.4f},  {max(ys):.4f}]")
        print(f"  z  [{min(zs):.4f},  {max(zs):.4f}]")


# ── Step 1: mesh plot ─────────────────────────────────────────────────────────

def plot_mesh(
    mesh: "Mesh",
    title: str = "Mesh",
    show: bool = True,
    max_nodes: int = 20_000,
):
    """
    3-D scatter of mesh nodes coloured by type, plus element wireframe for
    small meshes (<= 5 000 elements).

    Parameters
    ----------
    max_nodes : max nodes to render (subsampled if the mesh is larger)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    mem_nids = sorted(mesh.membrane_nodes)
    cab_nids = sorted(mesh.cable_nodes)

    def _scatter(nids, color, label):
        step = max(1, len(nids) // max_nodes)
        s = nids[::step]
        if not s:
            return
        xs = [mesh.nodes[n][0] for n in s]
        ys = [mesh.nodes[n][1] for n in s]
        zs = [mesh.nodes[n][2] for n in s]
        ax.scatter(xs, ys, zs, c=color, s=1, alpha=0.4,
                   label=f"{label} ({len(nids):,})")

    _scatter(mem_nids, "steelblue",  "Membrane nodes")
    _scatter(cab_nids, "darkorange", "Cable nodes")

    # Wireframe only for small meshes
    if len(mesh.elements) <= 5_000:
        for _, (_, node_ids) in mesh.elements.items():
            coords = [mesh.nodes[n] for n in node_ids]
            xs = [c[0] for c in coords] + [coords[0][0]]
            ys = [c[1] for c in coords] + [coords[0][1]]
            zs = [c[2] for c in coords] + [coords[0][2]]
            ax.plot(xs, ys, zs, "k-", lw=0.3, alpha=0.25)
    else:
        ax.text2D(0.02, 0.02,
                  f"({len(mesh.elements):,} elements — wireframe omitted)",
                  transform=ax.transAxes, fontsize=8, color="grey")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ── Step 2: crease plots ──────────────────────────────────────────────────────

def plot_creases(
    cp: "CreasePattern",
    title: str = "Crease Pattern",
    show: bool = True,
):
    """Mountain (red), valley (blue), boundary (grey dashed) fold lines."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_folds(ax,  cp.mountain,  "crimson",   "Mountain", lw=1.2)
    _draw_folds(ax,  cp.valley,    "royalblue", "Valley",   lw=1.2)
    _draw_bounds(ax, cp.boundary,  "dimgrey",   "Boundary")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_creases_on_mesh(
    mesh: "Mesh",
    cp: "CreasePattern",
    title: str = "Crease Pattern on Mesh",
    show: bool = True,
    max_nodes: int = 20_000,
):
    """Mesh node cloud (grey) with crease fold lines overlaid."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Mesh node cloud
    all_nids = list(mesh.nodes.keys())
    step = max(1, len(all_nids) // max_nodes)
    s = all_nids[::step]
    xs = [mesh.nodes[n][0] for n in s]
    ys = [mesh.nodes[n][1] for n in s]
    zs = [mesh.nodes[n][2] for n in s]
    ax.scatter(xs, ys, zs, c="grey", s=0.5, alpha=0.3, label="Mesh nodes")

    _draw_folds(ax,  cp.mountain,  "crimson",   "Mountain", lw=1.5)
    _draw_folds(ax,  cp.valley,    "royalblue", "Valley",   lw=1.5)
    _draw_bounds(ax, cp.boundary,  "dimgrey",   "Boundary")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ── Step 2: coverage check ────────────────────────────────────────────────────

def check_crease_coverage(
    mesh: "Mesh",
    cp: "CreasePattern",
    tol: float = 0.05,
) -> dict:
    """
    For every crease endpoint, find the nearest membrane mesh node and report
    what fraction fall within `tol`.

    This is a geometric sanity check: before remeshing (Step 3), crease
    endpoints don't need to land exactly on existing nodes, but they should
    sit within the mesh domain.  Adjust `tol` to the typical element size
    for a meaningful match rate.

    Returns
    -------
    dict with n_endpoints, n_matched, max_dist, unmatched_pts
    """
    try:
        from scipy.spatial import cKDTree
        import numpy as np
    except ImportError:
        print("scipy not available — skipping proximity check")
        return {}

    mem_nids = sorted(mesh.membrane_nodes)
    if not mem_nids:
        print("No membrane nodes — skipping proximity check")
        return {}

    node_coords = np.array([mesh.nodes[n] for n in mem_nids])
    tree = cKDTree(node_coords)

    endpoints: list = []
    for p1, p2, *_ in cp.all_folds:
        endpoints += [p1, p2]
    for p1, p2 in cp.boundary:
        endpoints += [p1, p2]

    if not endpoints:
        return {}

    pts = np.array(endpoints)
    dists, _ = tree.query(pts)

    matched   = dists <= tol
    n_total   = len(pts)
    n_matched = int(matched.sum())
    max_dist  = float(dists.max())
    unmatched = [tuple(float(v) for v in pts[i]) for i in range(n_total) if not matched[i]]

    print(f"Crease coverage check  (tol={tol})")
    print(f"  Endpoints         : {n_total}")
    print(f"  Within tolerance  : {n_matched}  ({100*n_matched/n_total:.1f}%)")
    print(f"  Max nearest dist  : {max_dist:.6f}")
    if unmatched:
        print(f"  Unmatched pts     : {len(unmatched)}")
        for pt in unmatched[:5]:
            print(f"    {tuple(f'{v:.4f}' for v in pt)}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched)-5} more")
    else:
        print("  All endpoints within tolerance!")

    return {
        "n_endpoints": n_total,
        "n_matched":   n_matched,
        "max_dist":    max_dist,
        "unmatched":   unmatched,
    }


# ── Step 3: mesh resolution check ────────────────────────────────────────────

def check_mesh_crease_resolution(
    coarse: "Mesh",
    cp: "CreasePattern",
    tol: float = 1e-3,
) -> dict:
    """
    After Path A remesh(), verify that every type-C (mountain/valley) crease
    segment has at least one mesh node in its strict interior.

    A segment with no interior node will not receive a revolute driver joint
    in Step 4 — the fold cannot be actuated.  This happens when mesh_size
    is larger than the crease segment length.

    Parameters
    ----------
    coarse : Mesh returned by remesh() (Path A)
    cp     : CreasePattern used for the remesh call
    tol    : distance a node may sit off the segment line and still count

    Returns
    -------
    dict with n_total, n_missing, missing_segs, suggested_mesh_size
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        print("scipy not available — skipping resolution check")
        return {}

    mem_nids = sorted(coarse.membrane_nodes)
    if not mem_nids:
        print("No membrane nodes — skipping resolution check")
        return {}

    node_coords = np.array([coarse.nodes[n] for n in mem_nids])
    tree        = cKDTree(node_coords)

    all_segs = [
        (np.asarray(p1, float), np.asarray(p2, float))
        for p1, p2, *_ in cp.mountain + cp.valley
    ]
    if not all_segs:
        return {}

    missing:  list = []
    seg_lens: list = []

    for p1, p2 in all_segs:
        v       = p2 - p1
        seg_len = float(np.linalg.norm(v))
        if seg_len < 1e-12:
            continue
        seg_lens.append(seg_len)

        # Query all nodes within seg_len/2 + tol of the segment midpoint
        mid    = 0.5 * (p1 + p2)
        radius = seg_len * 0.5 + tol
        idxs   = tree.query_ball_point(mid, radius)

        found = False
        for i in idxs:
            pt = node_coords[i]
            if (np.linalg.norm(pt - p1) < tol or
                    np.linalg.norm(pt - p2) < tol):
                continue                           # skip endpoints
            w = pt - p1
            t = float(np.dot(w, v)) / (seg_len ** 2)
            if t <= tol / seg_len or t >= 1 - tol / seg_len:
                continue
            if np.linalg.norm(pt - (p1 + t * v)) < tol:
                found = True
                break

        if not found:
            missing.append((p1, p2, seg_len))

    n_total   = len(all_segs)
    n_missing = len(missing)
    min_len   = min(seg_lens) if seg_lens else 0.0

    print("Crease mesh resolution check")
    print(f"  Type-C segments     : {n_total}")
    print(f"  Missing interior nodes: {n_missing}  "
          f"({100 * n_missing / max(n_total, 1):.1f}%)")
    if n_missing > 0:
        suggested = min_len / 2
        print(f"  Shortest segment    : {min_len:.4f} m")
        print(f"  Suggested mesh_size : <= {suggested:.4f} m")
        print(f"  WARNING: {n_missing} crease segment(s) have no interior mesh "
              f"nodes and will not be actuated in Step 4.")
    else:
        print("  All segments have interior nodes — full actuation coverage.")

    return {
        "n_total":             n_total,
        "n_missing":           n_missing,
        "missing_segs":        missing,
        "suggested_mesh_size": min_len / 2 if min_len > 0 else None,
    }


# ── Step 4: surrogate visualization ──────────────────────────────────────────

def plot_surrogate_axes(
    surrogate: "Surrogate",
    title: str = "Surrogate — Crease Hinge Axes (Step 4)",
    show: bool = True,
    arrow_length: float | None = None,
):
    """
    Quiver plot of revolute driver joint hinge axes.

    Each arrow is placed at the joint node position and points along the
    crease axis of rotation.  Mountain folds are red, valley folds blue,
    spherical joints (boundary) are shown as grey dots.

    Parameters
    ----------
    arrow_length : length of quiver arrows in mesh coordinates.
                   Defaults to ~5 % of the mesh bounding-box diagonal.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    nodes = surrogate.nodes

    # Estimate a sensible arrow length from the mesh extent
    if arrow_length is None:
        coords = np.array(list(nodes.values()))
        diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        arrow_length = max(diag * 0.05, 1e-3)

    rev_mountain_x, rev_mountain_y, rev_mountain_z = [], [], []
    rev_mountain_u, rev_mountain_v, rev_mountain_w = [], [], []
    rev_valley_x,   rev_valley_y,   rev_valley_z   = [], [], []
    rev_valley_u,   rev_valley_v,   rev_valley_w   = [], [], []
    sph_x, sph_y, sph_z = [], [], []

    for j in surrogate.joints:
        cx, cy, cz = nodes[j.node_a]
        if j.jtype == 120:
            sph_x.append(cx); sph_y.append(cy); sph_z.append(cz)
        else:
            ax_, ay_, az_ = j.axis
            if j.target_angle >= 0:
                rev_mountain_x.append(cx); rev_mountain_y.append(cy); rev_mountain_z.append(cz)
                rev_mountain_u.append(ax_); rev_mountain_v.append(ay_); rev_mountain_w.append(az_)
            else:
                rev_valley_x.append(cx); rev_valley_y.append(cy); rev_valley_z.append(cz)
                rev_valley_u.append(ax_); rev_valley_v.append(ay_); rev_valley_w.append(az_)

    def _quiver(xs, ys, zs, us, vs, ws, color, label):
        if not xs:
            return
        ax.quiver(xs, ys, zs, us, vs, ws,
                  length=arrow_length, normalize=True,
                  color=color, alpha=0.8, label=label,
                  arrow_length_ratio=0.25, linewidth=1.2)

    _quiver(rev_mountain_x, rev_mountain_y, rev_mountain_z,
            rev_mountain_u, rev_mountain_v, rev_mountain_w,
            "crimson", f"Mountain revolute ({len(rev_mountain_x)})")
    _quiver(rev_valley_x, rev_valley_y, rev_valley_z,
            rev_valley_u, rev_valley_v, rev_valley_w,
            "royalblue", f"Valley revolute ({len(rev_valley_x)})")

    if sph_x:
        ax.scatter(sph_x, sph_y, sph_z, c="dimgrey", s=15, alpha=0.5,
                   label=f"Spherical boundary ({len(sph_x)})")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ── Step 5: physics overview plot ────────────────────────────────────────────

def plot_physics(
    surrogate:    "Surrogate",
    config:       "ModelConfig | None" = None,
    title:        str = "Surrogate — Physics Overview (Step 5)",
    show:         bool = True,
    arrow_length: float | None = None,
):
    """
    Comprehensive 3-D plot of the fold surrogate with physics annotations.

    Layers (listed front-to-back in z-order):
      - Mesh wireframe (thin grey)
      - Cable elements (teal lines)
      - Revolute hinge axes (crimson = mountain, royalblue = valley, quiver arrows)
      - Spherical joints (grey dots)
      - LMPC-constrained nodes (★ gold stars)
      - DISP-constrained nodes (▼ orange triangles)
      - Force-applied nodes (○ lime circles)

    Parameters
    ----------
    config : ModelConfig from add_physics(); if None, only joints are shown.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    nodes = surrogate.nodes

    # Combined node lookup (surrogate + any new cable nodes)
    all_nodes = dict(nodes)
    if config is not None and config.cable_nodes:
        all_nodes.update(config.cable_nodes)

    # Arrow length from bounding box
    if arrow_length is None:
        coords = np.array(list(nodes.values()))
        diag   = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        arrow_length = max(diag * 0.05, 1e-3)

    # ── Mesh wireframe ────────────────────────────────────────────────────────
    drawn_edges: set = set()
    first_wire = True
    for eid in sorted(surrogate.elements):
        _, nids = surrogate.elements[eid]
        if len(nids) < 3:
            continue
        n = len(nids)
        for i in range(n):
            e = (min(nids[i], nids[(i+1)%n]), max(nids[i], nids[(i+1)%n]))
            if e in drawn_edges:
                continue
            drawn_edges.add(e)
            x0, y0, z0 = nodes[nids[i]]
            x1, y1, z1 = nodes[nids[(i+1)%n]]
            ax.plot([x0, x1], [y0, y1], [z0, z1],
                    color="silver", lw=0.4, alpha=0.5,
                    label="Mesh edges" if first_wire else None)
            first_wire = False

    # ── Cable elements ────────────────────────────────────────────────────────
    if config is not None and config.cable_elements:
        first_cable = True
        for eid, _, nids in config.cable_elements:
            if len(nids) != 2:
                continue
            n0, n1 = nids
            if n0 not in all_nodes or n1 not in all_nodes:
                continue
            x0, y0, z0 = all_nodes[n0]
            x1, y1, z1 = all_nodes[n1]
            ax.plot([x0, x1], [y0, y1], [z0, z1],
                    color="teal", lw=1.5, alpha=0.8,
                    label=f"Cables ({len(config.cable_elements)})" if first_cable else None)
            first_cable = False

    # ── Revolute and spherical joints ─────────────────────────────────────────
    (rm_x, rm_y, rm_z, rm_u, rm_v, rm_w,
     rv_x, rv_y, rv_z, rv_u, rv_v, rv_w,
     sp_x, sp_y, sp_z) = ([] for _ in range(15))

    for j in surrogate.joints:
        cx, cy, cz = nodes[j.node_a]
        if j.jtype == 120:
            sp_x.append(cx); sp_y.append(cy); sp_z.append(cz)
        else:
            u, v, w = j.axis
            if j.target_angle >= 0:
                rm_x.append(cx); rm_y.append(cy); rm_z.append(cz)
                rm_u.append(u);  rm_v.append(v);  rm_w.append(w)
            else:
                rv_x.append(cx); rv_y.append(cy); rv_z.append(cz)
                rv_u.append(u);  rv_v.append(v);  rv_w.append(w)

    def _quiver(xs, ys, zs, us, vs, ws, color, label):
        if not xs:
            return
        ax.quiver(xs, ys, zs, us, vs, ws,
                  length=arrow_length, normalize=True,
                  color=color, alpha=0.8, label=label,
                  arrow_length_ratio=0.25, linewidth=1.0)

    _quiver(rm_x, rm_y, rm_z, rm_u, rm_v, rm_w,
            "crimson",   f"Mountain revolute ({len(rm_x)})")
    _quiver(rv_x, rv_y, rv_z, rv_u, rv_v, rv_w,
            "royalblue", f"Valley revolute ({len(rv_x)})")
    if sp_x:
        ax.scatter(sp_x, sp_y, sp_z, c="dimgrey", s=18, alpha=0.6, zorder=5,
                   label=f"Spherical ({len(sp_x)})")

    # ── Physics BC annotations ────────────────────────────────────────────────
    if config is not None:

        # LMPC nodes — gold stars
        lmpc_nids: set[int] = set()
        for row in config.lmpc_rows:
            for nid, _, _ in row.terms:
                lmpc_nids.add(nid)
        if lmpc_nids:
            lx = [all_nodes[n][0] for n in lmpc_nids if n in all_nodes]
            ly = [all_nodes[n][1] for n in lmpc_nids if n in all_nodes]
            lz = [all_nodes[n][2] for n in lmpc_nids if n in all_nodes]
            ax.scatter(lx, ly, lz, marker="*", c="gold", s=60, zorder=6,
                       edgecolors="darkorange", linewidths=0.5,
                       label=f"LMPC nodes ({len(lmpc_nids)})")

        # DISP nodes — orange downward triangles
        disp_nids: set[int] = {nid for nid, _ in config.disp_bcs}
        if disp_nids:
            dx = [all_nodes[n][0] for n in disp_nids if n in all_nodes]
            dy = [all_nodes[n][1] for n in disp_nids if n in all_nodes]
            dz = [all_nodes[n][2] for n in disp_nids if n in all_nodes]
            ax.scatter(dx, dy, dz, marker="v", c="darkorange", s=50, zorder=7,
                       label=f"DISP BCs ({len(disp_nids)})")

        # Force nodes — lime circles
        force_nids: set[int] = {nid for nid, *_ in config.force_bcs}
        if force_nids:
            fx = [all_nodes[n][0] for n in force_nids if n in all_nodes]
            fy = [all_nodes[n][1] for n in force_nids if n in all_nodes]
            fz = [all_nodes[n][2] for n in force_nids if n in all_nodes]
            ax.scatter(fx, fy, fz, marker="o", c="limegreen", s=60, zorder=8,
                       edgecolors="darkgreen", linewidths=0.8,
                       label=f"Applied forces ({len(force_nids)})")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_folds(ax, folds, color, label, lw=1.0):
    first = True
    for p1, p2, *_ in folds:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw,
                label=label if first else None)
        first = False


def _draw_bounds(ax, bounds, color, label, lw=1.0):
    first = True
    for p1, p2 in bounds:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw, linestyle="--",
                label=label if first else None)
        first = False
