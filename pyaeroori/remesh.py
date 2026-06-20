"""
remesh.py — Step 3 of the AeroOrigami pipeline.

Two paths, both returning a Mesh with panel_map populated:

Path A — Gmsh remesh (default)
    Segment endpoints become Gmsh points; adjacent panels share curves so
    shared-boundary nodes stitch automatically.  Gmsh adds interior nodes
    along long crease lines, giving driver-joint sites in Step 4.  Works
    for arbitrary geometry including curved (cylindrical, tangent) surfaces.

Path B — crease-as-mesh  (Region use_crease_mesh=True)
    No Gmsh.  The half-edge face polygons from the crease graph become shell
    elements directly.  Every crease-segment endpoint is a mesh node, so
    every crease vertex becomes a driver-joint site — no fold is ever
    under-resolved regardless of geometry.  n-gons with > 4 sides are
    fan-triangulated from vertex 0 (valid for convex panels); all resulting
    triangles share the same panel_id so no joint is placed between them.
    Valid for any crease pattern; works best when panels are 3 or 4 sided.

Shared algorithm (both paths)
------------------------------
1.  T-junction splitting — endpoint of one segment on interior of another.
2.  Snap-and-dedup segment endpoints; build crease graph adjacency.
3.  Auto-detect 2D projection: cylindrical first, then planar, else tangent.
4.  Sort each vertex's neighbours by angle in the 2D frame.
5.  Half-edge traversal: for directed edge (u→v) turn hard-left at v
    (prev CCW neighbour) to recover the face to the left.
6.  Filter: discard negative-signed-area faces (exterior) and outliers.

Path A continues: one addPlaneSurface per interior face (curves shared).
Path B continues: write Mesh elements, fan-triangulate n > 4 faces.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .mesh import Mesh
from .crease import CreasePattern

_SNAP_TOL       = 1e-6
_EXT_AREA_RATIO = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Region:
    """
    One connected surface patch with its own crease pattern and mesh settings.

    Parameters
    ----------
    creases         : fold lines and boundary edges for this region
    mesh_size       : target Gmsh element edge length (metres) — Path A only
    projection      : '\'auto\'' | '\'planar\'' | '\'cylindrical\'' | '\'tangent\''
    name            : label for Gmsh Physical Group / AEROS output
    use_crease_mesh : True → Path B (crease polygons as shell elements, no Gmsh)
    """
    creases:         CreasePattern
    mesh_size:       float = 0.2
    projection:      str   = "auto"
    name:            str   = ""
    use_crease_mesh: bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def remesh(
    mesh: Mesh,
    *regions: Region,
    out_file: str = "",
    show: bool = False,
) -> Mesh:
    """
    Build a coarse mesh aligned to the crease pattern (Step 3).

    Parameters
    ----------
    mesh     : original hi-fi mesh (read-only; not modified)
    *regions : one or more Region objects
    out_file : optional path to write output (.msh for Path A; ignored for B)
    show     : open a viewer after meshing:
               Path A — Gmsh GUI (blocks until closed)
               Path B — matplotlib plot_mesh window

    Returns
    -------
    Mesh with panel_map : dict[element_id → panel_id]
    """
    if not regions:
        raise ValueError("Supply at least one Region.")

    # ── Path B: crease-as-mesh ────────────────────────────────────────────
    if any(r.use_crease_mesh for r in regions):
        if not all(r.use_crease_mesh for r in regions):
            raise ValueError(
                "Cannot mix use_crease_mesh=True and False regions in one "
                "remesh() call.  Use separate calls for each path."
            )
        result = _crease_path(list(regions))
        result.path = "B"
        if show:
            from . import plot as _plot
            _plot.plot_mesh(result, title="Crease Mesh (Step 3)", show=True)
        return result

    # ── Path A: Gmsh remesh ───────────────────────────────────────────────
    try:
        import gmsh
    except ImportError:
        raise ImportError("Path A remesh() requires gmsh:  pip install gmsh")

    if not regions:
        raise ValueError("Supply at least one Region.")

    gmsh.initialize()
    gmsh.model.add("pyaeroori_origami")

    # Shared across regions so boundary nodes are globally deduplicated
    pt_map:   dict[tuple, int]     = {}
    edge_map: dict[frozenset, int] = {}
    dir_map:  dict[tuple, int]     = {}

    def add_pt(xyz) -> int:
        key = _snap(xyz)
        if key not in pt_map:
            pt_map[key] = gmsh.model.occ.addPoint(
                float(xyz[0]), float(xyz[1]), float(xyz[2])
            )
        return pt_map[key]

    def add_line(pa: int, pb: int) -> int:
        key = frozenset((pa, pb))
        if key not in edge_map:
            tag = gmsh.model.occ.addLine(pa, pb)
            edge_map[key]     = tag
            dir_map[(pa, pb)] =  tag
            dir_map[(pb, pa)] = -tag
        return dir_map[(pa, pb)]

    panel_surf_map: dict[int, int]              = {}
    panel_counter:  list[int]                   = [1]
    region_groups:  list[tuple[str, list[int]]] = []

    for region in regions:
        surfs = _build_region(region, add_pt, add_line,
                              panel_surf_map, panel_counter)
        region_groups.append((region.name, surfs))
        print(f"  Region '{region.name}': {len(surfs)} panel surfaces")

    gmsh.model.occ.synchronize()
    for name, surfs in region_groups:
        if surfs:
            pg = gmsh.model.addPhysicalGroup(2, surfs)
            if name:
                gmsh.model.setPhysicalName(2, pg, name)

    lo = min(r.mesh_size for r in regions)
    hi = max(r.mesh_size for r in regions)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lo * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hi * 1.5)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    gmsh.model.mesh.generate(2)

    if out_file:
        gmsh.write(out_file)

    if show:
        gmsh.fltk.run()

    result = _gmsh_to_mesh(panel_surf_map)
    result.path = "A"
    gmsh.finalize()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Shared: panel detection (half-edge traversal)
# ─────────────────────────────────────────────────────────────────────────────

def _find_panels(
    segs:       list[tuple[np.ndarray, np.ndarray]],
    projection: str,
) -> tuple[list[list[int]], dict[int, np.ndarray], dict[int, tuple], str]:
    """
    Shared by both paths.  From a flat list of 3D segments:

      1. Split at T-junctions.
      2. Snap-and-dedup endpoints into local pids (1-based integers).
      3. Detect projection; build 2D coords and angle function.
      4. Half-edge traversal → raw face polygons.
      5. Filter exterior / degenerate faces.

    Returns
    -------
    interior  : list of face polygons (each a list of local pids)
    pid_xyz   : local pid → 3D numpy array
    coords2d  : local pid → 2D (u, v) tuple
    proj      : projection string actually used
    """
    segs = _split_at_junctions(segs)

    snap_local: dict[tuple, int]      = {}
    pid_xyz:    dict[int, np.ndarray] = {}
    adj:        dict[int, set]        = defaultdict(set)

    for p1, p2 in segs:
        k1, k2 = _snap(p1), _snap(p2)
        for k, p in ((k1, p1), (k2, p2)):
            if k not in snap_local:
                lid = len(snap_local) + 1
                snap_local[k] = lid
                pid_xyz[lid]  = p.copy()
        a, b = snap_local[k1], snap_local[k2]
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    if len(pid_xyz) < 3:
        return [], pid_xyz, {}, projection

    pts  = np.array(list(pid_xyz.values()))
    proj = _detect_projection(pts, projection)
    print(f"    projection = {proj}")

    coords2d, angle_fn = _make_angle_fn(pid_xyz, adj, proj)

    sorted_nbrs: dict[int, list[int]] = {
        v: sorted(adj[v], key=lambda n: angle_fn(v, n, coords2d[v], coords2d))
        for v in adj
    }

    visited:   set[tuple[int, int]] = set()
    raw_faces: list[list[int]]      = []
    max_face   = len(pid_xyz) * 2 + 10

    for pid_a in adj:
        for pid_b in adj[pid_a]:
            if (pid_a, pid_b) in visited:
                continue
            face: list[int] = []
            u, v = pid_a, pid_b
            for _ in range(max_face):
                visited.add((u, v))
                face.append(u)
                nbrs = sorted_nbrs.get(v, [])
                if not nbrs:
                    break
                try:
                    idx = nbrs.index(u)
                except ValueError:
                    break
                w = nbrs[(idx - 1) % len(nbrs)]
                u, v = v, w
                if (u, v) == (pid_a, pid_b):
                    break
            if len(face) >= 3:
                raw_faces.append(face)

    interior = _filter_faces(raw_faces, coords2d, pid_xyz, proj)
    print(f"    {len(raw_faces)} raw faces → {len(interior)} interior panels")

    return interior, pid_xyz, coords2d, proj


# ─────────────────────────────────────────────────────────────────────────────
# Path A: Gmsh surface builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_region(
    region:         Region,
    add_pt:         Callable,
    add_line:       Callable,
    panel_surf_map: dict,
    panel_counter:  list,
) -> list[int]:
    """Add all panel surfaces for one region; return list of Gmsh surf tags."""
    import gmsh

    cp   = region.creases
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for p1, p2, *_ in cp.mountain + cp.valley:
        segs.append((np.asarray(p1, float), np.asarray(p2, float)))
    for p1, p2 in cp.boundary:
        segs.append((np.asarray(p1, float), np.asarray(p2, float)))

    if not segs:
        return []

    interior, pid_xyz, _, _ = _find_panels(segs, region.projection)

    if not interior:
        print(f"    WARNING: no interior faces found for region '{region.name}'")
        return []

    surf_tags: list[int] = []
    for face in interior:
        n = len(face)
        # Translate local pids → Gmsh point tags, then build curve loop
        gtags = [add_pt(pid_xyz[lid]) for lid in face]
        lines, ok = [], True
        for i in range(n):
            a, b = gtags[i], gtags[(i + 1) % n]
            if a == b:
                ok = False
                break
            lines.append(add_line(a, b))
        if not ok or len(lines) < 3:
            continue
        try:
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            surf_tags.append(surf)
            panel_surf_map[surf] = panel_counter[0]
            panel_counter[0] += 1
        except Exception:
            continue

    return surf_tags


# ─────────────────────────────────────────────────────────────────────────────
# Path B: crease-as-mesh
# ─────────────────────────────────────────────────────────────────────────────

def _crease_path(regions: list[Region]) -> Mesh:
    """
    Build a Mesh directly from crease face polygons — no Gmsh.

    Element types (Gmsh numbering, consistent with Path A output):
      etype 2 = 3-node triangle
      etype 3 = 4-node quad

    n-gons with n > 4 are fan-triangulated from vertex 0:
      triangles (v0,v1,v2), (v0,v2,v3), …, (v0,v_{n-2},v_{n-1})
    All fan triangles of one polygon share the same panel_id, so Step 4
    will not place a joint between them.

    Regions share a snap map so common boundary nodes stitch automatically.
    """
    snap_to_nid: dict[tuple, int]                      = {}
    node_xyz:    dict[int, tuple[float, float, float]] = {}

    def add_node(xyz: np.ndarray) -> int:
        key = _snap(xyz)
        if key not in snap_to_nid:
            nid = len(snap_to_nid) + 1
            snap_to_nid[key] = nid
            node_xyz[nid]    = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        return snap_to_nid[key]

    elements:  dict[int, tuple[int, list[int]]] = {}
    panel_map: dict[int, int]                   = {}
    eid      = 1
    panel_id = 1

    for region in regions:
        cp   = region.creases
        segs: list[tuple[np.ndarray, np.ndarray]] = []
        for p1, p2, *_ in cp.mountain + cp.valley:
            segs.append((np.asarray(p1, float), np.asarray(p2, float)))
        for p1, p2 in cp.boundary:
            segs.append((np.asarray(p1, float), np.asarray(p2, float)))

        if not segs:
            continue

        interior, pid_xyz, _, _ = _find_panels(segs, region.projection)

        n_before = panel_id - 1
        for face in interior:
            nids = [add_node(pid_xyz[lid]) for lid in face]
            n    = len(nids)

            if n == 3:
                elements[eid]  = (2, nids)
                panel_map[eid] = panel_id
                eid += 1
            elif n == 4:
                elements[eid]  = (3, nids)
                panel_map[eid] = panel_id
                eid += 1
            else:
                # First-vertex fan — valid for convex polygons (typical for origami)
                for i in range(1, n - 1):
                    elements[eid]  = (2, [nids[0], nids[i], nids[i + 1]])
                    panel_map[eid] = panel_id
                    eid += 1

            panel_id += 1

        print(f"  Region '{region.name}': {panel_id - 1 - n_before} panels")

    return Mesh(nodes=node_xyz, elements=elements, panel_map=panel_map)


# ─────────────────────────────────────────────────────────────────────────────
# Projection detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_projection(pts: np.ndarray, mode: str) -> str:
    if mode != "auto":
        return mode

    mean     = pts.mean(axis=0)
    centered = pts - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    eigvals  = s ** 2
    total    = eigvals.sum()

    if total == 0:
        return "planar"

    # Cylindrical check FIRST: points lie at roughly constant radius in the
    # PCA ring plane.  Cylindrical surfaces also have near-zero 3rd eigenvalue
    # so checking planar first would swallow them.
    ring_proj = centered @ Vt[:2].T
    r = np.linalg.norm(ring_proj, axis=1)
    if r.mean() > 0 and r.std() / r.mean() < 0.05:
        return "cylindrical"

    # Planar: third eigenvalue negligible (and constant-radius check failed)
    if eigvals[2] / total < 0.02:
        return "planar"

    return "tangent"


# ─────────────────────────────────────────────────────────────────────────────
# 2D coordinate mapping and angle functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_angle_fn(
    pid_xyz: dict[int, np.ndarray],
    adj:     dict[int, set],
    proj:    str,
) -> tuple[dict[int, tuple], Callable]:
    """
    Return (coords2d, angle_fn) where:
      coords2d  : pid → (u, v) for signed-area calculations
      angle_fn  : (pid_v, pid_n, center_2d, coords2d) → float
    """
    pids = list(pid_xyz.keys())
    pts  = np.array([pid_xyz[p] for p in pids])
    mean = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)

    # ── Planar ──────────────────────────────────────────────────────────
    if proj == "planar":
        plane = Vt[:2]
        proj2 = (pts - mean) @ plane.T
        coords2d = {p: (float(proj2[i, 0]), float(proj2[i, 1]))
                    for i, p in enumerate(pids)}

        def angle_fn(pid_v, pid_n, cv, c2d):
            cn = c2d[pid_n]
            return math.atan2(cn[1] - cv[1], cn[0] - cv[0])

        return coords2d, angle_fn

    # ── Cylindrical ─────────────────────────────────────────────────────
    elif proj == "cylindrical":
        ring_axes  = Vt[:2]
        height_ax  = Vt[2]
        ring_proj  = (pts - mean) @ ring_axes.T
        theta      = np.arctan2(ring_proj[:, 1], ring_proj[:, 0])
        height     = (pts - mean) @ height_ax
        coords2d   = {p: (float(theta[i]), float(height[i]))
                      for i, p in enumerate(pids)}

        def angle_fn(pid_v, pid_n, cv, c2d):
            cn     = c2d[pid_n]
            dtheta = cn[0] - cv[0]
            dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
            dz     = cn[1] - cv[1]
            return math.atan2(dz, dtheta)

        return coords2d, angle_fn

    # ── Tangent (local per-vertex normal) ───────────────────────────────
    else:
        local_normal: dict[int, np.ndarray] = {}
        for v in pid_xyz:
            nbrs   = list(adj.get(v, []))
            pv     = pid_xyz[v]
            normal = np.zeros(3)
            for i in range(len(nbrs)):
                ea    = pid_xyz[nbrs[i]]               - pv
                eb    = pid_xyz[nbrs[(i + 1) % len(nbrs)]] - pv
                cross = np.cross(ea, eb)
                nrm   = np.linalg.norm(cross)
                if nrm > 1e-12:
                    normal += cross / nrm
            nrm = np.linalg.norm(normal)
            local_normal[v] = normal / nrm if nrm > 1e-12 else Vt[2]

        plane    = Vt[:2]
        proj2    = (pts - mean) @ plane.T
        coords2d = {p: (float(proj2[i, 0]), float(proj2[i, 1]))
                    for i, p in enumerate(pids)}

        ref_dirs: dict[int, np.ndarray] = {}
        for v, nrm in local_normal.items():
            ref = np.array([1.0, 0.0, 0.0])
            ref -= np.dot(ref, nrm) * nrm
            if np.linalg.norm(ref) < 0.1:
                ref = np.array([0.0, 1.0, 0.0])
                ref -= np.dot(ref, nrm) * nrm
            r_norm = np.linalg.norm(ref)
            ref_dirs[v] = ref / r_norm if r_norm > 1e-12 else ref

        def angle_fn(pid_v, pid_n, cv, c2d,
                     _ln=local_normal, _rd=ref_dirs, _pxyz=pid_xyz):
            nrm  = _ln[pid_v]
            ref  = _rd[pid_v]
            d    = _pxyz[pid_n] - _pxyz[pid_v]
            d   -= np.dot(d, nrm) * nrm
            dlen = np.linalg.norm(d)
            if dlen < 1e-12:
                return 0.0
            d /= dlen
            cross = np.cross(ref, d)
            return math.atan2(float(np.dot(cross, nrm)), float(np.dot(ref, d)))

        return coords2d, angle_fn


# ─────────────────────────────────────────────────────────────────────────────
# Face filtering
# ─────────────────────────────────────────────────────────────────────────────

def _signed_area_2d(face: list[int], coords2d: dict) -> float:
    """Shoelace signed area.  Positive = CCW = interior face."""
    area = 0.0
    n    = len(face)
    for i in range(n):
        u = coords2d[face[i]]
        v = coords2d[face[(i + 1) % n]]
        area += u[0] * v[1] - v[0] * u[1]
    return area * 0.5


def _area_3d(face: list[int], pid_xyz: dict) -> float:
    """Triangulated 3D area (always positive)."""
    if len(face) < 3:
        return 0.0
    p0   = pid_xyz[face[0]]
    area = 0.0
    for i in range(1, len(face) - 1):
        a = pid_xyz[face[i]]     - p0
        b = pid_xyz[face[i + 1]] - p0
        area += float(np.linalg.norm(np.cross(a, b)))
    return area * 0.5


def _filter_faces(
    faces:    list[list[int]],
    coords2d: dict,
    pid_xyz:  dict,
    proj:     str,
) -> list[list[int]]:
    """
    Retain only interior (panel) faces.

    1. Discard degenerate faces (< 3 distinct vertices).
    2. Discard faces with non-positive signed 2D area (exterior / CW).
    3. For cylindrical projections, discard cap faces (constant height).
    4. Discard faces whose 3D area > _EXT_AREA_RATIO × median (residual exterior).
    """
    keep: list[list[int]] = []

    for face in faces:
        if len(set(face)) < 3:
            continue
        if _signed_area_2d(face, coords2d) <= 0:
            continue
        if proj == "cylindrical":
            hs = [coords2d[p][1] for p in face]
            if max(hs) - min(hs) < 1e-6:
                continue
        keep.append(face)

    if not keep:
        return keep

    areas = [_area_3d(f, pid_xyz) for f in keep]
    med   = float(np.median(areas))
    if med > 0:
        keep = [f for f, a in zip(keep, areas) if a <= _EXT_AREA_RATIO * med]

    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Gmsh → Mesh conversion (Path A only)
# ─────────────────────────────────────────────────────────────────────────────

def _gmsh_to_mesh(panel_surf_map: dict[int, int]) -> Mesh:
    import gmsh

    raw_tags, raw_coords, _ = gmsh.model.mesh.getNodes()
    raw_tags   = np.array(raw_tags, dtype=int)
    raw_coords = np.array(raw_coords, dtype=float).reshape(-1, 3)
    tag_to_id  = {int(t): i + 1 for i, t in enumerate(raw_tags)}

    nodes: dict[int, tuple] = {
        tag_to_id[int(t)]: (float(raw_coords[i, 0]),
                             float(raw_coords[i, 1]),
                             float(raw_coords[i, 2]))
        for i, t in enumerate(raw_tags)
    }

    elements:  dict[int, tuple] = {}
    panel_map: dict[int, int]   = {}
    eid = 1

    for surf_tag, panel_id in panel_surf_map.items():
        try:
            etypes, etags_list, entags_list = gmsh.model.mesh.getElements(
                dim=2, tag=surf_tag
            )
        except Exception:
            continue
        for etype, etags, entags in zip(etypes, etags_list, entags_list):
            n_per = len(entags) // len(etags)
            for j in range(len(etags)):
                conn = [tag_to_id[int(entags[j * n_per + k])]
                        for k in range(n_per)]
                elements[eid]  = (int(etype), conn)
                panel_map[eid] = panel_id
                eid += 1

    return Mesh(nodes=nodes, elements=elements, panel_map=panel_map)


# ─────────────────────────────────────────────────────────────────────────────
# T-junction splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_at_junctions(
    segs: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Detect T-junctions and split the through-segment at each one.

    A T-junction exists when an endpoint of one segment lies in the strict
    interior of another.  Without splitting the crease graph has a dangling
    node that the half-edge traversal cannot handle correctly.
    """
    snap_to_xyz: dict[tuple, np.ndarray] = {}
    for p1, p2 in segs:
        snap_to_xyz.setdefault(_snap(p1), p1.copy())
        snap_to_xyz.setdefault(_snap(p2), p2.copy())

    all_pts = np.array(list(snap_to_xyz.values()))   # (P, 3)
    result:  list[tuple[np.ndarray, np.ndarray]] = []

    for p1, p2 in segs:
        v        = p2 - p1
        seg_len2 = float(np.dot(v, v))
        if seg_len2 < 1e-24:
            continue

        w    = all_pts - p1                               # (P, 3)
        t    = (w @ v) / seg_len2                         # (P,)
        foot = p1 + np.outer(t, v)                        # (P, 3)
        dist = np.linalg.norm(all_pts - foot, axis=1)     # (P,)

        interior_mask = (
            (t > _SNAP_TOL) & (t < 1 - _SNAP_TOL) & (dist < _SNAP_TOL * 10)
        )
        interior_idx = np.where(interior_mask)[0]

        if len(interior_idx) == 0:
            result.append((p1.copy(), p2.copy()))
            continue

        t_interior = t[interior_idx]
        order      = np.argsort(t_interior)
        chain      = [p1] + [all_pts[interior_idx[i]] for i in order] + [p2]
        for i in range(len(chain) - 1):
            result.append((chain[i].copy(), chain[i + 1].copy()))

    return result


def _snap(xyz) -> tuple:
    return (
        round(float(xyz[0]) / _SNAP_TOL) * _SNAP_TOL,
        round(float(xyz[1]) / _SNAP_TOL) * _SNAP_TOL,
        round(float(xyz[2]) / _SNAP_TOL) * _SNAP_TOL,
    )
