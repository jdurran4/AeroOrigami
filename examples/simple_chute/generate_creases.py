"""
Generate a Miura-Ori crease pattern CSV for the simple_chute mesh.

The simple chute membrane is flat at y=0, spanning:
  x: -0.8 to 0.8  (width = 1.6 m)
  z:  0.0 to  1.6 (height = 1.6 m — square)

Run with:
    python generate_creases.py

Outputs:
    simple_chute_creases.csv
"""

import csv
import math

# ── Mesh extents ──────────────────────────────────────────────────────────────
X0, XF = -0.8, 0.8
Z0, ZF = 0.0, 1.6
Y = 0.0

# ── Miura-Ori parameters ──────────────────────────────────────────────────────
N      = 8              # grid divisions (N×N cells → N-1 interior crease lines each way)
OFFSET = 0.05           # Miura zigzag horizontal offset (m)
ANGLE  = 3.14           # fold target angle (rad); mountains get +ANGLE, valleys get -ANGLE
PLOT   = True

# ─────────────────────────────────────────────────────────────────────────────

_SNAP_TOL = 1e-8

def _snap(pt):
    """Round 2-D point (x, z) to avoid floating-point hash mismatches."""
    return (round(pt[0] / _SNAP_TOL) * _SNAP_TOL,
            round(pt[1] / _SNAP_TOL) * _SNAP_TOL)


def miura_crease_generator(x0, z0, xf, zf, n, offset=0.0, angle=3.14, plot=True):
    dx = (xf - x0) / n
    dz = (zf - z0) / n

    mountain_segs = []
    valley_segs   = []
    all_crease_segs = []
    all_pts = set()

    # ── Horizontal crease lines (interior rows only) ──────────────────────────
    for j in range(1, n):
        z = z0 + j * dz
        row_offset = -offset if j % 2 == 0 else offset
        for i in range(n):
            x1 = x0 + i * dx
            x2 = x1 + dx
            # shift interior junction points; endpoints stay on boundary
            if i != 0:
                x1 += row_offset
            if i != n - 1:
                x2 += row_offset
            else:
                x2 = xf
            seg = (_snap((x1, z)), _snap((x2, z)))
            if (i + j) % 2 == 0:
                valley_segs.append(seg)
            else:
                mountain_segs.append(seg)
            all_crease_segs.append(seg)
            all_pts.update(seg)

    # ── Vertical (diagonal) crease lines (interior columns only) ─────────────
    for i in range(1, n):
        x_base = x0 + i * dx
        for j in range(n):
            if j % 2 == 0:
                x1, x2 = x_base - offset, x_base + offset
            else:
                x1, x2 = x_base + offset, x_base - offset
            z1 = z0 + j * dz
            z2 = z0 + (j + 1) * dz
            seg = (_snap((x1, z1)), _snap((x2, z2)))
            if i % 2 == 0:
                mountain_segs.append(seg)
            else:
                valley_segs.append(seg)
            all_crease_segs.append(seg)
            all_pts.update(seg)

    # ── Boundary edges, split at crease endpoints ─────────────────────────────
    bdy = {"bottom": [(x0, z0), (xf, z0)],
           "top":    [(x0, zf), (xf, zf)],
           "left":   [(x0, z0), (x0, zf)],
           "right":  [(xf, z0), (xf, zf)]}

    for p1, p2 in all_crease_segs:
        for px, pz in (p1, p2):
            if abs(pz - z0) < 1e-10: bdy["bottom"].append((px, z0))
            if abs(pz - zf) < 1e-10: bdy["top"].append((px, zf))
            if abs(px - x0) < 1e-10: bdy["left"].append((x0, pz))
            if abs(px - xf) < 1e-10: bdy["right"].append((xf, pz))

    edge_segs = []
    for key, sort_key in [("bottom", lambda p: p[0]), ("top", lambda p: p[0]),
                           ("left",   lambda p: p[1]), ("right", lambda p: p[1])]:
        pts = sorted(set(bdy[key]), key=sort_key)
        edge_segs += [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
    for p1, p2 in edge_segs:
        all_pts.update([_snap(p1), _snap(p2)])

    # ── Optional plot ─────────────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 12))
        for p1, p2 in valley_segs:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-o', ms=3)
        for p1, p2 in mountain_segs:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-o', ms=3)
        for p1, p2 in edge_segs:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2)
        ax.set_aspect('equal')
        ax.set_title(f"Miura-Ori  n={n}  offset={offset}  angle=±{angle}")
        ax.set_xlabel("x"); ax.set_ylabel("z")
        plt.tight_layout()
        plt.show()

    return mountain_segs, valley_segs, edge_segs


# ── Run ───────────────────────────────────────────────────────────────────────
mountain_segs, valley_segs, edge_segs = miura_crease_generator(
    X0, Z0, XF, ZF, N, offset=OFFSET, angle=ANGLE, plot=PLOT
)

output = "simple_chute_creases.csv"
rows = []
for p1, p2 in mountain_segs:
    rows.append((p1[0], Y, p1[1],  p2[0], Y, p2[1],  ANGLE, 'C'))
for p1, p2 in valley_segs:
    rows.append((p1[0], Y, p1[1],  p2[0], Y, p2[1], -ANGLE, 'C'))
for p1, p2 in edge_segs:
    rows.append((p1[0], Y, p1[1],  p2[0], Y, p2[1],  0.0,   'B'))

with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x1", "y1", "z1", "x2", "y2", "z2", "angle", "type"])
    writer.writerows(rows)

n_mountain = len(mountain_segs)
n_valley   = len(valley_segs)
n_boundary = len(edge_segs)
print(f"Wrote {output}")
print(f"  {n_mountain + n_valley} crease folds  ({n_mountain} mountain, {n_valley} valley)")
print(f"  {n_boundary} boundary edges")
