"""
Convert Alexandra's old-format Miura-Ori CSVs to the AeroOrigami crease format.

Alexandra's files store data column-major (3 rows × N cols) in separate files
for nodes, edges, angles, and boundary loops.  This script transposes and
merges them into two single-file, row-major CSVs.

Run from this directory:
    python convert_alexandra_creases.py

Outputs (written next to this script):
    dgb_disk_creases.csv   — 1420 fold lines + 200 boundary edges for the disk
    dgb_band_creases.csv   —  500 fold lines + 200 boundary edges for the band
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent
SRC  = HERE / "alexandra_csv_miura"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv_float(filepath: Path) -> list[list[float]]:
    """Read a headerless CSV; return list of rows as float lists."""
    with open(filepath, newline="") as f:
        return [[float(v) for v in row] for row in csv.reader(f) if row]


def _points_from_matrix(mat: list[list[float]]) -> list[tuple[float, float, float]]:
    """
    Alexandra's node matrices are 3 rows × N cols (row 0 = x, 1 = y, 2 = z).
    Return a list of N (x, y, z) tuples.
    """
    xs, ys, zs = mat[0], mat[1], mat[2]
    return [(xs[i], ys[i], zs[i]) for i in range(len(xs))]


def _boundary_segments(pts: list[tuple]) -> list[tuple[tuple, tuple]]:
    """Connect N ordered boundary points into N closed-loop segments."""
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n]) for i in range(n)]


def _write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "y1", "z1", "x2", "y2", "z2", "angle", "type"])
        writer.writerows(rows)


def _summarise(name: str, rows: list) -> None:
    n_m = sum(1 for r in rows if r[7] == "C" and r[6] >= 0)
    n_v = sum(1 for r in rows if r[7] == "C" and r[6] <  0)
    n_b = sum(1 for r in rows if r[7] == "B")
    print(f"  {name}: {n_m} mountain, {n_v} valley, {n_b} boundary")


# ── Disk ──────────────────────────────────────────────────────────────────────

def convert_disk(out: Path) -> None:
    """Convert disk Miura-Ori files (580 nodes, 1420 edges, 2 boundary loops)."""
    pts    = _points_from_matrix(_read_csv_float(SRC / "miura_N40_nodes.csv"))
    edges  = _read_csv_float(SRC / "miura_N40_edges.csv")   # 2 × 1420
    angles = _read_csv_float(SRC / "miura_N40_angles.csv")  # 1 × 1420

    rows = []

    # Fold lines (1-based node indices → 0-based lookup).
    # NaN angles mark panel-boundary edges with no fold target — treated as type 'B'.
    import math
    for i, angle in enumerate(angles[0]):
        n1 = int(edges[0][i]) - 1
        n2 = int(edges[1][i]) - 1
        p1, p2 = pts[n1], pts[n2]
        if math.isnan(angle):
            rows.append([*p1, *p2, 0.0, "B"])
        else:
            rows.append([*p1, *p2, angle, "C"])

    # Outer disk boundary loop (120 pts → 120 closed segments)
    edge_pts = _points_from_matrix(_read_csv_float(SRC / "miura_N40_edge.csv"))
    for p1, p2 in _boundary_segments(edge_pts):
        rows.append([*p1, *p2, 0.0, "B"])

    # Inner vent boundary loop (80 pts → 80 closed segments)
    vent_pts = _points_from_matrix(_read_csv_float(SRC / "miura_N40_vent.csv"))
    for p1, p2 in _boundary_segments(vent_pts):
        rows.append([*p1, *p2, 0.0, "B"])

    _write_csv(out, rows)
    _summarise(out.name, rows)


# ── Band ──────────────────────────────────────────────────────────────────────

def convert_band(out: Path) -> None:
    """Convert band Miura-Ori files (300 nodes, 500 edges, 2 boundary loops)."""
    pts    = _points_from_matrix(_read_csv_float(SRC / "miura_N40_nodes_band.csv"))
    edges  = _read_csv_float(SRC / "miura_N40_edges_band.csv")   # 2 × 500
    angles = _read_csv_float(SRC / "miura_N40_angles_band.csv")  # 1 × 500

    rows = []

    # Fold lines (NaN angles → panel-boundary edges, type 'B')
    import math
    for i, angle in enumerate(angles[0]):
        n1 = int(edges[0][i]) - 1
        n2 = int(edges[1][i]) - 1
        p1, p2 = pts[n1], pts[n2]
        if math.isnan(angle):
            rows.append([*p1, *p2, 0.0, "B"])
        else:
            rows.append([*p1, *p2, angle, "C"])

    # Bottom boundary loop (80 pts → 80 closed segments)
    bottom_pts = _points_from_matrix(_read_csv_float(SRC / "miura_N40_bottom_band.csv"))
    for p1, p2 in _boundary_segments(bottom_pts):
        rows.append([*p1, *p2, 0.0, "B"])

    # Top boundary loop (120 pts → 120 closed segments)
    top_pts = _points_from_matrix(_read_csv_float(SRC / "miura_N40_top_band.csv"))
    for p1, p2 in _boundary_segments(top_pts):
        rows.append([*p1, *p2, 0.0, "B"])

    _write_csv(out, rows)
    _summarise(out.name, rows)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Converting Alexandra's CSVs...")
    convert_disk(HERE / "dgb_disk_creases.csv")
    convert_band(HERE / "dgb_band_creases.csv")
    print("Done.")
