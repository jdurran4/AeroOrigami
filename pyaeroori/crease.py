"""
crease.py — Step 2 of the AeroOrigami pipeline.

Loads one or more crease pattern CSV files and returns a CreasePattern.

CSV format (one row per line segment):
    x1, y1, z1, x2, y2, z2, angle, type[, start_time[, end_time]]

    angle      — target fold angle in radians
                 positive → mountain fold
                 negative → valley fold
                 ignored  → if type is 'B'
    type       — 'C' (crease fold) or 'B' (boundary edge)
    start_time — actuator ramp start time (default 0.0)
    end_time   — actuator ramp end time   (default None → use build_surrogate's
                 actuator_ramp_time parameter)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# 3D point as a plain tuple
Point3D = tuple[float, float, float]

# Crease segment: (p1, p2, angle, start_time, end_time)
# start_time and end_time are None when not specified in the CSV.
CreaseSeg = tuple[Point3D, Point3D, float, float | None, float | None]


@dataclass
class CreasePattern:
    """
    Fold line segments and boundary edges loaded from crease CSV files.

    mountain : list of (p1, p2, angle, start_time, end_time) for mountain folds
    valley   : list of (p1, p2, angle, start_time, end_time) for valley folds
    boundary : list of (p1, p2) for boundary edges (no folding)

    start_time / end_time are None when not specified in the CSV; build_surrogate
    fills in the global actuator_ramp_time default.
    """
    mountain: list[CreaseSeg]          = field(default_factory=list)
    valley:   list[CreaseSeg]          = field(default_factory=list)
    boundary: list[tuple[Point3D, Point3D]] = field(default_factory=list)

    @property
    def all_folds(self) -> list[tuple[Point3D, Point3D, float]]:
        """Mountain and valley folds combined."""
        return self.mountain + self.valley

    def __repr__(self) -> str:
        return (
            f"CreasePattern("
            f"mountain={len(self.mountain)}, "
            f"valley={len(self.valley)}, "
            f"boundary={len(self.boundary)})"
        )


def load_creases(*filepaths: str | Path) -> CreasePattern:
    """
    Parse one or more crease CSV files and merge into a single CreasePattern.

    Rows from all files are combined in order. This lets you split a crease
    pattern across multiple files (e.g., disk and band for a DGB parachute)
    without any special handling on the library side.

    Parameters
    ----------
    *filepaths : one or more paths to crease CSV files

    Returns
    -------
    CreasePattern

    Raises
    ------
    FileNotFoundError  if any file does not exist
    ValueError         if a file has no valid data rows, or an unknown type value
    """
    mountain: list[tuple[Point3D, Point3D, float]] = []
    valley:   list[tuple[Point3D, Point3D, float]] = []
    boundary: list[tuple[Point3D, Point3D]]        = []

    for filepath in filepaths:
        filepath = Path(filepath)
        _parse_csv(filepath, mountain, valley, boundary)

    if not mountain and not valley and not boundary:
        raise ValueError(
            f"No valid rows found in: {', '.join(str(p) for p in filepaths)}"
        )

    return CreasePattern(mountain=mountain, valley=valley, boundary=boundary)


def _parse_csv(
    filepath: Path,
    mountain: list,
    valley: list,
    boundary: list,
) -> None:
    """Parse one CSV file, appending rows into the provided lists."""
    import csv

    n_rows = 0
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Skip header row and blank lines
            if i == 0 and not row[0].strip().lstrip("-").replace(".", "").isdigit():
                continue
            if not row:
                continue

            if len(row) < 8:
                raise ValueError(
                    f"{filepath}:{i+1} — expected 8 columns "
                    f"(x1,y1,z1,x2,y2,z2,angle,type), got {len(row)}"
                )

            try:
                p1 = (float(row[0]), float(row[1]), float(row[2]))
                p2 = (float(row[3]), float(row[4]), float(row[5]))
                angle = float(row[6])
                kind  = row[7].strip().upper()
                start_t = float(row[8])  if len(row) > 8 else None
                end_t   = float(row[9])  if len(row) > 9 else None
            except ValueError as e:
                raise ValueError(f"{filepath}:{i+1} — could not parse row: {e}") from e

            if kind == "B":
                boundary.append((p1, p2))
            elif kind == "C":
                if angle >= 0:
                    mountain.append((p1, p2, angle, start_t, end_t))
                else:
                    valley.append((p1, p2, angle, start_t, end_t))
            else:
                raise ValueError(
                    f"{filepath}:{i+1} — unknown type '{kind}', expected 'C' or 'B'"
                )

            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"No data rows found in {filepath}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python crease.py <crease.csv> [crease2.csv ...]")
        sys.exit(1)

    cp = load_creases(*sys.argv[1:])
    print(cp)
    if cp.mountain:
        p1, p2, a, *_ = cp.mountain[0]
        print(f"  First mountain: {p1} → {p2}  angle={a:.4f} rad")
    if cp.valley:
        p1, p2, a, *_ = cp.valley[0]
        print(f"  First valley  : {p1} → {p2}  angle={a:.4f} rad")
    if cp.boundary:
        p1, p2 = cp.boundary[0]
        print(f"  First boundary: {p1} → {p2}")
