"""
Generate a Miura-Ori crease pattern CSV for the simple_chute mesh.

The simple chute membrane is flat at y=0, spanning:
  x: -0.8 to 0.8  (width = 1.6 m)
  z:  0.0 to 16.0 (height = 16 m)

Run with:
    python generate_creases.py

Outputs:
    simple_chute_creases.csv
"""

import csv

# ── Mesh extents ──────────────────────────────────────────────────────────────
X0, XF = -0.8, 0.8
Z0, ZF = 0.0, 16.0
Y = 0.0                 # membrane is flat at y = 0

# ── Miura-Ori parameters ──────────────────────────────────────────────────────
N_ROWS   = 8            # number of horizontal divisions (→ N_ROWS-1 interior folds)
N_COLS   = 4            # number of column divisions    (→ N_COLS-1 interior fold lines)
OFFSET   = 0.05         # Miura zigzag horizontal offset (m)
ANGLE    = 2.5          # fold target angle (radians); +mountain, -valley

# ─────────────────────────────────────────────────────────────────────────────

rows = []   # each entry: (x1,y1,z1, x2,y2,z2, angle, type)

dz = (ZF - Z0) / N_ROWS
dx = (XF - X0) / N_COLS

# ── Horizontal fold lines (mountain/valley alternating row by row) ─────────
for j in range(1, N_ROWS):
    z = Z0 + j * dz
    angle = ANGLE if j % 2 == 1 else -ANGLE
    rows.append((X0, Y, z,  XF, Y, z,  angle, 'C'))

# ── Diagonal fold lines (zigzag per column, mountain/valley by column) ────
for i in range(1, N_COLS):
    x_center = X0 + i * dx
    angle = ANGLE if i % 2 == 1 else -ANGLE
    for j in range(N_ROWS):
        z1 = Z0 + j * dz
        z2 = Z0 + (j + 1) * dz
        # alternate x offset per row for the Miura characteristic shape
        if j % 2 == 0:
            x1, x2 = x_center - OFFSET, x_center + OFFSET
        else:
            x1, x2 = x_center + OFFSET, x_center - OFFSET
        rows.append((x1, Y, z1,  x2, Y, z2,  angle, 'C'))

# ── Boundary edges (perimeter of the crease region, type = 'B') ───────────
# Each edge is one line segment; angle is ignored for boundaries.
rows.append((X0, Y, Z0,  XF, Y, Z0,  0.0, 'B'))   # bottom
rows.append((X0, Y, ZF,  XF, Y, ZF,  0.0, 'B'))   # top
rows.append((X0, Y, Z0,  X0, Y, ZF,  0.0, 'B'))   # left
rows.append((XF, Y, Z0,  XF, Y, ZF,  0.0, 'B'))   # right

# ── Write CSV ─────────────────────────────────────────────────────────────────
output = "simple_chute_creases.csv"
with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x1", "y1", "z1", "x2", "y2", "z2", "angle", "type"])
    writer.writerows(rows)

n_crease   = sum(1 for r in rows if r[7] == 'C')
n_mountain = sum(1 for r in rows if r[7] == 'C' and r[6] > 0)
n_valley   = sum(1 for r in rows if r[7] == 'C' and r[6] < 0)
n_boundary = sum(1 for r in rows if r[7] == 'B')

print(f"Wrote {output}")
print(f"  {n_crease} crease folds  ({n_mountain} mountain, {n_valley} valley)")
print(f"  {n_boundary} boundary edges")
