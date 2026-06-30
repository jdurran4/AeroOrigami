"""
Generate a square simple-chute FEM mesh for AeroOrigami.

Canopy  : flat square at y=0, x ∈ [-L/2, L/2], z ∈ [0, L].
Cables  : 4 suspension lines (type-6 bars) from each corner to a
          single confluence node directly below the canopy centre.

Run with:
    python generate_mesh.py

Outputs:
    simple_chute_mesh.fem
"""

# ── Parameters ────────────────────────────────────────────────────────────────
L          = 1.6    # side length (m)
N          = 32     # divisions per side → (N+1)² canopy nodes, 0.05 m spacing
RIG_DEPTH  = 1.6    # depth of confluence node below canopy (m)
OUTPUT     = "simple_chute_mesh.fem"
SHELL_TYPE = 129    # AERO-S 3-node membrane element
CABLE_TYPE = 6      # AERO-S 2-node beam/bar element

# ─────────────────────────────────────────────────────────────────────────────
x0, xf = -L / 2, L / 2
z0, zf =  0.0,   L
dx = L / N
dz = L / N

# ── Canopy nodes ──────────────────────────────────────────────────────────────
# Node ID = j*(N+1) + i + 1  (i=x index, j=z index, both 0-based)
nodes = {}  # nid → (x, y, z)
for j in range(N + 1):
    for i in range(N + 1):
        nid = j * (N + 1) + i + 1
        nodes[nid] = (x0 + i * dx, 0.0, z0 + j * dz)

# ── Confluence node ───────────────────────────────────────────────────────────
confluence_nid = (N + 1) ** 2 + 1
nodes[confluence_nid] = (0.0, -RIG_DEPTH, (z0 + zf) / 2)

# ── Canopy shell elements (two triangles per cell) ────────────────────────────
elements = []  # (etype, [node_ids])
for j in range(N):
    for i in range(N):
        bl = j * (N + 1) + i + 1
        br = bl + 1
        tl = (j + 1) * (N + 1) + i + 1
        tr = tl + 1
        elements.append((SHELL_TYPE, [bl, br, tr]))
        elements.append((SHELL_TYPE, [bl, tr, tl]))

# ── Suspension cable elements (one bar from each corner to confluence) ─────────
corners = [
    1,          # (-L/2, 0, 0)    bottom-left
    N + 1,      # ( L/2, 0, 0)    bottom-right
    N * (N+1) + 1,        # (-L/2, 0, L)    top-left
    (N + 1) ** 2,         # ( L/2, 0, L)    top-right
]
for c in corners:
    elements.append((CABLE_TYPE, [c, confluence_nid]))

# ── Write .fem ────────────────────────────────────────────────────────────────
with open(OUTPUT, "w") as f:
    f.write("NODES\n")
    for nid in sorted(nodes):
        x, y, z = nodes[nid]
        f.write(f"{nid}   {x}  {y}  {z} \n")
    f.write("*\n")

    f.write("TOPOLOGY\n")
    for eid, (etype, nids) in enumerate(elements, start=1):
        f.write(f"{eid}   {etype}  {'  '.join(str(n) for n in nids)} \n")
    f.write("*\n")

n_shell = sum(1 for et, _ in elements if et == SHELL_TYPE)
n_cable = sum(1 for et, _ in elements if et == CABLE_TYPE)
print(f"Wrote {OUTPUT}")
print(f"  {len(nodes)} nodes  ({(N+1)**2} canopy + 1 confluence)")
print(f"  {n_shell} shell elements  (type {SHELL_TYPE})")
print(f"  {n_cable} cable elements  (type {CABLE_TYPE})")
print(f"  Canopy: x=[{x0}, {xf}]  z=[{z0}, {zf}]  spacing={dx:.4f} m")
print(f"  Confluence: {nodes[confluence_nid]}")
