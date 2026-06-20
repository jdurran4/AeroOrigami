"""
writer.py — Step 6 of the AeroOrigami pipeline.

write_aeros(surrogate, output_dir) writes the AERO-S include files needed
to run the origami fold simulation:

  mesh_modified.include  — NODES + TOPOLOGY (shells + joints + cables) + ATTRIBUTES
                           + MATERIAL for spherical joints
  ACTUATORS.include      — MATERIAL section: one CONMAT RAMP line per revolute
  EFRAMES.include        — EFRAMES section: local frame per revolute joint

When a ModelConfig (from add_physics) is also passed, additional files are written:

  DISP.include           — Dirichlet BCs (fixed DOFs)
  LMPC.include           — inequality constraints (min_z, min_radius, custom)
  FORCE.include          — point force loads
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .surrogate import Surrogate, JointInfo
    from .physics import ModelConfig

# Attribute IDs
_SHELL_ATTR     = 1    # all shell elements share one attribute (material TBD)
_SPH_ATTR       = 2    # all spherical joints share one attribute
_REV_ATTR_START = 3    # revolute joints start here, one per joint
_CABLE_ATTR     = 10   # cable bar elements (from physics.ModelConfig)

# AERO-S element types (TOPOLOGY section)
#   203  — tension-only spring (2 nodes) — used for cables during fold
#   15   — 3-node triangular AQR shell (6 dof/node)
#   1515 — 4-node quadrilateral AQR shell (6 dof/node)
#   120  — spherical joint
#   126  — revolute driver joint
def _aeros_etype(nids: list[int]) -> int:
    """Map element to its AERO-S type by node count."""
    n = len(nids)
    if n == 2:
        return 203     # tension-only spring (cable)
    if n == 3:
        return 15      # triangular AQR shell
    if n == 4:
        return 1515    # quadrilateral AQR shell
    raise ValueError(f"No AERO-S shell type for {n}-node element")


def write_aeros(
    surrogate:   "Surrogate",
    output_dir:  str | Path,
    config:      "ModelConfig | None" = None,
    beta_factor: float = 0.1,
) -> dict[str, Path]:
    """
    Write AERO-S include files for the fold surrogate.

    Parameters
    ----------
    surrogate   : Surrogate returned by build_surrogate()
    output_dir  : directory to write files into (created if needed)
    config      : ModelConfig from add_physics() — if provided, also writes
                  DISP.include, LMPC.include, FORCE.include and merges cable
                  elements into mesh_modified.include
    beta_factor : revolute joint beta = penalty_stiffness * beta_factor

    Returns
    -------
    dict mapping file labels to the Path written
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    rev_joints = surrogate.revolute_joints
    sph_joints = surrogate.spherical_joints
    beta = surrogate.penalty_stiffness * beta_factor

    # Assign per-revolute attr IDs (attr 3, 4, 5, ...)
    rev_attr: dict[int, int] = {
        j.eid: _REV_ATTR_START + i for i, j in enumerate(rev_joints)
    }

    mesh_path  = out / "mesh_modified.include"
    act_path   = out / "ACTUATORS.include"
    efr_path   = out / "EFRAMES.include"

    _write_mesh(surrogate, mesh_path, sph_joints, rev_joints, rev_attr, config)
    _write_actuators(rev_joints, rev_attr, beta, act_path)
    _write_eframes(rev_joints, efr_path)

    written = {"mesh": mesh_path, "actuators": act_path, "eframes": efr_path}
    print(f"  Wrote {mesh_path.name}")
    print(f"  Wrote {act_path.name}  ({len(rev_joints)} revolute material entries)")
    print(f"  Wrote {efr_path.name}  ({len(rev_joints)} EFRAMES entries)")

    if config is not None:
        if config.disp_bcs:
            p = out / "DISP.include"
            _write_disp(config.disp_bcs, p)
            written["disp"] = p
            print(f"  Wrote {p.name}  ({len(config.disp_bcs)} BC entries)")

        if config.lmpc_rows:
            p = out / "LMPC.include"
            _write_lmpc(config.lmpc_rows, p)
            written["lmpc"] = p
            print(f"  Wrote {p.name}  ({len(config.lmpc_rows)} constraints)")

        if config.force_bcs:
            p = out / "FORCE.include"
            _write_force(config.force_bcs, p)
            written["force"] = p
            print(f"  Wrote {p.name}  ({len(config.force_bcs)} force entries)")

    return written


# ── File writers ───────────────────────────────────────────────────────────────

def _write_mesh(
    surrogate:  "Surrogate",
    path:       Path,
    sph_joints: list["JointInfo"],
    rev_joints: list["JointInfo"],
    rev_attr:   dict[int, int],
    config:     "ModelConfig | None" = None,
) -> None:
    """Write mesh_modified.include: NODES, TOPOLOGY, ATTRIBUTES, MATERIAL."""
    cable_nodes    = config.cable_nodes    if config else {}
    cable_elements = config.cable_elements if config else []

    with open(path, "w") as f:

        # ── NODES ────────────────────────────────────────────────────────────
        f.write("NODES\n")
        for nid in sorted(surrogate.nodes):
            x, y, z = surrogate.nodes[nid]
            f.write(f"  {nid}  {x:.10e}  {y:.10e}  {z:.10e}\n")
        for nid in sorted(cable_nodes):
            x, y, z = cable_nodes[nid]
            f.write(f"  {nid}  {x:.10e}  {y:.10e}  {z:.10e}\n")
        f.write("*\n")

        # ── TOPOLOGY ─────────────────────────────────────────────────────────
        f.write("TOPOLOGY\n")
        for eid in sorted(surrogate.elements):
            _, nids = surrogate.elements[eid]
            aetype   = _aeros_etype(nids)
            node_str = "  ".join(str(n) for n in nids)
            f.write(f"  {eid}  {aetype}  {node_str}\n")
        for j in sph_joints:
            f.write(f"  {j.eid}  120  {j.node_a}  {j.node_b}\n")
        for j in rev_joints:
            f.write(f"  {j.eid}  126  {j.node_a}  {j.node_b}\n")
        for eid, _, nids in cable_elements:
            aetype   = _aeros_etype(nids)
            node_str = "  ".join(str(n) for n in nids)
            f.write(f"  {eid}  {aetype}  {node_str}\n")
        f.write("*\n")

        # ── ATTRIBUTES ───────────────────────────────────────────────────────
        f.write("ATTRIBUTES\n")
        for eid in sorted(surrogate.elements):
            f.write(f"  {eid}  {_SHELL_ATTR}\n")
        for j in sph_joints:
            f.write(f"  {j.eid}  {_SPH_ATTR}\n")
        for j in rev_joints:
            f.write(f"  {j.eid}  {rev_attr[j.eid]}\n")
        for eid, _, _ in cable_elements:
            f.write(f"  {eid}  {_CABLE_ATTR}\n")
        f.write("*\n")

        # ── MATERIAL: spherical joint shared definition ───────────────────────
        f.write("MATERIAL\n")
        f.write(f"  {_SPH_ATTR}  CONMAT penalty {surrogate.penalty_stiffness:.6e}\n")
        f.write("*\n")


def _write_actuators(
    rev_joints: list["JointInfo"],
    rev_attr:   dict[int, int],
    beta:       float,
    path:       Path,
) -> None:
    """Write ACTUATORS.include: MATERIAL section for revolute driver joints."""
    with open(path, "w") as f:
        f.write("MATERIAL\n")
        for j in rev_joints:
            mid = rev_attr[j.eid]
            f.write(
                f"  {mid}  CONMAT penalty {beta:.6e}"
                f"  RAMP  {j.target_angle:.6f}  0.0"
                f"  {j.start_time}  {j.end_time}\n"
            )
        f.write("*\n")


def _write_disp(disp_bcs: list, path: Path) -> None:
    """Write DISP.include: Dirichlet BC entries, one (node, dof) per line."""
    with open(path, "w") as f:
        f.write("DISP\n")
        for nid, dofs in disp_bcs:
            for dof in dofs:
                f.write(f"  {nid}  {dof}  0.0\n")
        f.write("*\n")


def _write_lmpc(lmpc_rows: list, path: Path) -> None:
    """Write LMPC.include: one block per constraint (cid, rhs, MODE 1, term lines)."""
    with open(path, "w") as f:
        f.write("LMPC\n")
        for row in lmpc_rows:
            f.write(f"  {row.cid}  {row.rhs:.10e}  MODE 1\n")
            for nid, dof, coeff in row.terms:
                f.write(f"  {nid}  {dof}  {coeff:.10e}\n")
        f.write("*\n")


def _write_force(force_bcs: list, path: Path) -> None:
    """Write FORCE.include: one (node, fx, fy, fz) per line."""
    with open(path, "w") as f:
        f.write("FORCE\n")
        for nid, fx, fy, fz in force_bcs:
            f.write(f"  {nid}  {fx:.6e}  {fy:.6e}  {fz:.6e}\n")
        f.write("*\n")


def _write_eframes(rev_joints: list["JointInfo"], path: Path) -> None:
    """Write EFRAMES.include: local frame per revolute joint (e1=crease axis)."""
    with open(path, "w") as f:
        f.write("EFRAMES\n")
        for j in rev_joints:
            e1 = np.array(j.axis, dtype=float)

            # Build orthonormal e2, e3
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, e1)) > 0.8:
                tmp = np.array([0.0, 1.0, 0.0])
            e3 = np.cross(e1, tmp)
            e3 /= np.linalg.norm(e3)
            e2 = np.cross(e3, e1)
            e2 /= np.linalg.norm(e2)

            f.write(
                f"  {j.eid}"
                f"  {e1[0]:.6e} {e1[1]:.6e} {e1[2]:.6e}"
                f"  {e2[0]:.6e} {e2[1]:.6e} {e2[2]:.6e}"
                f"  {e3[0]:.6e} {e3[1]:.6e} {e3[2]:.6e}\n"
            )
        f.write("*\n")
