"""
writer.py — Step 6 of the AeroOrigami pipeline.

write_aeros(surrogate, output_dir, config=None, sim=None) writes all AERO-S
include files needed to run the origami fold simulation.

Always written
--------------
ORIGAMI_MESH.include   NODES + TOPOLOGY (shells + joints + cables) + ATTRIBUTES
                       + MATERIAL for spherical joints
ACTUATORS.include      MATERIAL: one CONMAT RAMP line per revolute joint
EFRAMES.include        EFRAMES: local frame per revolute joint

When config (ModelConfig from add_physics) is also passed
---------------------------------------------------------
DISP.include           Dirichlet BCs (if any)
LMPC.include           Inequality constraints (if any)
USDF.include           User-defined force DOF list (if any force_bcs)
control.C              C++ force application file (if any force_bcs)

When sim (SimConfig) is also passed
------------------------------------
MATERIAL.include       Shell + cable spring material properties
fold.fem               Main AERO-S input file with configurable parameters;
                       INCLUDE lines for LMPC / USDF / DISP omitted when unused
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .surrogate import Surrogate, JointInfo
    from .physics import ModelConfig

# Attribute IDs — must match AERO-S ATTRIBUTES section
_SHELL_ATTR     = 1    # all shell elements (types 15 / 1515)
_SPH_ATTR       = 2    # all spherical joints (type 120)
_REV_ATTR_START = 3    # revolute joints start here, one per joint (type 126)
_CABLE_ATTR     = 10000  # cable tension-only springs (type 203)
# NOTE: _REV_ATTR_START + len(revolute_joints) can reach thousands, so cable
# attr must be a large fixed value that won't collide with revolute joint attrs.


@dataclass
class SimConfig:
    """
    Simulation parameters written into fold.fem and MATERIAL.include.

    Tags in parentheses (e.g. ENDTIME) match the AERO-S template comments.

    Shell material (attribute 1, AERO-S types 15 / 1515)
    Cable spring  (attribute 10, AERO-S type 203 — SPRINGMAT)
    """
    # Identification
    project_name:    str   = "AeroOrigami"
    sim_name:        str   = "origami_fold"

    # Generalized-alpha time integration
    time_step:       float = 5e-5     # dt
    end_time:        float = 1.0      # ENDTIME
    rho:             float = 0.7      # RHO — numerical dissipation

    # Rayleigh damping
    a_damp:      float = 1e-7     # ADAMP (stiffness proportional)
    b_damp:       float = 2.0      # BDAMP (mass proportional)

    # Solver / constraints
    solver:          str   = "sparse" # SOLVERNAME
    lmpc_penalty:    float = 1e8      # LMPCPENALTYSTRENGTH

    # Output / restart frequency (every N time steps)
    output_freq:     int   = 100      # OUTPUTFREQ
    restart_freq:    int   = 100      # RESTARTFREQ

    # Shell material: AERO-S format  MID 0 E nu rho 0 0 t 0 0 0 0 0 0 0
    shell_E:         float = 1e7      # Young's modulus
    shell_nu:        float = 0.4      # Poisson ratio
    shell_rho:       float = 40000.0  # area mass density
    shell_t:         float = 1.0      # thickness

    # Cable (type-203 tension-only spring) stiffness
    cable_stiffness: float = 10000.0  # SPRINGMAT axial stiffness


# ── AERO-S element type mapping ───────────────────────────────────────────────

def _aeros_etype(nids: list[int]) -> int:
    """Map element node count → AERO-S element type."""
    n = len(nids)
    if n == 2:
        return 203     # tension-only spring (cable)
    if n == 3:
        return 15      # triangular AQR shell
    if n == 4:
        return 1515    # quad shell — only reached when split_quads=False
    raise ValueError(f"No AERO-S shell type for {n}-node element")


# ── Public API ────────────────────────────────────────────────────────────────

def write_aeros(
    surrogate:   "Surrogate",
    output_dir:  str | Path,
    config:      "ModelConfig | None" = None,
    sim:         SimConfig | None = None,
    beta_factor: float = 1.0,
) -> dict[str, Path]:
    """
    Write AERO-S include files for the fold surrogate.

    Parameters
    ----------
    surrogate   : Surrogate from build_surrogate()
    output_dir  : directory to write files into (created if needed)
    config      : ModelConfig from add_physics() — writes DISP.include,
                  LMPC.include, USDF.include, control.C as needed, and
                  merges cable elements into ORIGAMI_MESH.include
    sim         : SimConfig — if provided, also writes MATERIAL.include and
                  fold.fem (the main AERO-S input file)
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

    rev_attr: dict[int, int] = {
        j.eid: _REV_ATTR_START + i for i, j in enumerate(rev_joints)
    }

    mesh_path = out / "ORIGAMI_MESH.include"
    act_path  = out / "ACTUATORS.include"
    efr_path  = out / "EFRAMES.include"

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
            usdf_entries = _expand_force_bcs(config.force_bcs)
            p = out / "USDF.include"
            _write_usdf(usdf_entries, p)
            written["usdf"] = p
            print(f"  Wrote {p.name}  ({len(usdf_entries)} USDF entries)")

            p = out / "control.C"
            _write_control_c(usdf_entries, config.force_bcs, p)
            written["control_c"] = p
            print(f"  Wrote {p.name}")

    if sim is not None:
        p = out / "MATERIAL.include"
        _write_material(sim, surrogate.penalty_stiffness, p)
        written["material"] = p
        print(f"  Wrote {p.name}")

        p = out / "fold.fem"
        _write_input_file(sim, config, out, p)
        written["input"] = p
        print(f"  Wrote {p.name}")

        _write_cluster_scripts(sim, out)

    return written


# ── File writers ───────────────────────────────────────────────────────────────

def _write_mesh(
    surrogate:  "Surrogate",
    path:       Path,
    sph_joints: list["JointInfo"],
    rev_joints: list["JointInfo"],
    rev_attr:   dict[int, int],
    config:     "ModelConfig | None",
) -> None:
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
            etype    = _aeros_etype(nids)
            node_str = "  ".join(str(n) for n in nids)
            f.write(f"  {eid}  {etype}  {node_str}\n")
        for j in sph_joints:
            f.write(f"  {j.eid}  120  {j.node_a}  {j.node_b}\n")
        for j in rev_joints:
            f.write(f"  {j.eid}  126  {j.node_a}  {j.node_b}\n")
        for eid, _, nids in cable_elements:
            etype    = _aeros_etype(nids)
            node_str = "  ".join(str(n) for n in nids)
            f.write(f"  {eid}  {etype}  {node_str}\n")
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

        # ── MATERIAL: spherical joint ─────────────────────────────────────────
        # Shell (attr 1) and cable (attr 10) materials are in MATERIAL.include.
        # Revolute joint materials are in ACTUATORS.include.
        f.write("MATERIAL\n")
        f.write(f"  {_SPH_ATTR}  CONMAT penalty {surrogate.penalty_stiffness:.6e}\n")
        f.write("*\n")


def _write_actuators(
    rev_joints: list["JointInfo"],
    rev_attr:   dict[int, int],
    beta:       float,
    path:       Path,
) -> None:
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


def _write_material(sim: SimConfig, penalty: float, path: Path) -> None:
    """
    Write MATERIAL.include: shell element properties and cable spring stiffness.

    Shell (attr 1) format: MID 0 E nu rho 0 0 t 0 0 0 0 0 0 0
    Cable (attr 10) format: MID SPRINGMAT stiffness
    """
    with open(path, "w") as f:
        f.write("MATERIAL\n")
        f.write(
            f"  {_SHELL_ATTR}  0"
            f"  {sim.shell_E:.6e}"
            f"  {sim.shell_nu:.6f}"
            f"  {sim.shell_rho:.6f}"
            f"  0  0"
            f"  {sim.shell_t:.6f}"
            f"  0  0  0  0  0  0  0\n"
        )
        f.write(
            f"  {_CABLE_ATTR}  SPRINGMAT  {sim.cable_stiffness:.6e}\n"
        )
        f.write("*\n")


def _write_disp(disp_bcs: list, path: Path) -> None:
    with open(path, "w") as f:
        f.write("DISP\n")
        for nid, dofs in disp_bcs:
            for dof in dofs:
                f.write(f"  {nid}  {dof}  0.0\n")
        f.write("*\n")


def _write_lmpc(lmpc_rows: list, path: Path) -> None:
    with open(path, "w") as f:
        f.write("LMPC\n")
        for row in lmpc_rows:
            f.write(f"  {row.cid}  {row.rhs:.10e}  MODE 1\n")
            for nid, dof, coeff in row.terms:
                f.write(f"  {nid}  {dof}  {coeff:.10e}\n")
        f.write("*\n")


def _expand_force_bcs(
    force_bcs: list[tuple[int, float, float, float]],
) -> list[tuple[int, int, float]]:
    """
    Expand (nid, fx, fy, fz) force entries to per-DOF USDF entries.

    Zero force components are skipped. Returns a list of
    (nid, dof, magnitude) sorted by nid then dof, with USDF index = list position.
    """
    entries: list[tuple[int, int, float]] = []
    for nid, fx, fy, fz in force_bcs:
        for dof, mag in ((1, fx), (2, fy), (3, fz)):
            if mag != 0.0:
                entries.append((nid, dof, mag))
    return entries


def _write_usdf(
    usdf_entries: list[tuple[int, int, float]],
    path: Path,
) -> None:
    """
    Write USDF.include.

    Format per line: <nid> <dof>
    Index in the list = usdForce[] index in control.C usd_forc().
    """
    with open(path, "w") as f:
        f.write("USDF\n")
        for nid, dof, _ in usdf_entries:
            f.write(f"  {nid}  {dof}\n")


def _write_control_c(
    usdf_entries: list[tuple[int, int, float]],
    force_bcs:    list[tuple[int, float, float, float]],
    path: Path,
) -> None:
    """
    Write control.C — compilable C++ control file for AERO-S.

    usd_forc() applies constant forces from config.force_bcs.
    Edit the magnitudes or add time-variation as needed before compiling.

    Compile:  g++ -shared -fPIC control.C -o control.so
    """
    assignments = []
    for i, (nid, dof, mag) in enumerate(usdf_entries):
        dof_name = {1: "Fx", 2: "Fy", 3: "Fz"}.get(dof, f"DOF{dof}")
        assignments.append(
            f"  usdForce[{i}] = {mag:.6e};  "
            f"/* node {nid}  {dof_name} */"
        )

    body = "\n".join(assignments) if assignments else "  /* no forces */"

    text = f"""\
#include <cstdio>
#include <cmath>
#include "ControlInterface.h"

// Compile: g++ -shared -fPIC control.C -o control.so
//
// usdForce[i] is the force for USDF entry i (see USDF.include for node/DOF mapping).
// Modify the magnitudes below to add time variation, ramps, etc.

class MyControl : public ControlInterface {{
  public:
    void init(double *displacement, double *velocity, double *acceleration,
              SingleDomainDynamic * probDesc=0);

    void ctrl(double *displacement, double *velocity, double *acceleration,
              double *force, double time=0, SysState<Vector> *state=0,
              Vector *ext_f=0);

    void usd_disp(double time, double *userDefineDisplacement,
                  double *userDefineVelocity, double *userDefineAcceleration);

    void usd_forc(double time, double *userDefineForce);

    void usd_joint(double time, int mid, double *userDefineFunc,
                   double *userDefineVelocity, double *userDefineAcceleration);
}};

ControlInterface *controlObj = new MyControl();

void MyControl::ctrl(double *displacement, double *velocity, double *acceleration,
                     double *force, double time, SysState<Vector> *state,
                     Vector *ext_f)
{{ /* Not used */ }}

void MyControl::init(double *displacement, double *velocity, double *acceleration,
                     SingleDomainDynamic * probDesc)
{{ /* Not used */ }}

void MyControl::usd_disp(double time, double *userDefineDisp,
                         double *userDefineVel, double *userDefineAcc)
{{ /* Not used */ }}

void MyControl::usd_forc(double time, double *usdForce)
{{
{body}
}}

void MyControl::usd_joint(double time, int mid, double *userDefineFunc,
                          double *userDefineVelocity, double *userDefineAcceleration)
{{ /* Not used */ }}
"""
    path.write_text(text)


def _write_input_file(
    sim:    SimConfig,
    config: "ModelConfig | None",
    out:    Path,
    path:   Path,
) -> None:
    """
    Write fold.fem — the main AERO-S input file.

    INCLUDE lines for LMPC, USDF/LOAD, and DISP are omitted when the
    corresponding config sections are empty or config is None.
    """
    has_lmpc = config is not None and bool(config.lmpc_rows)
    has_disp = config is not None and bool(config.disp_bcs)
    has_usdf = config is not None and bool(config.force_bcs)

    lines: list[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    sep = "*" * 80

    L(sep)
    L(f"** PROJECT NAME:      {sim.project_name}")
    L(f"** SIMULATION NAME:   {sim.sim_name}")
    L(sep)
    L("* See AERO-S manual (https://frg.bitbucket.io/aero-s/) for documentation")
    L(sep)
    L("CONTROL")
    L(f'"{sim.sim_name}"')
    L("1")
    L('"NodeSet"')
    L('"ElemSet"')
    L(sep)
    L("* Mesh file")
    L('INCLUDE "./ORIGAMI_MESH.include"')
    L(sep)
    L("* Material properties")
    L('INCLUDE "./MATERIAL.include"')
    L(sep)
    if has_lmpc:
        L("* LMPCs to guide folding")
        L('INCLUDE "./LMPC.include"')
        L(sep)
    L("* EFRAMES reference axes for joint elements")
    L('INCLUDE "./EFRAMES.include"')
    L(sep)
    L("* Actuation: driver joints")
    L('INCLUDE "./ACTUATORS.include"')
    if has_usdf:
        L("* User-defined forces (see control.C for time profiles)")
        L('INCLUDE "./USDF.include"')
        L('LOAD "./control.so"')
    L(sep)
    if has_disp:
        L("* DISP constraints (Dirichlet BCs)")
        L('INCLUDE "./DISP.include"')
        L(sep)
    L("NONLINEAR")
    L(sep)
    L("DYNAMICS")
    L("newmark")
    L(f"mech {sim.rho}")
    L(f"time 0 {sim.time_step:.6e} {sim.end_time:.6e}")
    L(f"RAYDAMP {sim.a_damp:.6e} {sim.b_damp:.6e}")
    L(sep)
    L("STATICS")
    L(f"{sim.solver}")
    L(sep)
    L("CONSTRAINTS")
    L(f"penalty {sim.lmpc_penalty:.6e}")
    L(sep)
    L("RESTART")
    L(f'"references/StructuralRestart.data" {sim.restart_freq}')
    L("* Uncomment to restart and append .2 to output files:")
    L('*"references/StructuralRestart.data" ".2"')
    L(sep)
    L("OUTPUT")
    L(f'gdisplac "results/gdisplac3.xpost" {sim.output_freq}')
    L(f'stressp1 "results/stressp1.xpost" {sim.output_freq}')
    L(f'strainp1 "results/strainp1.xpost" {sim.output_freq}')
    L(sep)
    L("* 6-DOF output for initial conditions of inflation simulation (Step 7)")
    L("OUTPUT6")
    L(f'gdisplac "results/gdisplac6.xpost" {sim.output_freq}')
    L(sep)
    L("END")

    path.write_text("\n".join(lines) + "\n")


def _write_eframes(rev_joints: list["JointInfo"], path: Path) -> None:
    with open(path, "w") as f:
        f.write("EFRAMES\n")
        for j in rev_joints:
            e1 = np.array(j.axis, dtype=float)

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


# ── Cluster script templates ───────────────────────────────────────────────────

_RUN_SH = """\
#!/bin/bash
# ── Paths — update to match your cluster installation ────────────────────────
AEROS=/home/rtezaur/codes/aero-s/build/bin/aeros
AEROSDIR=/home/pavery/Codes/FEM
GSLDIR=/home/tdurrant/GSL
EIGENDIR=/home/tdurrant/eigen-3.4.0

# Ensure AERO-S output directories exist
mkdir -p postpro references results

# Compile USDF shared library if present
if [ -f control.C ]; then
    g++ -O3 -fPIC -D_TEMPLATE_FIX_ \\
        -I$AEROSDIR \\
        -I$AEROSDIR/Control.d \\
        -I$AEROSDIR/Math.d \\
        -I$AEROSDIR/Utils.d \\
        -I$AEROSDIR/SysState.d \\
        -I$GSLDIR/include \\
        -I$EIGENDIR \\
        -c control.C && \\
    g++ -shared control.o -o control.so
fi

$AEROS -q fold.fem |& tee log.out
"""

_RUN_SBATCH = """\
#!/bin/bash
#SBATCH --job-name={sim_name}
#SBATCH --output=log.out
#SBATCH --error=error.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

chmod 755 run.sh
./run.sh
"""

_POSTPRO_SH = """\
#!/bin/bash
#SBATCH --job-name=postpro
#SBATCH --output=postpro.log
#SBATCH --error=postpro.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=1:00:00

# ── Paths — update to match your cluster installation ────────────────────────
AEROS=/home/rtezaur/codes/aero-s/build/bin/aeros
XP2EXO=/home/pavery/bin/xp2exo

# Generate topology file and convert to Exodus for ParaView
$AEROS -t fold.fem
$XP2EXO {sim_name}.top postpro/{sim_name}.exo results/gdisplac3.xpost results/stressp1.xpost results/strainp1.xpost
"""

_CLEAN_SH = """\
#!/bin/bash
# Removes all AERO-S output, keeping the directory structure intact.
rm -f log.out log.err postpro.log postpro.err
rm -f residuals *.timing *.top control.o control.so
rm -f postpro/*
rm -f results/*
rm -f references/*
"""


def _write_cluster_scripts(sim: SimConfig, out: Path) -> None:
    """Write run.sh, run.sbatch, postpro.sh, clean.sh into the sim dir."""
    (out / "run.sh").write_text(_RUN_SH)
    (out / "run.sbatch").write_text(_RUN_SBATCH.format(sim_name=sim.sim_name))
    (out / "postpro.sh").write_text(_POSTPRO_SH.format(sim_name=sim.sim_name))
    (out / "clean.sh").write_text(_CLEAN_SH)
    print("  Wrote run.sh, run.sbatch, postpro.sh, clean.sh")
