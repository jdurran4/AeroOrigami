"""
Quick Step 7 iteration — loads pipeline state from run.py's cache.

Usage:
    python examples/dgb_parachute/step7.py

Tweak method, rbf_neighbors, etc. below and re-run; no need to redo Steps 1-6.
Run run.py at least once first to generate pipeline_state.pkl.
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyaeroori import map_displacements, write_idisp6, write_folded_vtk

HERE      = Path(__file__).parent
state_pkl = HERE / "sim_files" / "pipeline_state.pkl"
output_dir = HERE / "sim_files"

with open(state_pkl, "rb") as f:
    state = pickle.load(f)

surrogate = state["surrogate"]
mesh      = state["mesh"]
config    = state["config"]

disp_file  = output_dir / "gdisplac6.xpost"
idisp_file = output_dir / "IDISP6.include"
vtk_file   = output_dir / "folded_fine_mesh.vtk"

# ── Tweak these ───────────────────────────────────────────────────────────────
METHOD        = "rbf"   # "rbf" or "panel_rigid"
RBF_NEIGHBORS = 100             # only used by method="rbf"
RBF_SMOOTHING = 1e-5
# ─────────────────────────────────────────────────────────────────────────────

displacements = map_displacements(
    surrogate,
    mesh,
    disp_file,
    config=config,
    step=-1,
    method=METHOD,
    rbf_neighbors=RBF_NEIGHBORS,
    rbf_smoothing=RBF_SMOOTHING,
)

write_idisp6(displacements, mesh, idisp_file)
write_folded_vtk(mesh, displacements, vtk_file)
print("Done.")
