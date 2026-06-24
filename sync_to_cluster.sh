#!/bin/bash
# Sync a sim_files directory to your computing cluster.
#
# Usage:
#   ./sync_to_cluster.sh <simulation_dir> <run_name>
#   ./sync_to_cluster.sh examples/dgb_parachute dgb_v1
#
# The run_name becomes the directory name on the cluster.
# Run again with the same run_name to overwrite/update files.

# ── Configure once ────────────────────────────────────────────────────────────
HOST=independence2
REMOTE_BASE=/home/tdurrant/parachute/aeroorigami

# ── Args ──────────────────────────────────────────────────────────────────────
SIMULATION=${1:?Usage: $0 <simulation_dir> <run_name>}
RUNNAME=${2:?Usage: $0 <simulation_dir> <run_name>}

SIM_FILES="$SIMULATION/sim_files"
REMOTE="$HOST:$REMOTE_BASE/$RUNNAME"

if [ ! -d "$SIM_FILES" ]; then
    echo "Error: $SIM_FILES does not exist. Run the simulation script first."
    exit 1
fi

echo "Syncing $SIM_FILES → $REMOTE ..."
rsync -avz --exclude="*.msh" --exclude="__pycache__" "$SIM_FILES/" "$REMOTE/"

echo ""
echo "Done. To submit the job:"
echo "  ssh $HOST 'cd $REMOTE_BASE/$RUNNAME && sbatch run.sbatch'"
