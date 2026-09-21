#!/bin/bash
#SBATCH --job-name=spot-genzarr
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=data/slurm/%x-%j.out
#
# Replay PerAct demonstrations in CoppeliaSim and write the SPOT zarr buffers.
# CPU-only job (cluster caps CPU-only jobs at 32 CPU / 128 GiB).
#
#   sbatch scripts/sbatch_gen_demonstration.sh insert_onto_square_peg
#   sbatch scripts/sbatch_gen_demonstration.sh            # all 13 tasks
#
# One task per job is usually better: they are independent, and a single job
# for all 13 would sit on the queue far longer than it needs to.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/spot_env.sh
mkdir -p data/slurm

export PERACT_RAW="${PERACT_RAW:-/home/nas_main/hyojinjang/data/spot/peract/raw}"
export ZARR_OUT="${ZARR_OUT:-/home/nas_main/hyojinjang/data/spot/rlbench_zarr}"
export TASKS="${1:-}"

# CoppeliaSim needs an X display even headless; there is none on the cluster,
# so run a private Xvfb for the lifetime of this job.
# GLX must be on: CoppeliaSim creates a real GL context (direct rendering
# against the NVIDIA driver works fine even on Xvfb).
DISPLAY_NUM=$(( 90 + ${SLURM_JOB_ID:-0} % 100 ))
Xvfb ":$DISPLAY_NUM" -screen 0 1024x768x24 +extension GLX +render -noreset >/dev/null 2>&1 &
XVFB_PID=$!
trap 'kill $XVFB_PID 2>/dev/null || true' EXIT
export DISPLAY=":$DISPLAY_NUM"
sleep 5
# fail fast rather than crashing inside CoppeliaSim if the server did not come up
python -c "import ctypes,sys; x=ctypes.CDLL('libX11.so.6'); x.XOpenDisplay.restype=ctypes.c_void_p; sys.exit(0 if x.XOpenDisplay(b'$DISPLAY') else 1)" \
    || { echo "Xvfb on $DISPLAY did not start"; exit 1; }

echo "PERACT_RAW=$PERACT_RAW"
echo "ZARR_OUT=$ZARR_OUT"
echo "TASKS=${TASKS:-<all>}"

bash scripts/gen_demonstration_rlbench.sh
