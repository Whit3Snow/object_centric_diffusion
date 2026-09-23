#!/bin/bash
#SBATCH --job-name=spot-genzarr
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=data/slurm/%x-%j.out
#
# Replay PerAct demonstrations in CoppeliaSim and write the SPOT zarr buffers.
#
# CPU-only, and no X server: the collector runs with --disable_cameras, so
# CoppeliaSim never creates an OpenGL context. That is not an optimisation but
# a necessity - worker pods have no working GLX even with a GPU attached
# (verified: segfault in COffscreenGlContext with --gres=gpu:1, and
# glXChooseVisual fails outright on a GPU-less pod).
#
#   sbatch scripts/sbatch_gen_demonstration.sh insert_onto_square_peg
#   sbatch scripts/sbatch_gen_demonstration.sh            # all 13 tasks
#
# Runs on the default "extra" QOS (preemptible) on purpose: `own` is only
# 4 GPU / 56 CPU per user and is usually taken by real training runs, so an
# `own` generation job can sit PENDING for days. The work is idempotent, and
# the SIGTERM trap below resubmits the job if it gets preempted.
#
# One task per job is usually better: they are independent, and a single job
# for all 13 would sit on the queue far longer than it needs to.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/spot_env.sh
mkdir -p data/slurm

# Nothing is rendered (--disable_cameras), so Qt must use the offscreen
# platform: spot_env.sh defaults to xcb, which aborts without a DISPLAY.
unset DISPLAY
export QT_QPA_PLATFORM=offscreen

export PERACT_RAW="${PERACT_RAW:-/home/nas_main/hyojinjang/data/spot/peract/raw}"
export ZARR_OUT="${ZARR_OUT:-/home/nas_main/hyojinjang/data/spot/rlbench_zarr}"
export TASKS="${1:-}"


# Preempted on `extra`? Resubmit ourselves. Only on preemption - not when the
# user scancels us (both deliver SIGTERM, so inspect the accounting state).
_handle_preempt() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    if [[ "$state" == "PREEMPTED" || "$state" == *"CANCELLED"* ]]; then
        echo "[sbatch] preempted -> resubmitting ${TASKS:-<all>}"
        sbatch "$0" "${TASKS:-}"
    fi
    exit 143
}
trap _handle_preempt SIGTERM

echo "PERACT_RAW=$PERACT_RAW"
echo "ZARR_OUT=$ZARR_OUT"
echo "TASKS=${TASKS:-<all>}"

# background + wait so the SIGTERM trap fires immediately
bash scripts/gen_demonstration_rlbench.sh &
wait $!
