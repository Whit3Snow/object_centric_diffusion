#!/bin/bash
#SBATCH --job-name=spot-icl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=data/slurm/%x-%j.out
# QOS: no --qos => "extra" (preemptible, can be CANCELLED mid-run).
# This account is in the `sub` group (own cap: 4 GPU / 56 CPU / 900000 MiB), so
# up to 4 of these 1-GPU jobs fit under protected quota - add `--qos=own` to
# the sbatch line (or pass it on the command line) for the runs that matter.
#
# One sbatch = one experiment (cluster rule 5). Submit the A/B/C conditions
# as three separate jobs rather than looping inside one script:
#
#   sbatch scripts/sbatch_train_icl.sh rlbench_icl_multi zero           # A no prompt
#   sbatch scripts/sbatch_train_icl.sh rlbench_icl_multi other_episode  # B ICL
#   sbatch scripts/sbatch_train_icl.sh rlbench_icl_multi wrong_task     # C control
#
# Single-task smoke test:
#   sbatch scripts/sbatch_train_icl.sh rlbench_icl/insert_onto_square_peg other_episode

set -euo pipefail

TASK="${1:-rlbench_icl_multi}"
PROMPT_MODE="${2:-other_episode}"
SEED="${3:-0}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/spot_env.sh

mkdir -p data/slurm

EXP_NAME="icl_${PROMPT_MODE}"
RUN_DIR="data/outputs/${TASK//\//_}_${PROMPT_MODE}_seed${SEED}"

# ---- resubmit ourselves if we get preempted (extra/share QOS) --------------
_handle_preempt() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State | head -1)
    if [[ "$state" == "PREEMPTED" || "$state" == *"CANCELLED"* ]]; then
        echo "[sbatch] preempted -> resubmitting"
        sbatch "$0" "$TASK" "$PROMPT_MODE" "$SEED"
    fi
    exit 143
}
trap _handle_preempt SIGTERM

python tools/train_dp3.py --config-name=prompt_dp3.yaml \
    task="${TASK}" \
    prompt_mode="${PROMPT_MODE}" \
    hydra.run.dir="${RUN_DIR}" \
    training.seed="${SEED}" \
    training.device="cuda:0" \
    exp_name="${EXP_NAME}" \
    logging.mode=online \
    checkpoint.save_ckpt=True &
wait $!
