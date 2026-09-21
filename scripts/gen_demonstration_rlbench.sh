#!/bin/bash
# Generate the SPOT zarr datasets from PerAct's raw demonstrations.
#
# Paths and the task list are overridable so this can point at NAS and be run
# one task at a time (e.g. from scripts/sbatch_gen_demonstration.sh):
#
#   PERACT_RAW=<DATA>/peract/raw ZARR_OUT=<DATA>/rlbench_zarr \
#   TASKS=insert_onto_square_peg bash scripts/gen_demonstration_rlbench.sh
#
# Needs an X display (CoppeliaSim); see scripts/sbatch_gen_demonstration.sh.
set -euo pipefail

PERACT_RAW="${PERACT_RAW:-/tmp/peract/raw/}"
ZARR_OUT="${ZARR_OUT:-/tmp/rlbench_zarr/}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
TEST_EPISODES="${TEST_EPISODES:-25}"

# single-object tasks / multi-stage tasks; TASKS= filters both lists
SINGLE_TASKS="meat_off_grill place_wine_at_rack_location insert_onto_square_peg \
  put_groceries_in_cupboard place_shape_in_shape_sorter reach_and_drag \
  put_money_in_safe turn_tap light_bulb_in close_jar"
MULTI_TASKS="stack_cups stack_blocks place_cups"

if [ -n "${TASKS:-}" ]; then
    keep() { for t in ${TASKS//,/ }; do [ "$t" = "$1" ] && return 0; done; return 1; }
else
    keep() { return 0; }
fi

run_split() {  # <script> <task> <split> <episodes>
    python "$1" \
        --peract_demo_dir="$PERACT_RAW" \
        --save_path="$ZARR_OUT" \
        --tasks="$2" --variations=-1 --processes=1 --split="$3" --episodes_per_task="$4"
}

# peract setting
for task_name in $SINGLE_TASKS; do
    keep "$task_name" || continue
    echo "=== $task_name (single) ==="
    run_split env_rlbench_peract/data/collect_zarr_rlbench_peract.py "$task_name" train "$TRAIN_EPISODES"
    run_split env_rlbench_peract/data/collect_zarr_rlbench_peract.py "$task_name" test  "$TEST_EPISODES"
done

# peract setting (multi-object task)
for task_name in $MULTI_TASKS; do
    keep "$task_name" || continue
    echo "=== $task_name (multi-stage) ==="
    run_split env_rlbench_peract/data/collect_zarr_rlbench_peract_multi_stage.py "$task_name" train "$TRAIN_EPISODES"
    run_split env_rlbench_peract/data/collect_zarr_rlbench_peract_multi_stage.py "$task_name" test  "$TEST_EPISODES"
done
