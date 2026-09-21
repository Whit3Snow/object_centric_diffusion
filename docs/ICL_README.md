# Object trajectory as an in-context task prompt

Research question of this MVP:

> **Can an object trajectory function as an in-context task prompt?**

Not video ICL, not AgentSTAR yet - the cheapest possible test on top of SPOT's
perfect simulator object states.

Standard SPOT learns

```
s_t  ->  a_{t:t+H}
```

where `s_t = T_target^-1 T_object` (7D: xyz + quaternion) and `a_t` is the delta
between consecutive states. Here every sample is additionally conditioned on the
*whole object trajectory of another episode of the same task*:

```
(tau_i , s_{j,t})  ->  a_{j,t:t+H}        i != j
```

so the prompt episode and the answer episode are deliberately different.

## What was added

```
diffusion_policy_3d/
  dataset/rlbench_icl_dataset.py        NEW  __getitem__ adds obs['demo_traj'] (K,7)
  dataset/rlbench_icl_dataset_list.py   NEW  multi-task version, configurable task list
  model/trajectory_prompt_encoder.py    NEW  (K,7) -> prompt_dim MLP
  policy/prompt_simple_dp3.py           NEW  global_cond = [h_obs ; h_demo]
env_rlbench/
  policy/dp3_policy.py                  demo_traj forwarded to predict_action
  runner/rlbench_runner.py              prompt_path -> demo_traj loaded at eval
config/
  prompt_dp3.yaml                       NEW  training entrypoint config
  task/rlbench_icl_multi.yaml           NEW  multi-task, language OFF
  task/rlbench_icl/insert_onto_square_peg.yaml  NEW  single-task smoke test
tools/
  export_prompt_traj.py                 NEW  zarr episodes -> prompts/<task>/demo_XXX.npy
  debug_icl_batch.py                    NEW  Step 1/2 shape + forward-pass check
```

`rlbench_base_dataset.py`, `rlbench_dataset.py`, `simple_dp3.py`,
`train_dp3.py` and the original configs are **unchanged**, so the SPOT baseline
still runs exactly as before.

### Two implementation notes

* The prompt lives in **`obs['demo_traj']`**, not at the top level of the
  sample. `tools/train_dp3.py` always forwards `batch['obs']` to the policy, so
  training, validation and the sampling step keep working without any edit.
  `PromptSimpleDP3.predict_action` pops it out before the observation encoder.
  For rollout it can also be passed explicitly: `predict_action(obs, demo_traj=...)`.
* `SequenceSampler.sample_sequence` does not return an episode id, so
  `RLBenchICLDataset` rebuilds the *sample index -> episode index* map from
  `sampler.indices` (`searchsorted` over `episode_ends`, with the extra leading
  `aug_idx` column handled).

## prompt_mode - the conditions to compare

| `prompt_mode`   | prompt content                        | role |
|-----------------|---------------------------------------|------|
| `other_episode` | another episode of the **same** task  | **B** the ICL model |
| `wrong_task`    | an episode of a **different** task    | **C** control |
| `zero`          | all zeros (no information)            | **A** no-prompt baseline, same architecture |
| `shuffled`      | correct prompt, time order destroyed  | control: is it the *trajectory* or just the poses? |
| `self`          | the sample's own episode              | leakage upper bound / debugging |

If A ~ B ~ C, the policy is ignoring the prompt. The result we are looking for
is B >> A and C << A.

## Order of work

**Step 0** - original SPOT trains and evaluates on one task (unchanged code).

**Step 1 + 2** - shapes and one forward pass:

```bash
python tools/debug_icl_batch.py --config-name prompt_dp3
```

prints `obs.agent_pos [B,4,7]`, `obs.demo_traj [B,16,7]`, `action [B,4,8]`,
asserts the prompt never comes from the sample's own episode, and runs
`compute_loss` + `predict_action`.

**Step 3** - single task, correct vs. shuffled prompt:

```bash
python tools/train_dp3.py --config-name prompt_dp3 prompt_mode=other_episode
python tools/train_dp3.py --config-name prompt_dp3 prompt_mode=shuffled
```

With a single task the policy has no reason to read the prompt
("I always do insert_onto_square_peg anyway"), so this only proves the
plumbing works.

**Step 4** - multi-task, **language off**. This is where the question actually
gets answered: with `has_lang_emb=false` / `use_lang_emb=false` the demo
trajectory is the only thing that identifies the task.

```bash
python tools/train_dp3.py --config-name prompt_dp3 task=rlbench_icl_multi prompt_mode=zero          # A
python tools/train_dp3.py --config-name prompt_dp3 task=rlbench_icl_multi prompt_mode=other_episode  # B
python tools/train_dp3.py --config-name prompt_dp3 task=rlbench_icl_multi prompt_mode=wrong_task     # C
```

**Step 5** - held-out tasks / compositions (shrink `task.dataset.task_list`).

**Step 6** - only then: video prompts and AgentSTAR.

## Evaluation with a prompt

Export a prompt trajectory, then evaluate:

```bash
python tools/export_prompt_traj.py \
    --zarr_dir /tmp/rlbench_zarr/train/insert_onto_square_peg/all_variations \
    --out_dir  prompts/insert_onto_square_peg \
    --prompt_length 16 --num_prompts 5

python tools/eval_dp3.py --config-name prompt_dp3 \
    evaluation.eval_epoch=1000 \
    task.env_runner.prompt_path=prompts/insert_onto_square_peg/demo_000.npy
```

Point `prompt_path` at another task's prompt to get condition **C** at rollout
time. `prompt_path: null` restores the prompt-free behaviour.

## MVP simplifications (deliberate)

* **Augmentation off** (`random_aug: false`, `symmetric_axis: null`). The prompt
  and the execution episode come from different episodes; augmenting them
  independently would scramble the representation before we know whether the
  idea works at all. Turn it back on after ICL is shown to work.
* **Uniform index subsampling**, no interpolation, so every prompt pose is a
  real pose and the quaternions stay unit norm.
* **MLP prompt encoder**, not a Transformer - whether a sequence architecture
  helps is not the question yet.
* Prompt positions are normalized with the dataset's `agent_pos` normalizer and
  quaternions are re-normalized separately, matching SPOT's `custom_normalize`.
