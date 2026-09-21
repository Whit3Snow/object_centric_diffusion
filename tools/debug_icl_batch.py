"""Step 1 / Step 2 smoke test for the trajectory-prompt ICL setup.

    python tools/debug_icl_batch.py --config-name prompt_dp3

Prints the shapes of one batch

    obs.agent_pos: [B, 4, 7]
    obs.demo_traj: [B, 16, 7]
    action:        [B, 4, 8]

checks that the prompt really comes from a *different* episode than the sample
it is paired with, and (unless ``--dataset-only``) runs one forward pass of
``PromptSimpleDP3`` so that a shape error shows up in seconds instead of after
a full training launch.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader

from diffusion_policy_3d.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _leaf_datasets(dataset):
    return getattr(dataset, 'dataset_list', [dataset])


def check_prompt_episodes(dataset, n_check=200):
    """The prompt must not come from the episode the sample was taken from."""
    for leaf in _leaf_datasets(dataset):
        if leaf.prompt_mode not in ('other_episode',):
            cprint(f"[check] prompt_mode={leaf.prompt_mode}, skipping episode-disjointness check", "yellow")
            continue
        if len(leaf.prompt_episodes) <= 1:
            cprint("[check] only one episode available, skipping", "yellow")
            continue
        n = min(n_check, len(leaf))
        rng = np.random.default_rng(0)
        for idx in rng.choice(len(leaf), size=n, replace=False):
            idx = int(idx)
            target_ep = int(leaf._index_to_episode[idx])
            prompt = leaf.get_prompt(idx)
            # the prompt must match exactly one episode of the bank, not ours
            match = np.all(np.isclose(leaf.prompt_bank, prompt[None]), axis=(1, 2))
            assert not match[target_ep], \
                f"sample {idx} got the prompt of its own episode {target_ep}"
        cprint(f"[check] {n} samples: prompt episode != execution episode  OK", "green")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parent.parent.joinpath('config'))
)
def main(cfg):
    cprint(f"prompt_length: {cfg.prompt_length}  prompt_mode: {cfg.prompt_mode}", "cyan")

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    cprint(f"[dataset] {type(dataset).__name__}  len={len(dataset)}", "cyan")

    check_prompt_episodes(dataset)

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    cprint("---- batch ----", "cyan")
    for key, value in batch['obs'].items():
        print(f"  obs.{key}: {tuple(value.shape)}  {value.dtype}")
    print(f"  action:    {tuple(batch['action'].shape)}  {batch['action'].dtype}")

    demo = batch['obs']['demo_traj']
    assert demo.shape[1:] == (cfg.prompt_length, 7), demo.shape
    quat_norm = torch.linalg.norm(demo[..., 3:7], dim=-1)
    print(f"  demo_traj quaternion norm: min={quat_norm.min():.4f} max={quat_norm.max():.4f}")
    print(f"  demo_traj differs across the batch: "
          f"{not torch.allclose(demo[0], demo[1])}")

    # ---- one forward pass of the policy -------------------------------
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device)

    batch = dict_apply(batch, lambda x: x.to(device))
    loss, loss_dict = policy.compute_loss(batch)
    cprint(f"[policy] compute_loss -> {loss.item():.6f}  {loss_dict}", "green")

    with torch.no_grad():
        torch.manual_seed(0)  # same diffusion noise in both calls below
        result = policy.predict_action(batch['obs'])
    cprint(f"[policy] predict_action -> action {tuple(result['action'].shape)}, "
           f"action_pred {tuple(result['action_pred'].shape)}", "green")

    # the prompt must actually change the prediction
    with torch.no_grad():
        obs = dict(batch['obs'])
        obs['demo_traj'] = torch.zeros_like(obs['demo_traj'])
        torch.manual_seed(0)
        result_zero = policy.predict_action(obs)
    delta = (result['action'] - result_zero['action']).abs().mean().item()
    cprint(f"[policy] |action(prompt) - action(zero prompt)| = {delta:.6f} "
           f"(must be > 0, an untrained net is still prompt-sensitive)", "green")


if __name__ == "__main__":
    main()
