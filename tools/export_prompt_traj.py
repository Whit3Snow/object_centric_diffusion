"""Export object-trajectory prompts from a SPOT zarr buffer.

Each episode's ``state`` (grasp object pose in the target object frame) is
uniformly resampled to ``--prompt_length`` poses and written as a
``(prompt_length, 7)`` ``.npy`` file, which is what the RLBench runner loads at
evaluation time (``env_runner.prompt_path``).

Example
-------
    python tools/export_prompt_traj.py \
        --zarr_dir /tmp/rlbench_zarr/train/insert_onto_square_peg/all_variations \
        --out_dir  prompts/insert_onto_square_peg \
        --prompt_length 16 --num_prompts 5
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import os

import numpy as np

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.dataset.rlbench_icl_dataset import resample_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_dir", required=True,
                        help="<...>/<task>/all_variations (the dir containing 'zarr')")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prompt_length", type=int, default=16)
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="how many episodes to export (-1 = all)")
    parser.add_argument("--start_episode", type=int, default=0)
    args = parser.parse_args()

    zarr_path = os.path.join(args.zarr_dir, 'zarr_pt' if 'pt' in args.zarr_dir else 'zarr')
    print(f"[export_prompt_traj] reading {zarr_path}")
    replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['state'])

    episode_ends = np.asarray(replay_buffer.episode_ends[:])
    states = replay_buffer['state']
    n_episodes = len(episode_ends)

    end = n_episodes if args.num_prompts < 0 else min(n_episodes, args.start_episode + args.num_prompts)
    os.makedirs(args.out_dir, exist_ok=True)

    for ep_idx in range(args.start_episode, end):
        start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        traj = np.asarray(states[start:episode_ends[ep_idx]])
        prompt = resample_trajectory(traj, args.prompt_length).astype(np.float32)

        out_path = os.path.join(args.out_dir, f"demo_{ep_idx:03d}.npy")
        np.save(out_path, prompt)
        print(f"  episode {ep_idx}: {traj.shape} -> {prompt.shape} -> {out_path}")

    print(f"[export_prompt_traj] wrote {end - args.start_episode} prompts to {args.out_dir}")


if __name__ == "__main__":
    main()
