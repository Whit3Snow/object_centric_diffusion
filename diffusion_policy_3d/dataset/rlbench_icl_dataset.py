"""In-context-learning (ICL) variant of the RLBench dataset.

A normal SPOT sample is

    s_t  ->  a_{t:t+H}

where ``s_t`` is the grasp object pose expressed in the target object frame
(7D: xyz + quaternion) and ``a_t`` is the delta between consecutive states.

Here every sample is additionally conditioned on the *whole object trajectory
of another episode of the same task*:

    (tau_i, s_{j,t}) -> a_{j,t:t+H}        with  i != j

``tau_i`` is uniformly resampled to a fixed number of poses (``prompt_length``)
so that it can be batched, and is returned under ``obs['demo_traj']`` with
shape ``[prompt_length, 7]``.

Keeping it inside ``obs`` (instead of at the top level of the sample) means the
whole training / validation / sampling loop in ``tools/train_dp3.py`` keeps
working untouched - it always forwards ``batch['obs']`` to the policy.
"""

from typing import Dict

import numpy as np
import torch
from termcolor import cprint

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.sampler import SequenceSampler
from diffusion_policy_3d.dataset.rlbench_base_dataset import RLBenchBaseDataset


# prompt sampling strategies
PROMPT_MODES = (
    'other_episode',  # same task, a different episode (the actual ICL setting)
    'self',           # same episode (leakage upper bound / debugging)
    'shuffled',       # correct prompt with its time order destroyed (control)
    'wrong_task',     # a prompt from a different task (control, needs a list)
    'zero',           # all-zero prompt (== no prompt, control)
)


def resample_trajectory(traj: np.ndarray, length: int) -> np.ndarray:
    """Uniformly subsample ``traj`` (T, D) down to ``length`` poses.

    No interpolation on purpose - for the first sanity check plain index
    sampling is enough, and it keeps every prompt pose a real object pose
    (in particular the quaternions stay unit norm).
    """
    assert traj.ndim == 2, f"expected (T, D), got {traj.shape}"
    if len(traj) == 0:
        return np.zeros((length, traj.shape[1]), dtype=traj.dtype)
    indices = np.linspace(0, len(traj) - 1, num=length).round().astype(np.int64)
    return traj[indices]


class RLBenchICLDataset(RLBenchBaseDataset):
    def __init__(self,
                 *args,
                 prompt_length=16,
                 prompt_mode='other_episode',
                 prompt_key='state',
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert prompt_mode in PROMPT_MODES, \
            f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode}"
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.prompt_key = prompt_key

        # prompts coming from other tasks (filled in by RLBenchICLDatasetList)
        self.external_prompt_bank = None

        self._setup_prompts(self.replay_buffer, self.train_mask)

        cprint(f"[ICLDataset] prompt_mode: {self.prompt_mode}", "yellow")
        cprint(f"[ICLDataset] prompt_length: {self.prompt_length}", "yellow")
        cprint(f"[ICLDataset] prompt episodes: {len(self.prompt_episodes)}", "yellow")

    # ------------------------------------------------------------------
    # prompt bank
    # ------------------------------------------------------------------
    def _setup_prompts(self, replay_buffer, episode_mask):
        """Precompute one fixed-length trajectory per episode + the
        sample-index -> episode-index lookup for the current sampler."""
        episode_ends = np.asarray(replay_buffer.episode_ends[:])
        states = replay_buffer[self.prompt_key]

        bank = []
        for ep_idx in range(len(episode_ends)):
            start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end = episode_ends[ep_idx]
            traj = np.asarray(states[start:end])
            bank.append(resample_trajectory(traj, self.prompt_length))
        # (n_episodes, prompt_length, 7)
        self.prompt_bank = np.stack(bank).astype(np.float32)

        # episodes we are allowed to draw prompts from
        if episode_mask is None:
            self.prompt_episodes = np.arange(len(episode_ends))
        else:
            self.prompt_episodes = np.nonzero(np.asarray(episode_mask))[0]

        self._episode_ends = episode_ends
        self._index_to_episode = self._build_index_to_episode(self.sampler, episode_ends)

    @staticmethod
    def _build_index_to_episode(sampler: SequenceSampler, episode_ends: np.ndarray):
        """Map every sampler index to the episode it was taken from.

        ``SequenceSampler.indices`` rows are
        ``(buffer_start, buffer_end, sample_start, sample_end)``, prefixed by
        ``aug_idx`` when symmetric augmentation is enabled.
        """
        indices = np.asarray(sampler.indices)
        if len(indices) == 0:
            return np.zeros((0,), dtype=np.int64)
        col = 0 if sampler.n_aug < 0 else 1
        buffer_start = indices[:, col]
        return np.searchsorted(episode_ends, buffer_start, side='right')

    def set_external_prompt_bank(self, bank: np.ndarray):
        """Prompts from other tasks, used by ``prompt_mode='wrong_task'``."""
        self.external_prompt_bank = np.asarray(bank, dtype=np.float32)

    def get_prompt(self, idx: int) -> np.ndarray:
        """Return the demo trajectory (prompt_length, 7) used for sample ``idx``."""
        if self.prompt_mode == 'zero':
            return np.zeros((self.prompt_length, self.prompt_bank.shape[-1]), dtype=np.float32)

        if self.prompt_mode == 'wrong_task':
            assert self.external_prompt_bank is not None and len(self.external_prompt_bank) > 0, \
                "prompt_mode='wrong_task' requires set_external_prompt_bank()"
            pick = np.random.randint(len(self.external_prompt_bank))
            return self.external_prompt_bank[pick].copy()

        target_episode = int(self._index_to_episode[idx])

        if self.prompt_mode == 'self':
            prompt_episode = target_episode
        else:
            prompt_episode = self._sample_other_episode(target_episode)

        prompt = self.prompt_bank[prompt_episode].copy()

        if self.prompt_mode == 'shuffled':
            # keep the poses, destroy the temporal structure
            prompt = prompt[np.random.permutation(len(prompt))]
        return prompt

    def _sample_other_episode(self, target_episode: int) -> int:
        candidates = self.prompt_episodes
        if len(candidates) <= 1:
            # single-episode dataset: nothing else to draw from
            return int(candidates[0]) if len(candidates) else int(target_episode)
        while True:
            pick = int(candidates[np.random.randint(len(candidates))])
            if pick != target_episode:
                return pick

    # ------------------------------------------------------------------
    # dataset API
    # ------------------------------------------------------------------
    def get_validation_dataset(self):
        val_set = super().get_validation_dataset()
        # the parent swapped in a sampler over replay_buffer_val -> rebuild the
        # prompt bank / lookup so validation prompts come from validation episodes
        val_set._setup_prompts(self.replay_buffer_val, val_set.train_mask)
        val_set.external_prompt_bank = self.external_prompt_bank
        return val_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample, aug_idx = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, aug_idx)
        data['obs']['demo_traj'] = self.get_prompt(idx).astype(np.float32)
        return dict_apply(data, torch.from_numpy)
