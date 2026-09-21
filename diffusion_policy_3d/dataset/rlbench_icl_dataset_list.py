"""Multi-task version of :class:`RLBenchICLDataset`.

Same idea as ``RLBenchDatasetList`` (concatenate the per-task datasets), with
two differences:

* every sub-dataset is an ``RLBenchICLDataset``, so each sample carries a demo
  trajectory drawn from *another episode of the same task*;
* the task list and the per-task dataset overrides are configurable, so the
  key multi-task ablation (language off -> the only thing identifying the task
  is the demo trajectory) does not require editing this file.

After construction every sub-dataset is also handed the prompt banks of the
*other* tasks, so ``prompt_mode='wrong_task'`` works as a control condition.
"""

import copy
from typing import Dict

import hydra
import numpy as np
import torch
import yaml
from termcolor import cprint

from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.rlbench_icl_dataset import RLBenchICLDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


DEFAULT_TASK_LIST = [
    'meat_off_grill',
    'put_money_in_safe',
    'place_wine_at_rack_location',
    'reach_and_drag',
    'stack_blocks',
    'close_jar',
    'light_bulb_in',
    'put_groceries_in_cupboard',
    'place_shape_in_shape_sorter',
    'insert_onto_square_peg',
    'stack_cups',
    'place_cups',
    'turn_tap',
]


class RLBenchICLDatasetList(BaseDataset):
    def __init__(self,
                 root_dir=None,
                 task_list=None,
                 task_config_dir="config/task/rlbench",
                 prompt_length=16,
                 prompt_mode='other_episode',
                 dataset_overrides=None):
        self.task_list = list(task_list) if task_list is not None else list(DEFAULT_TASK_LIST)
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode

        self.task_id_to_name = {}
        self.task_name_to_dataset = {}

        self.dataset_list = []
        self.global_idx_to_task_id_local_idx = {}

        # user overrides applied on top of every single-task dataset config
        overrides = dict(dataset_overrides) if dataset_overrides is not None else {}

        start = 0
        for task_idx, task_name in enumerate(self.task_list):

            # read yaml
            with open(f"{task_config_dir}/{task_name}.yaml", 'r') as stream:
                dataset_cfg = yaml.safe_load(stream)["dataset"]

            if root_dir is not None:
                dataset_cfg["root_dir"] = f"{root_dir}/{task_name}/all_variations"  # overrides all single-task configs
            else:
                dataset_cfg["root_dir"] = dataset_cfg["root_dir"].replace("${task.task_name}", task_name)

            # swap in the ICL dataset and its prompt settings
            dataset_cfg["_target_"] = \
                "diffusion_policy_3d.dataset.rlbench_icl_dataset.RLBenchICLDataset"
            dataset_cfg["prompt_length"] = self.prompt_length
            dataset_cfg["prompt_mode"] = self.prompt_mode
            dataset_cfg.update(overrides)

            cprint(f"[ICLDatasetList] {task_idx}: {task_name}", "cyan")
            dataset = hydra.utils.instantiate(dataset_cfg)
            assert isinstance(dataset, RLBenchICLDataset)
            self.dataset_list.append(dataset)

            self.task_id_to_name[task_idx] = task_name
            self.task_name_to_dataset[task_name] = dataset

            # global index to (task_id, local index)
            dataset_length = len(dataset)
            for sample_idx in range(dataset_length):
                self.global_idx_to_task_id_local_idx[start + sample_idx] = (task_idx, sample_idx)
            start += dataset_length

        self._share_cross_task_prompts(self.dataset_list)

    @staticmethod
    def _share_cross_task_prompts(dataset_list):
        """Give every task the prompt banks of all the other tasks."""
        if len(dataset_list) < 2:
            return
        for task_idx, dataset in enumerate(dataset_list):
            others = [d.prompt_bank for j, d in enumerate(dataset_list) if j != task_idx]
            dataset.set_external_prompt_bank(np.concatenate(others, axis=0))

    def get_validation_dataset(self):
        val_set_list = copy.copy(self)
        val_set_list.dataset_list = [d.get_validation_dataset() for d in self.dataset_list]
        val_set_list.task_name_to_dataset = {
            name: val_set_list.dataset_list[idx]
            for idx, name in self.task_id_to_name.items()
        }
        self._share_cross_task_prompts(val_set_list.dataset_list)

        # the validation buffers have their own episode counts -> rebuild the map
        val_set_list.global_idx_to_task_id_local_idx = {}
        start = 0
        for task_idx, dataset in enumerate(val_set_list.dataset_list):
            for sample_idx in range(len(dataset)):
                val_set_list.global_idx_to_task_id_local_idx[start + sample_idx] = (task_idx, sample_idx)
            start += len(dataset)
        return val_set_list

    def get_normalizer(self, mode='limits', **kwargs):
        action_all = []
        agent_pos_all = []
        for dataset in self.dataset_list:
            action_all.append(dataset.replay_buffer['action'][:, :3])
            agent_pos_all.append(dataset.replay_buffer['state'][..., :][:, :3])

        action_all = np.concatenate(action_all, axis=0)
        agent_pos_all = np.concatenate(agent_pos_all, axis=0)

        data = {
            'action': action_all,
            'agent_pos': agent_pos_all,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.global_idx_to_task_id_local_idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_idx, sample_idx = self.global_idx_to_task_id_local_idx[idx]
        select_dataset = self.dataset_list[task_idx]
        return select_dataset.__getitem__(sample_idx)
