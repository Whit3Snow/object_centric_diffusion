"""SimpleDP3 conditioned on an object-trajectory prompt.

    h_o = E_o(s_t)                  # existing SPOT observation encoder
    h_d = E_d(tau_demo)             # TrajectoryPromptEncoder
    global_cond = [h_o ; h_d]

Everything else (noise scheduler, UNet, losses, normalization) is unchanged
w.r.t. :class:`SimpleDP3`; only the conditioning vector grows by ``prompt_dim``.

The prompt is read from ``obs_dict['demo_traj']`` (shape ``[B, K, 7]``), which
is where :class:`RLBenchICLDataset` puts it, so ``tools/train_dp3.py`` needs no
changes. ``predict_action`` also accepts an explicit ``demo_traj=`` argument,
which is what the RLBench rollout wrapper uses at evaluation time.
"""

import inspect
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d_progress import ConditionalUnet1D_progress
from diffusion_policy_3d.model.trajectory_prompt_encoder import TrajectoryPromptEncoder
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3, custom_normalize, custom_unnormalize


class PromptSimpleDP3(SimpleDP3):
    def __init__(self,
                 *args,
                 prompt_length=16,
                 prompt_state_dim=7,
                 prompt_dim=128,
                 prompt_hidden_dim=256,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # resolve the parent's arguments (including its defaults) so the UNet
        # can be rebuilt with the enlarged conditioning dimension
        bound = inspect.signature(SimpleDP3.__init__).bind_partial(self, *args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments

        if "cross_attention" in self.condition_type:
            raise NotImplementedError(
                "PromptSimpleDP3 only supports global (film) conditioning for now")
        if not self.obs_as_global_cond:
            raise NotImplementedError(
                "PromptSimpleDP3 requires obs_as_global_cond=True")

        self.prompt_encoder = TrajectoryPromptEncoder(
            prompt_length=prompt_length,
            state_dim=prompt_state_dim,
            prompt_dim=prompt_dim,
            hidden_dim=prompt_hidden_dim,
        )
        self.prompt_length = prompt_length
        self.prompt_state_dim = prompt_state_dim
        self.prompt_dim = prompt_dim

        # [h_obs ; h_demo]
        global_cond_dim = self.obs_feature_dim * self.n_obs_steps + prompt_dim

        input_dim = self.action_dim
        model_cls = ConditionalUnet1D
        if self.use_progress:
            input_dim = input_dim - 1  # handle progress/gripper separately
            model_cls = ConditionalUnet1D_progress

        self.model = model_cls(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=p['diffusion_step_embed_dim'],
            down_dims=p['down_dims'],
            kernel_size=p['kernel_size'],
            n_groups=p['n_groups'],
            condition_type=p['condition_type'],
            use_down_condition=p['use_down_condition'],
            use_mid_condition=p['use_mid_condition'],
            use_up_condition=p['use_up_condition'],
        )
        cprint(f"[PromptSDP3] global_cond_dim: {global_cond_dim} "
               f"(obs {self.obs_feature_dim} x {self.n_obs_steps} + prompt {prompt_dim})", "yellow")

    # ------------------------------------------------------------------
    # prompt handling
    # ------------------------------------------------------------------
    def _split_prompt(self, obs_dict: Dict[str, torch.Tensor], demo_traj=None):
        """Return (obs without the prompt, prompt tensor)."""
        obs_dict = dict(obs_dict)
        popped = obs_dict.pop('demo_traj', None)
        if demo_traj is None:
            demo_traj = popped
        assert demo_traj is not None, \
            "demo_traj missing: pass it explicitly or put it in obs_dict['demo_traj']"

        if isinstance(demo_traj, np.ndarray):
            demo_traj = torch.from_numpy(demo_traj)
        demo_traj = demo_traj.to(device=self.device, dtype=self.dtype)
        if demo_traj.dim() == 2:  # (K, 7) -> (1, K, 7)
            demo_traj = demo_traj.unsqueeze(0)
        return obs_dict, demo_traj

    def encode_prompt(self, demo_traj: torch.Tensor) -> torch.Tensor:
        """(B, K, 7) -> (B, prompt_dim), with the same xyz/quaternion split as
        the observation normalization."""
        ndemo = custom_normalize(demo_traj, self.normalizer['agent_pos'], key=None)
        return self.prompt_encoder(ndemo)

    # ========= inference  ============
    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       target=None,
                       demo_traj=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "agent_pos"; the prompt is taken from
            obs_dict["demo_traj"] unless demo_traj is given explicitly.
        result: includes "action"
        """
        obs_dict, demo_traj = self._split_prompt(obs_dict, demo_traj)

        # normalize quaternion seperately
        nobs = custom_normalize(obs_dict, self.normalizer, key='agent_pos')

        if 'point_cloud' in nobs:
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        value = next(iter(nobs.values()))

        # !! Important
        if value.shape[1] > 1:
            value = value[:, 0].unsqueeze(1)

        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        obs_feature = nobs_features.reshape(B, -1)

        # object trajectory prompt
        prompt_feature = self.encode_prompt(demo_traj)
        assert prompt_feature.shape[0] == B, \
            f"prompt batch {prompt_feature.shape[0]} != obs batch {B}"
        global_cond = torch.cat([obs_feature, prompt_feature], dim=-1)

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = custom_unnormalize(naction_pred, self.normalizer['action'], key=None)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,           # [B, n_action_steps, 8]
            'action_pred': action_pred, # [B, horizon, 8]
        }

        if target is not None:
            action_pred = action_pred.to(target.device)
            mse = torch.nn.functional.mse_loss(action_pred, target)
            result["loss"] = mse.item()
        return result

    # ========= training  ============
    def compute_loss(self, batch):
        obs_dict, demo_traj = self._split_prompt(batch['obs'])

        # normalize quaternion seperately
        nobs = custom_normalize(obs_dict, self.normalizer, key='agent_pos')
        nactions = custom_normalize(batch['action'], self.normalizer['action'], key=None)

        if 'point_cloud' in nobs:
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]

        local_cond = None
        trajectory = nactions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        obs_feature = nobs_features.reshape(batch_size, -1)

        # object trajectory prompt
        prompt_feature = self.encode_prompt(demo_traj)
        global_cond = torch.cat([obs_feature, prompt_feature], dim=-1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # forward diffusion process
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory,
                          timestep=timesteps,
                          local_cond=local_cond,
                          global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if self.use_progress:
            loss_trans = F.mse_loss(pred[..., :3], target[..., :3], reduction='none')
            loss_ori = F.mse_loss(pred[..., 3:-1], target[..., 3:-1], reduction='none')
            loss_gripper = F.binary_cross_entropy(pred[..., -1:], target[..., -1:], reduction='none')

            loss = torch.concat([loss_trans, loss_ori, loss_gripper], dim=-1)
            loss[..., -1:] *= 0.1
        else:
            loss = F.mse_loss(pred, target, reduction='none')

        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
        }
        return loss, loss_dict
