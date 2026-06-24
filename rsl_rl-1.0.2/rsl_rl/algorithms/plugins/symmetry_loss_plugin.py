from __future__ import annotations

import torch

from rsl_rl.algorithms.plugins.base import PPOPlugin


class SymmetryLossPlugin(PPOPlugin):
    """Mirror-action regularization for the PPO plugin API."""

    def __init__(self, obs_permutation, act_permutation, frame_stack=1, sym_coef=1.0, obs_group="actor"):
        self.obs_permutation = obs_permutation
        self.act_permutation = act_permutation
        self.frame_stack = frame_stack
        self.sym_coef = sym_coef
        self.obs_group = obs_group
        self.obs_perm_mat = None
        self.obs_hist_perm_mat = None
        self.act_perm_mat = None

    def on_init(self, ppo, env) -> None:
        device = ppo.device
        self.actor_obs_groups = list(getattr(ppo.actor.backbone, "obs_groups", [self.obs_group]))
        self.obs_perm_mat = self._build_perm_mat(self.obs_permutation, device)
        self.act_perm_mat = self._build_perm_mat(self.act_permutation, device)
        obs_hist_permutation = []
        obs_dim = len(self.obs_permutation)
        for frame_idx in range(self.frame_stack):
            offset = frame_idx * obs_dim
            for perm in self.obs_permutation:
                sign = -1.0 if perm < 0 else 1.0
                obs_hist_permutation.append(sign * (abs(perm) + offset))
        self.obs_hist_perm_mat = self._build_perm_mat(obs_hist_permutation, device)

    def on_per_batch_extra_loss(self, ppo, batch):
        obs = batch.observations
        actor_obs = torch.cat([obs[group] for group in self.actor_obs_groups], dim=-1)
        history = obs["history"]
        if actor_obs.shape[-1] != self.obs_perm_mat.shape[0]:
            raise ValueError(
                f"obs_permutation length {self.obs_perm_mat.shape[0]} does not match obs dim {actor_obs.shape[-1]}."
            )
        if history.shape[-1] != self.obs_hist_perm_mat.shape[0]:
            raise ValueError(
                f"history permutation length {self.obs_hist_perm_mat.shape[0]} does not match history dim {history.shape[-1]}."
            )

        mirror_obs = obs.clone()
        mirrored_actor_obs = torch.matmul(actor_obs, self.obs_perm_mat)
        self._write_actor_obs_groups(mirror_obs, mirrored_actor_obs)
        mirror_obs["history"] = torch.matmul(history, self.obs_hist_perm_mat)

        action_mean = ppo.actor.output_mean.clone()
        mirror_actions = ppo.actor(mirror_obs, stochastic_output=False, train_mode=True)["actions"]
        mapped_mirror_actions = torch.matmul(mirror_actions, self.act_perm_mat)
        sym_loss = (action_mean - mapped_mirror_actions).pow(2).mean() * self.sym_coef
        return {"symmetry": sym_loss}

    def _write_actor_obs_groups(self, obs, actor_obs):
        start = 0
        for group in self.actor_obs_groups:
            width = obs[group].shape[-1]
            obs[group] = actor_obs[..., start : start + width]
            start += width
        if self.obs_group in obs and obs[self.obs_group].shape[-1] == actor_obs.shape[-1]:
            obs[self.obs_group] = actor_obs

    @staticmethod
    def _build_perm_mat(permutation, device):
        perm_len = len(permutation)
        perm_mat = torch.zeros((perm_len, perm_len), device=device)
        for col_idx, perm in enumerate(permutation):
            row_idx = int(abs(perm))
            if row_idx >= perm_len:
                raise ValueError(f"Permutation index {row_idx} out of range for length {perm_len}.")
            sign = -1.0 if perm < 0 else 1.0
            perm_mat[row_idx, col_idx] = sign
        return perm_mat
