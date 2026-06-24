# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
from dataclasses import MISSING

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_callable, unpad_trajectories


class Model_Base(nn.Module):
    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg  # 收集所有额外的关键字参数
    ) -> None:
        
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.output_dim = output_dim
        # Observation normalization
        self.obs_normalization = backbone_cfg.get("obs_normalization", False)
        if self.obs_normalization:
            self.obs_normalizers = nn.ModuleDict(
                {g: EmpiricalNormalization(obs[g].shape[-1]) for g in obs_groups[obs_set]})
        else:
            self.obs_normalizers = None  # [FIX 1] 防止 update_normalization 报 AttributeError

        
         
    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        if self.obs_normalization:
            obs_normed = obs.clone()  # [FIX 2] 不改原始 TensorDict，避免 actor/critic 共享 obs 时互相污染
            for g in self.obs_groups:
                obs_normed[g] = self.obs_normalizers[g](obs[g])
        return obs_normed

    def _compute_aux_losses(
            self,
            obs: TensorDict,
            named_latents: dict[str, torch.Tensor],
            active_latent: torch.Tensor,
        ):
        pass

 # ------------------------------------------------------------------
    # Normalisation update
    # ------------------------------------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        """Update per-group running normalisation statistics."""
        if self.obs_normalizers is not None:
            for g in self.obs_groups:
                self.obs_normalizers[g].update(obs[g])

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = {}
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim[obs_group] = obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim
    

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

   