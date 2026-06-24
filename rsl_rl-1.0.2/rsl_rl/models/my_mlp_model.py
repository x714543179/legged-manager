from __future__ import annotations

from .model_base import Model_Base

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import  HiddenState
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable

class MyMLPModel(Model_Base):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)
        #encoder
        inputdim = sum(self.obs_dim[g] for g in obs_groups[obs_set])
        self.mlp = MLP(inputdim,output_dim, backbone_cfg.get("hidden_dims", []), backbone_cfg.get("activation", "relu"))

         
                 

    def forward(
            self,
            obs: TensorDict,
            masks: torch.Tensor | None = None,
            hidden_state: HiddenState = None,
            train_mode: bool = False,
        ) -> dict[str, torch.Tensor]:
            obs = super().forward(obs, masks, hidden_state, train_mode) # 这个是norm后的

             
            backbone_output = {}  # 用于存储 encoder 输出和 decoder 输出
            actions = self.mlp(
                torch.cat([obs[g] for g in self.obs_groups], dim=-1)
            )
            backbone_output["actions"] = actions
            return backbone_output
