# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class ActorModel(nn.Module):
    """Actor shell that wraps a backbone network and a stochastic distribution head.

    The backbone is responsible for producing distribution inputs (for example,
    action mean logits), while this shell owns sampling, log-probability,
    entropy, and KL-related logic.
    """

    def __init__(self, backbone: nn.Module, output_dim: int, distribution_cfg: dict) -> None:
        super().__init__()
        if distribution_cfg is None:
            raise ValueError("ActorModel requires 'distribution_cfg'.")

        dist_cfg = copy.deepcopy(distribution_cfg)
        dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore

        self.backbone = backbone
        self.distribution: Distribution = dist_class(output_dim, **dist_cfg)

    @property
    def is_recurrent(self) -> bool:
        """Whether the wrapped backbone is recurrent."""
        return bool(getattr(self.backbone, "is_recurrent", False))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run backbone forward, update distribution, and return output dict.

        Returns:
            dict with keys:
            - ``"actions"``: sampled (stochastic) or deterministic action tensor
            - ``"extra"``: any additional outputs the backbone returned beyond ``"actions"``
              (e.g. ``"sub_latent"`` from :class:`~rsl_rl.models.SubEncoderMLPModel`)
        """
        backbone_output = self.backbone(
            obs,
            masks=masks,
            hidden_state=hidden_state,
            train_mode=train_mode,
        )
        
        # Support both dict-returning backbones (new) and tensor-returning ones (legacy)
        if isinstance(backbone_output, dict):
            dist_input = backbone_output["actions"]
            extra = {k: v for k, v in backbone_output.items() if k != "actions"}
        else:
            dist_input = backbone_output
            extra = {}

        self.distribution.update(dist_input)
        if stochastic_output:
            return {"actions": self.distribution.sample(), "extra": extra}
        return {"actions": self.distribution.deterministic_output(dist_input), "extra": extra}

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset hidden states through the backbone."""
        if hasattr(self.backbone, "reset"):
            self.backbone.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return hidden states from the backbone."""
        if hasattr(self.backbone, "get_hidden_state"):
            return self.backbone.get_hidden_state()
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states through the backbone."""
        if hasattr(self.backbone, "detach_hidden_state"):
            self.backbone.detach_hidden_state(dones)

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return current distribution parameters."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute output log-probability from the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return TorchScript export module from the backbone.

        For current default Gaussian distributions, deterministic output is
        identical to backbone output.
        """
        if not hasattr(self.backbone, "as_jit"):
            raise AttributeError("Backbone does not support JIT export via 'as_jit'.")
        return self.backbone.as_jit()

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return ONNX export module from the backbone.

        For current default Gaussian distributions, deterministic output is
        identical to backbone output.
        """
        if not hasattr(self.backbone, "as_onnx"):
            raise AttributeError("Backbone does not support ONNX export via 'as_onnx'.")
        return self.backbone.as_onnx(verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Forward normalization-stat updates to the backbone."""
        if hasattr(self.backbone, "update_normalization"):
            self.backbone.update_normalization(obs)
