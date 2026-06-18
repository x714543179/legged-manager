from __future__ import annotations

import torch

from .manager_base import ManagerBase


class ActionManager(ManagerBase):
    """Applies action clipping and optional environment-specific action delay."""

    def __init__(self, env, cfg=None):
        super().__init__(env, cfg)
        self.raw_actions = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device)
        self.actions = torch.zeros_like(self.raw_actions)

    def process(self, actions: torch.Tensor) -> torch.Tensor:
        clip_actions = self.env.cfg.normalization.clip_actions
        self.raw_actions = torch.clip(actions, -clip_actions, clip_actions).to(self.env.device)
        if self._terms:
            self.actions = self.raw_actions
            for term in self._terms:
                if term.mode in (None, "policy"):
                    self.actions = self._call_term(term, self.actions)
        elif hasattr(self.env, "_process_actions"):
            self.actions = self.env._process_actions(self.raw_actions)
        else:
            self.actions = self.raw_actions
        self.env.actions = self.actions
        return self.actions

    def apply(self, mode: str, actions: torch.Tensor) -> torch.Tensor:
        processed_actions = actions
        for term in self._terms:
            if term.mode == mode:
                processed_actions = self._call_term(term, processed_actions)
        return processed_actions

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
