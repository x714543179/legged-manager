from __future__ import annotations

import torch

from .manager_base import ManagerBase


class TerminationManager(ManagerBase):
    """Computes reset and timeout buffers."""

    def compute(self):
        env = self.env
        if self._terms:
            reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for term in self._terms:
                reset |= self._call_term(term).bool()
            env.reset_buf = reset
            env.time_out_buf = env.episode_length_buf > env.max_episode_length
            env.reset_buf |= env.time_out_buf
        else:
            env._check_termination_impl()
        return env.reset_buf
