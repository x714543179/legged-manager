from __future__ import annotations

import torch
from isaacgym.torch_utils import quat_apply

from legged_gym.utils.math import wrap_to_pi
from .manager_base import ManagerBase


class CommandManager(ManagerBase):
    """Owns command resampling and heading-command post processing."""

    def __init__(self, env, cfg=None):
        super().__init__(env, cfg)
        self.commands = env.commands

    def reset(self, env_ids):
        self.resample(env_ids)

    def compute(self):
        env = self.env
        interval = int(env.cfg.commands.resampling_time / env.dt)
        if interval > 0:
            env_ids = (env.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
            self.resample(env_ids)
        self._update_heading_command()

    def resample(self, env_ids):
        if len(env_ids) == 0:
            return
        if self._terms:
            for term in self._terms:
                if term.mode in (None, "resample"):
                    self._call_term(term, env_ids)
        else:
            self.env._resample_commands(env_ids)
        self.commands = self.env.commands

    def _update_heading_command(self):
        env = self.env
        if not env.cfg.commands.heading_command:
            return
        forward = quat_apply(env.base_quat, env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        env.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(env.commands[:, 3] - heading), -1.0, 1.0)
