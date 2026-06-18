from __future__ import annotations

import torch

from legged_gym.utils.helpers import class_to_dict
from .manager_base import ManagerBase, ManagerTermCfg


class RewardManager(ManagerBase):
    """Computes scaled reward terms and episode sums."""

    def __init__(self, env, cfg=None):
        if not self.has_terms(cfg):
            cfg = self._legacy_reward_terms(env)
        super().__init__(env, cfg)
        self._raw_scales = {name: term.scale for name, term in zip(self._term_names, self._terms)}
        self.episode_sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
            for name in self._term_names
        }

    @property
    def reward_scales(self):
        return {name: self._term_scale(term) for name, term in zip(self._term_names, self._terms)}

    def compute(self):
        env = self.env
        env.rew_buf[:] = 0.0
        for name, term in zip(self._term_names, self._terms):
            if name == "termination":
                continue
            rew = self._call_term(term) * self._term_scale(term)
            env.rew_buf += rew
            self.episode_sums[name] += rew
        if env.cfg.rewards.only_positive_rewards:
            env.rew_buf[:] = torch.clip(env.rew_buf[:], min=0.0)
        if "termination" in self._term_names:
            index = self._term_names.index("termination")
            term = self._terms[index]
            rew = self._call_term(term) * self._term_scale(term)
            env.rew_buf += rew
            self.episode_sums["termination"] += rew
        return env.rew_buf

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return {}
        episode = {}
        for key in self.episode_sums.keys():
            episode["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.env.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        return episode

    @staticmethod
    def _legacy_reward_terms(env):
        terms = {}
        reward_scales = class_to_dict(env.cfg.rewards.scales)
        for name, scale in reward_scales.items():
            if scale == 0:
                continue
            terms[name] = ManagerTermCfg(func=f"_reward_{name}", scale=scale, use_dt=True)
        return terms

    def _term_scale(self, term: ManagerTermCfg):
        scale = term.scale
        if term.weight is not None:
            scale = term.weight
        if term.use_dt:
            scale *= self.env.dt
        return scale
