from __future__ import annotations

import torch

from .manager_base import ManagerBase


class ObservationManager(ManagerBase):
    """Computes actor and critic observations and maintains observation history."""

    def __init__(self, env, cfg=None):
        super().__init__(env, cfg)
        self.noise_scale_vec = None

    def compute(self):
        if self._terms:
            obs_terms = [term for term in self._terms if term.mode in (None, "policy")]
            privileged_terms = [term for term in self._terms if term.mode == "privileged"]
            if not obs_terms:
                raise RuntimeError("ObservationManager requires at least one policy observation term.")
            self.env.obs_buf = torch.cat([self._call_term(term) for term in obs_terms], dim=-1)
            if hasattr(self.env, "_post_process_observations"):
                self.env._post_process_observations()
            if privileged_terms:
                self.env.privileged_obs_buf = torch.cat([self._call_term(term) for term in privileged_terms], dim=-1)
            elif hasattr(self.env, "_compute_privileged_observations"):
                self.env.privileged_obs_buf = self.env._compute_privileged_observations()
        else:
            self.env._compute_observations_impl()
        return self.env.obs_buf, self.env.privileged_obs_buf

    def compute_noise_scale_vec(self):
        """Build policy observation noise scales from per-term noise configs.

        This follows IsaacLab's observation-term style: noise belongs to the
        observation term that produces the tensor, so changing term order or
        width does not require editing hard-coded observation indices.
        """
        if not self._terms or not any(term.noise is not None for term in self._terms):
            return None

        noise_terms = []
        for term in self._policy_terms():
            term_value = self._call_term(term)
            noise_terms.append(self._noise_for_term(term, term_value))
        return torch.cat(noise_terms, dim=-1)

    def update_history_before_step(self):
        env = self.env
        if env.obs_hist_buf.numel() == 0:
            return
        env.obs_hist_buf = torch.cat((env.obs_hist_buf[:, env.num_obs :], env.obs_buf), dim=-1)

    def finalize(self):
        env = self.env
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(env.obs_buf, -clip_obs, clip_obs)
        if env.privileged_obs_buf is not None:
            env.privileged_obs_buf = torch.clip(env.privileged_obs_buf, -clip_obs, clip_obs)
        return env.obs_buf, env.privileged_obs_buf

    def _policy_terms(self):
        return [term for term in self._terms if term.mode in (None, "policy")]

    def _noise_for_term(self, term, term_value):
        if term.noise is None:
            return torch.zeros_like(term_value[0])

        noise_cfg = term.noise
        if callable(noise_cfg):
            noise = noise_cfg(self.env, term_value)
        elif isinstance(noise_cfg, str):
            noise = getattr(self.env, noise_cfg)
        elif isinstance(noise_cfg, (int, float)):
            noise = float(noise_cfg)
        else:
            noise = noise_cfg

        if torch.is_tensor(noise):
            noise = noise.to(device=self.env.device, dtype=term_value.dtype)
        else:
            noise = torch.as_tensor(noise, device=self.env.device, dtype=term_value.dtype)

        if noise.ndim == 0:
            return torch.ones_like(term_value[0]) * noise
        return torch.broadcast_to(noise, term_value[0].shape).clone()
