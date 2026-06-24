from __future__ import annotations

import torch

from .manager_base import ManagerBase


class ObservationManager(ManagerBase):
    """Computes actor and critic observations and maintains observation history."""

    def __init__(self, env, cfg=None):
        super().__init__(env, cfg)
        self.noise_scale_vec = None
        self._group_terms = self._resolve_group_terms(cfg)
        self.obs_groups = self._build_obs_groups()
        self._flat_group_terms = [
            (name, term)
            for terms in self._group_terms.values()
            for name, term in terms
        ]
        self._policy_term_widths = []
        self._privileged_term_widths = []
        self._policy_term_names = []
        self._privileged_term_names = []

    @property
    def active_terms(self):
        if self._flat_group_terms:
            return [name for name, _ in self._flat_group_terms]
        return super().active_terms

    def compute(self):
        if self._group_terms:
            obs_terms = self._terms_for_obs_set("actor")
            privileged_terms = self._terms_for_obs_set("critic")
            if not obs_terms:
                raise RuntimeError("ObservationManager requires at least one policy observation term.")

            policy_values = [(name, self._call_term(term)) for name, term in obs_terms]
            self.env.obs_buf = torch.cat([value for _, value in policy_values], dim=-1)
            self._set_policy_term_buffers(policy_values)
            if hasattr(self.env, "_post_process_observations"):
                self.env._post_process_observations()
                self._sync_policy_term_buffers_from_obs_buf()

            if privileged_terms:
                privileged_values = [(name, self._call_term(term)) for name, term in privileged_terms]
                self.env.privileged_obs_buf = torch.cat([value for _, value in privileged_values], dim=-1)
                self._set_privileged_term_buffers(privileged_values)
            elif hasattr(self.env, "_compute_privileged_observations"):
                self.env.privileged_obs_buf = self.env._compute_privileged_observations()
                self.env.privileged_obs_term_bufs = {}
                self._privileged_term_names = []
                self._privileged_term_widths = []
        else:
            self.env._compute_observations_impl()
        return self.env.obs_buf, self.env.privileged_obs_buf

    def compute_noise_scale_vec(self):
        """Build policy observation noise scales from per-term noise configs.

        This follows IsaacLab's observation-term style: noise belongs to the
        observation term that produces the tensor, so changing term order or
        width does not require editing hard-coded observation indices.
        """
        if not self._group_terms or not any(term.noise is not None for _, term in self._terms_for_obs_set("actor")):
            return None

        noise_terms = []
        for _, term in self._terms_for_obs_set("actor"):
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
        self._sync_policy_term_buffers_from_obs_buf()
        self._sync_privileged_term_buffers_from_obs_buf()
        return env.obs_buf, env.privileged_obs_buf

    def _policy_terms(self):
        return [term for term in self._terms if term.mode in (None, "policy")]

    def _policy_terms_with_names(self):
        return [
            (name, term)
            for name, term in zip(self._term_names, self._terms)
            if term.mode in (None, "policy")
        ]

    def _privileged_terms_with_names(self):
        return [
            (name, term)
            for name, term in zip(self._term_names, self._terms)
            if term.mode == "privileged"
        ]

    def _set_policy_term_buffers(self, values):
        self.env.obs_term_bufs = {name: value for name, value in values}
        self._policy_term_names = [name for name, _ in values]
        self._policy_term_widths = [value.shape[-1] for _, value in values]

    def _set_privileged_term_buffers(self, values):
        self.env.privileged_obs_term_bufs = {name: value for name, value in values}
        self._privileged_term_names = [name for name, _ in values]
        self._privileged_term_widths = [value.shape[-1] for _, value in values]

    def _sync_policy_term_buffers_from_obs_buf(self):
        if not self._policy_term_names or self.env.obs_buf is None:
            return
        chunks = torch.split(self.env.obs_buf, self._policy_term_widths, dim=-1)
        self.env.obs_term_bufs = dict(zip(self._policy_term_names, chunks))

    def _sync_privileged_term_buffers_from_obs_buf(self):
        if (
            not self._privileged_term_names
            or self.env.privileged_obs_buf is None
        ):
            return
        chunks = torch.split(self.env.privileged_obs_buf, self._privileged_term_widths, dim=-1)
        self.env.privileged_obs_term_bufs = dict(zip(self._privileged_term_names, chunks))

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

    def _resolve_group_terms(self, cfg):
        groups = self._resolve_explicit_groups(cfg)
        if groups:
            return groups
        if not self._terms:
            return {}
        return {
            "actor": [
                (name, term)
                for name, term in zip(self._term_names, self._terms)
                if term.mode in (None, "policy")
            ],
            "critic": [
                (name, term)
                for name, term in zip(self._term_names, self._terms)
                if term.mode == "privileged"
            ],
        }

    def _resolve_explicit_groups(self, cfg):
        if cfg is None:
            return {}
        groups = {}
        for group_name, group_cfg in self._iter_cfg_items(cfg):
            if not self._is_obs_group(group_cfg):
                continue
            group_terms = []
            for term_name, value in self._iter_cfg_items(group_cfg):
                term_cfg = self._coerce_term_cfg(value)
                if term_cfg is None or not term_cfg.enabled:
                    continue
                group_terms.append((f"{group_name}_{term_name}", term_cfg))
            if group_terms:
                groups[group_name] = group_terms
        return groups

    @staticmethod
    def _is_obs_group(value):
        group_cls = value if isinstance(value, type) else value.__class__
        for base_cls in getattr(group_cls, "__mro__", ()):
            if base_cls.__name__ == "ObsGroup":
                return True
        return False

    def _build_obs_groups(self):
        if self._group_terms:
            return {group_name: [term_name for term_name, _ in terms] for group_name, terms in self._group_terms.items()}
        if self._terms:
            return {
                "actor": [name for name, term in zip(self._term_names, self._terms) if term.mode in (None, "policy")],
                "critic": [name for name, term in zip(self._term_names, self._terms) if term.mode == "privileged"],
            }
        return {}

    def _terms_for_obs_set(self, obs_set):
        return self._group_terms.get(obs_set, [])
