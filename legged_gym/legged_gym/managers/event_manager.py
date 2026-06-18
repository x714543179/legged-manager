from __future__ import annotations

from .manager_base import ManagerBase


class EventManager(ManagerBase):
    """Runs non-RL side effects such as pushes, terrain height sampling, and DR."""

    def apply(self, mode: str, env_ids=None):
        hook_name = f"_apply_{mode}_events"
        result = None
        mode_terms = [term_cfg for term_cfg in self._terms if term_cfg.mode == mode]
        if not mode_terms and hasattr(self.env, hook_name):
            result = getattr(self.env, hook_name)(env_ids)

        for term_cfg in mode_terms:
            self._call_term(term_cfg, env_ids=env_ids)
        return result

    def reset(self, env_ids):
        self.apply("reset", env_ids)
