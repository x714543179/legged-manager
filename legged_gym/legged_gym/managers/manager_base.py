from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ManagerTermCfg:
    """Configuration for one manager term.

    The shape follows IsaacLab's manager-term idea: a term names a callable and
    optional parameters; specialized managers may use extra fields such as
    ``scale`` or ``mode``.
    """

    func: Callable[..., Any] | str
    params: Dict[str, Any] = field(default_factory=dict)
    scale: float = 1.0
    weight: Optional[float] = None
    mode: Optional[str] = None
    enabled: bool = True
    clip: Optional[tuple[float, float]] = None
    noise: Any = None
    use_dt: bool = True
    env_arg: bool = False


class ObsGroup:
    """Marker base for grouped observation terms.

    Subclasses are parsed by :class:`ObservationManager` as ordered observation
    groups. The group name defaults to the class attribute name in the config.
    """

    mode: Optional[str] = None


class ManagerBase:
    """Small manager base class for Isaac Gym environments."""

    def __init__(self, env, cfg: Any = None):
        self.env = env
        self.cfg = cfg
        self._terms: List[ManagerTermCfg] = []
        self._term_names: List[str] = []
        if self.has_terms(cfg):
            self._terms, self._term_names = self._resolve_terms(cfg)

    @property
    def active_terms(self) -> List[str]:
        return list(self._term_names)

    def reset(self, env_ids):
        """Reset per-env state owned by this manager."""

    def serialize(self) -> Dict[str, Any]:
        return {"active_terms": self.active_terms}

    def _resolve_terms(self, cfg: Any) -> tuple[List[ManagerTermCfg], List[str]]:
        items = self._iter_cfg_items(cfg)

        terms: List[ManagerTermCfg] = []
        names: List[str] = []
        for name, value in items:
            term_cfg = self._coerce_term_cfg(value)
            if term_cfg is None or not term_cfg.enabled:
                continue
            terms.append(term_cfg)
            names.append(name)
        return terms, names

    def _coerce_term_cfg(self, value: Any) -> Optional[ManagerTermCfg]:
        if isinstance(value, ManagerTermCfg):
            return value
        if isinstance(value, type):
            return None
        if callable(value) or isinstance(value, str):
            return ManagerTermCfg(func=value)
        if isinstance(value, dict):
            if value.get("enabled", True) is False:
                return None
            if "func" in value:
                return ManagerTermCfg(**value)
        return None

    @staticmethod
    def has_terms(cfg: Any) -> bool:
        if cfg is None:
            return False
        for _, value in ManagerBase._iter_cfg_items(cfg):
            if isinstance(value, type):
                continue
            if isinstance(value, ManagerTermCfg) or callable(value) or isinstance(value, str):
                return True
            if isinstance(value, dict) and "func" in value and value.get("enabled", True):
                return True
        return False

    @staticmethod
    def _iter_cfg_items(cfg: Any):
        if isinstance(cfg, dict):
            return cfg.items()
        items = {}
        cls = cfg if isinstance(cfg, type) else cfg.__class__
        for base_cls in reversed(getattr(cls, "__mro__", ())):
            if base_cls is object:
                continue
            for name, value in base_cls.__dict__.items():
                if not name.startswith("_"):
                    items[name] = value
        if not isinstance(cfg, type) and hasattr(cfg, "__dict__"):
            for name, value in cfg.__dict__.items():
                if not name.startswith("_"):
                    items[name] = value
        if not items:
            items.update((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith("_"))
        return items.items()

    def _resolve_callable(self, func: Callable[..., Any] | str) -> Callable[..., Any]:
        if callable(func):
            return func
        if hasattr(self.env, func):
            return getattr(self.env, func)
        if "." in func:
            module_name, attr_name = func.rsplit(".", 1)
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        raise AttributeError(f"Manager term callable '{func}' was not found")

    def _call_term(self, term_cfg: ManagerTermCfg, *args, **kwargs):
        func = self._resolve_callable(term_cfg.func)
        params = dict(term_cfg.params)
        params.update(kwargs)
        if term_cfg.env_arg:
            return func(self.env, *args, **params)
        return func(*args, **params)

    @staticmethod
    def _iter_env_ids(env_ids) -> Iterable[int]:
        if hasattr(env_ids, "tolist"):
            return env_ids.tolist()
        return env_ids
