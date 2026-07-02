from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .registry import resolve_backend, resolve_callable
from .term_cfg import PlotTermCfg


class NullPlotManager:
    enabled = False

    def step(self, *args, **kwargs):
        return None

    def close(self):
        return None


class PlotManager:
    """Collect plot terms during play and render them with a pluggable backend."""

    enabled = True

    def __init__(self, env, cfg, dt, backend=None, output_dir=None, show=None, max_steps=None, interval=None):
        self.env = env
        self.cfg = cfg
        self.dt = float(dt)
        self.output_dir = output_dir if output_dir is not None else getattr(cfg, "output_dir", None)
        self.show = bool(getattr(cfg, "show", True) if show is None else show)
        self.max_steps = _get_cfg_value(cfg, "max_steps", max_steps, default=1000)
        self.interval = max(1, int(_get_cfg_value(cfg, "interval", interval, default=1)))
        self.figures = list(getattr(cfg, "figures", []) or [])
        self._step_count = 0
        self._record_count = 0
        self._closed = False
        self._times = []
        self._series = defaultdict(list)
        self._terms = list(_iter_terms(getattr(cfg, "terms", None)))

        backend_name = backend if backend is not None else getattr(cfg, "backend", "matplotlib")
        backend_cls = resolve_backend(backend_name)
        self.backend = backend_cls()

    @property
    def series(self):
        return dict(self._series)

    @property
    def times(self):
        return list(self._times)

    def step(self, env=None, **kwargs):
        if self._closed:
            return None
        if self.max_steps is not None and self._record_count >= int(self.max_steps):
            self._step_count += 1
            return None
        if self._step_count % self.interval != 0:
            self._step_count += 1
            return None

        env = env if env is not None else self.env
        values = {}
        for _, term_cfg in self._terms:
            term_values = self._call_term(term_cfg, env, **kwargs)
            if term_values:
                values.update(_flatten_term_values(term_values))

        if values:
            self._times.append(self._step_count * self.dt)
            for key, value in values.items():
                self._series[key].append(value)
            self._record_count += 1
        self._step_count += 1
        return values

    def close(self):
        if self._closed:
            return None
        self._closed = True
        if not self._times:
            return None
        return self.backend.render(
            times=self._times,
            series=dict(self._series),
            figures=self.figures,
            output_dir=self.output_dir,
            show=self.show,
        )

    def _call_term(self, term_cfg, env, **kwargs):
        func = resolve_callable(term_cfg.func)
        params = dict(getattr(term_cfg, "params", {}) or {})
        return func(env, **params)


def create_plot_manager(env, cfg=None, args=None, dt=None, output_dir=None):
    cfg = cfg if cfg is not None else getattr(getattr(env, "cfg", None), "plots", None)
    args_enabled = bool(getattr(args, "plot", False)) if args is not None else False
    cfg_enabled = bool(getattr(cfg, "enabled", False)) if cfg is not None else False
    if cfg is None or not (args_enabled or cfg_enabled):
        return NullPlotManager()

    backend = getattr(args, "plot_backend", None) if args is not None else None
    max_steps = getattr(args, "plot_steps", None) if args is not None else None
    interval = getattr(args, "plot_interval", None) if args is not None else None
    args_output_dir = getattr(args, "plot_output_dir", None) if args is not None else None
    if args_output_dir is not None:
        output_dir = args_output_dir
    show = None
    if args is not None and getattr(args, "plot_no_show", False):
        show = False
    if dt is None:
        dt = getattr(env, "dt", 1.0)
    return PlotManager(
        env,
        cfg,
        dt,
        backend=backend,
        output_dir=output_dir,
        show=show,
        max_steps=max_steps,
        interval=interval,
    )


def _iter_terms(terms_cfg):
    if terms_cfg is None:
        return []
    items = terms_cfg.items() if isinstance(terms_cfg, dict) else _iter_cfg_items(terms_cfg)
    terms = []
    for name, value in items:
        term_cfg = _coerce_term_cfg(value)
        if term_cfg is None or not term_cfg.enabled:
            continue
        terms.append((name, term_cfg))
    return terms


def _iter_cfg_items(cfg):
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
    return items.items()


def _coerce_term_cfg(value):
    if isinstance(value, PlotTermCfg):
        return value
    if isinstance(value, type):
        return None
    if callable(value) or isinstance(value, str):
        return PlotTermCfg(func=value)
    if isinstance(value, dict):
        if value.get("enabled", True) is False:
            return None
        if "func" in value:
            return PlotTermCfg(**value)
    return None


def _flatten_term_values(values, prefix=None):
    flat = {}
    for key, value in values.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_term_values(value, full_key))
            continue
        scalar_values = _to_scalar_values(value)
        if len(scalar_values) == 1:
            flat[full_key] = scalar_values[0]
        else:
            for index, scalar in enumerate(scalar_values):
                flat[f"{full_key}_{index}"] = scalar
    return flat


def _to_scalar_values(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().numpy()
    elif isinstance(value, np.ndarray):
        value = value.flatten()
    elif isinstance(value, (list, tuple)):
        value = np.asarray(value).flatten()
    else:
        return [float(value)]
    return [float(item) for item in value]


def _get_cfg_value(cfg, name, override, default=None):
    if override is not None:
        return override
    return getattr(cfg, name, default)
