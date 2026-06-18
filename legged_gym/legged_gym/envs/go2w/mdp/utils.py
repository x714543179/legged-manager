"""Small helpers shared by go2w MDP terms."""

from __future__ import annotations


def cfg_term(env, cfg_name, term_name):
    manager_cfg = getattr(env.cfg, cfg_name, None)
    if manager_cfg is None:
        return None
    term = getattr(manager_cfg, term_name, None)
    if isinstance(term, dict) and term.get("enabled", True) is False:
        return None
    if getattr(term, "enabled", True) is False:
        return None
    return term


def cfg_term_params(env, cfg_name, term_name, default=None):
    term = cfg_term(env, cfg_name, term_name)
    if term is None:
        return {} if default is None else default
    if isinstance(term, dict):
        return dict(term.get("params", {}))
    return dict(getattr(term, "params", {}))


def call_cfg_term(env, cfg_name, term_name, *args, default=None, **kwargs):
    term = cfg_term(env, cfg_name, term_name)
    if term is None:
        return default
    func = term.get("func") if isinstance(term, dict) else getattr(term, "func", None)
    if func is None:
        return default
    if isinstance(func, str):
        func = getattr(env, func)
    params = cfg_term_params(env, cfg_name, term_name)
    params.update(kwargs)
    if isinstance(term, dict) and term.get("env_arg", False):
        return func(env, *args, **params)
    if not isinstance(term, dict) and getattr(term, "env_arg", False):
        return func(env, *args, **params)
    return func(*args, **params)


def manager_term_params(env, manager_name, term_name, default=None):
    manager = getattr(env, f"{manager_name}_manager", None)
    if manager is not None:
        for name, term in zip(manager.active_terms, manager._terms):
            if name == term_name:
                return dict(term.params)
    cfg_names = {
        "action": "actions",
        "observation": "observations",
        "event": "events",
        "reward": "rewards_manager",
        "termination": "terminations",
    }
    return cfg_term_params(env, cfg_names.get(manager_name, manager_name), term_name, default)


def latency_buffer_length(env, cfg_name, term_name, default=(0, 0)):
    latency_range = cfg_term_params(env, cfg_name, term_name).get("latency_range", default)
    return int(latency_range[1]) + 1


def sequence_value(values, index, default=0.0):
    if values is None or index >= len(values):
        return default
    return values[index]
