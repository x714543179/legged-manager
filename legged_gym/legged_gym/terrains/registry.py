from __future__ import annotations

import importlib


def resolve_callable(path_or_obj):
    if callable(path_or_obj):
        return path_or_obj
    if not isinstance(path_or_obj, str):
        raise TypeError(f"Expected callable or import path, got {type(path_or_obj)!r}.")
    if ":" in path_or_obj:
        module_name, attr_name = path_or_obj.split(":", 1)
    else:
        module_name, attr_name = path_or_obj.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def cfg_to_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return {key: cfg_to_dict(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [cfg_to_dict(value) for value in cfg]
    if not hasattr(cfg, "__dict__") and not isinstance(cfg, type):
        return cfg
    items = {}
    for key in dir(cfg):
        if key.startswith("__"):
            continue
        value = getattr(cfg, key)
        if callable(value):
            continue
        items[key] = cfg_to_dict(value)
    return items


def build_from_cfg(cfg, *args, **kwargs):
    cfg_dict = cfg_to_dict(cfg)
    class_name = cfg_dict.pop("class_name")
    cls = resolve_callable(class_name)
    return cls(*args, **cfg_dict, **kwargs)

