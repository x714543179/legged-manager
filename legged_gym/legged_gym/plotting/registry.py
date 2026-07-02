import importlib
from typing import Any, Callable


def resolve_callable(func):
    """Resolve a callable from a callable object or module path string."""
    if callable(func):
        return func
    if not isinstance(func, str):
        raise TypeError("Expected a callable or import path string.")

    if ":" in func:
        module_name, attr_name = func.split(":", 1)
    else:
        module_name, attr_name = func.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_backend(name_or_path):
    """Resolve a plotting backend class."""
    if callable(name_or_path):
        return name_or_path
    if name_or_path in (None, "", "matplotlib"):
        from .backends.matplotlib_backend import MatplotlibBackend

        return MatplotlibBackend
    return resolve_callable(name_or_path)
