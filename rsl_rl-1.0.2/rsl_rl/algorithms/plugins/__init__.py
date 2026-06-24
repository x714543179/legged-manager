"""Optional PPO plugins.

Heavy plugins are imported lazily so a plain PPO run does not pull in AMP or
teacher dependencies unless their class is actually requested.
"""

from .base import PPOPlugin
from .symmetry_loss_plugin import SymmetryLossPlugin

__all__ = ["PPOPlugin", "AMPPlugin", "TeacherKLPlugin", "SymmetryLossPlugin"]


def __getattr__(name):
    if name == "AMPPlugin":
        from .amp_plugins import AMPPlugin

        return AMPPlugin
    if name == "TeacherKLPlugin":
        from .teacher_kl_plugin import TeacherKLPlugin

        return TeacherKLPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
