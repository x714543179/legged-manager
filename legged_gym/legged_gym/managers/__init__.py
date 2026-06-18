"""Manager-style environment components inspired by IsaacLab.

The managers in this package keep the existing Isaac Gym/rsl_rl API intact while
moving task behavior out of the monolithic environment class.
"""

from .manager_base import ManagerBase, ManagerTermCfg
from .action_manager import ActionManager
from .command_manager import CommandManager
from .event_manager import EventManager
from .observation_manager import ObservationManager
from .reward_manager import RewardManager
from .termination_manager import TerminationManager

__all__ = [
    "ActionManager",
    "CommandManager",
    "EventManager",
    "ManagerBase",
    "ManagerTermCfg",
    "ObservationManager",
    "RewardManager",
    "TerminationManager",
]
