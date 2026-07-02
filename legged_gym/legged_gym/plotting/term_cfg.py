from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class PlotTermCfg:
    """Configuration for one plot data term."""

    func: Any
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
