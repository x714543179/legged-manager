from __future__ import annotations

from dataclasses import dataclass, field

from legged_gym.terrains.registry import resolve_callable
from legged_gym.terrains.terrain_data import SubTerrainResult


@dataclass
class ExternalTerrainFunction:
    """Adapter for legacy terrain functions.

    The callable receives the IsaacGym SubTerrain as its first argument. Extra
    parameters can be provided through ``params`` and dynamic generation values
    through ``pass_*`` flags.
    """

    function: str
    params: dict = field(default_factory=dict)
    terrain_type: int = -1
    pass_difficulty: bool = False
    pass_row: bool = False
    pass_col: bool = False

    def generate(self, terrain, difficulty: float, row: int, col: int, rng) -> SubTerrainResult:
        func = resolve_callable(self.function)
        kwargs = dict(self.params)
        if self.pass_difficulty:
            kwargs["difficulty"] = difficulty
        if self.pass_row:
            kwargs["row"] = row
        if self.pass_col:
            kwargs["col"] = col
        func(terrain, **kwargs)
        return SubTerrainResult(terrain, getattr(terrain, "idx", self.terrain_type), self._collect_extras(terrain))

    @staticmethod
    def _collect_extras(terrain):
        extras = {}
        for key in ("center_position", "center_position_stone", "center_position_narrow"):
            if hasattr(terrain, key):
                extras[key] = getattr(terrain, key)
        return extras

