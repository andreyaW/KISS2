"""
series_system.py

Implements a series system model.
"""

from system_objects.base_system import BaseSystem

from dataclasses import dataclass

@dataclass
class SeriesSystem(BaseSystem):
    """A system where components are arranged in series."""

    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        if all(component.state == 1 for component in self.components):
            return 1  # "working"
        return 0  # "failed"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"