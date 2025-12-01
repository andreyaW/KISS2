"""
k-out-of-n_system.py

Implements a k-out-of-n system model.
"""

from system_objects.base_system import BaseSystem

from dataclasses import dataclass

@dataclass
class KOutOfNSystem(BaseSystem):
    """A system where k out of n components must be working."""

    k: int = 1# Minimum number of working components required for system to be "working"

    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        if sum(component.state == 1 for component in self.components) >= self.k:
            return 1  # "working"
        return 0  # "failed"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"
    
    def TTF(self):
        comp_TTFs = [c.time_to_failure for c in self.comps]
        comp_TTFs = sorted(comp_TTFs, reverse=True)
        return comp_TTFs[self.k-1]