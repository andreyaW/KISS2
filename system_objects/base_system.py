"""
base_system.py

Defines an abstract base class for ship systems.
Provides a clear API for different types of ship systems.
"""
from component_objects.base_component import BaseComponent
from objects import BasicObject
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

@dataclass
class BaseSystem(BasicObject, ABC):
    """
    Abstract base class for systems in long-term reliability simulation.
    """
    name: str
    components : list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)       # overall system state: 1 = working, 0 = failed

    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        pass        
    
    # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    def step(self, dt: float = 1.0) -> int:
        """Advance the simulation by dt time units and return system state."""
        for component in self.components:
            component.step(dt)
        if self.state == 0:      # failed
            self.logger.info(f"{self.name}: System has failed.")
            return
        self.state = self.structure_function()
