"""
component_exponential.py

Implements an exponential failure model component.
"""

from dataclasses import dataclass
from shipClass_continous.objects import BasicObject
from .base_component import BaseComponent


import numpy as np

@dataclass
class ExponentialComponent(BaseComponent, BasicObject):
    """Component with exponentially distributed failure times."""

    def sample_failure_time(self) -> float:
        """Draw from exponential distribution with mean = MTTF."""
        return np.random.exponential(self.MTTF)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}), distribution=Exponential"