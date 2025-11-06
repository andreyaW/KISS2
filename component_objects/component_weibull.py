"""
component_weibull.py

Example of a custom component with Weibull-distributed failure time.
"""
from dataclasses import dataclass
from .base_component import BaseComponent
from shipClass_continous.objects import BasicObject

import numpy as np

@dataclass
class WeibullComponent(BaseComponent, BasicObject):
    shape: float = 2.0  # shape parameter (k) (K>1 implies increasing failure rate (aging))

    def sample_failure_time(self) -> float:
        """Weibull distributed failure time."""
        return np.random.weibull(self.shape) * self.MTTF
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}), distribution=Weibull(shape={self.shape})"