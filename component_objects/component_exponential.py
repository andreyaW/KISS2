"""
component_exponential.py

Implements an exponential failure model component.
"""

from dataclasses import dataclass
from component_objects.base_component import BaseComponent

import numpy as np


@dataclass
class ExponentialComponent(BaseComponent):
    """Component with exponentially distributed failure times."""
    
    def __post_init__(self):
        "set up necessary parameters and select a failure time for component instance"
        super().__post_init__() 
        self._lambda = 1/self.MTTF  # failure rate (constant for exponential function)   
        self.time_to_failure = self.sample_failure_time()

    def R_t(self, t):
        "Suvivor Function: returns the reliability of the component at a desired time t"
        R_t = np.exp(-self._lambda*t)    
        return R_t
    
    def z_t(self,t):
        """Failure Rate Function: returns the failure rate of the component at a desired time t"""
        return np.array([self._lambda for _ in t])

    def f_t(self, t):
        """Probability Density Function (pdf): The distribution of failure times based on the model type"""
        return self._lambda*np.exp(-self._lambda*t)
    
    def sample_failure_time(self) -> float:
        """Draw from exponential distribution with mean = MTTF."""
        self.failure_rate = 1/self.MTTF # set the failure rate of the component
        return np.random.exponential(self.MTTF)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, distribution=Exponential)"