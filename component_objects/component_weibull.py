"""
component_weibull.py

Example of a custom component with Weibull-distributed failure time.
"""
from dataclasses import dataclass
from component_objects.base_component import BaseComponent
from math import gamma

import numpy as np

@dataclass
class WeibullComponent(BaseComponent):
    shape: float = 2.0  # shape parameter (k) (>1 implies increasing failure rate with t (aging/wear out behavior))

    def __post_init__(self):
        "set up necessary parameters and select a failure time for component instance"
        super().__post_init__() 
        self.scale = self.MTTF / gamma(1 + 1/self.shape)
        self._lambda = 1/self.scale  # failure rate (constant for exponential function)
        self.time_to_failure = self.sample_failure_time()
   
    
    def R_t(self, t) -> float:
        "Suvivor Function: returns the reliability of the component at a desired time t"
        return np.exp( -(self._lambda*t)**self.shape )
    
    def z_t(self,t):
        """Failure Rate Function: returns the failure rate of the component at a desired time t"""
        return self.shape * self._lambda * (self._lambda * t)**(self.shape-1)
    
    def f_t(self, t):
        """Probability Density Function (pdf): The distribution of failure times based on the model type"""
        return self.shape * self._lambda * (self._lambda*t)**(self.shape-1)*np.exp( -(self._lambda*t)**self.shape )

    def sample_failure_time(self) -> float:
        """Weibull distributed failure time."""
        return np.random.weibull(self.shape) * self.scale
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, distribution=Weibull(shape={self.shape})"