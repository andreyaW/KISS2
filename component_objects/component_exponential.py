from dataclasses import dataclass
from component_objects.base_component import BaseComponent

import numpy as np
import sympy as sp

@dataclass
class ExponentialComponent(BaseComponent):
    """Component with exponentially distributed failure times."""
    
    def __post_init__(self):
        super().__post_init__() 
        self._lambda = 1/self.MTTF  # constant failure rate
        self.time_to_failure = self.sample_failure_time()

    def R_t(self, t):
        """Survivor function: system reliability over time."""
        if isinstance(t, sp.Basic):
            return sp.exp(-self._lambda * t)    # symbolic
        return np.exp(-self._lambda * t) # numeric
    
    def z_t(self, t):
        """Hazard function: failure rate."""
        if isinstance(t, sp.Basic):
            return self._lambda # symbolic
        return self._lambda * np.ones_like(t, dtype=float) # numeric

    def f_t(self, t):
        """Probability density function (pdf)."""
        if isinstance(t, sp.Basic):
            return self._lambda * sp.exp(-self._lambda * t) # symbolic
        return self._lambda * np.exp(-self._lambda * t) # numeric
    
    def sample_failure_time(self) -> float:
        """Sample failure time from exponential distribution."""
        return np.random.exponential(self.MTTF) 
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, distribution=Exponential)"