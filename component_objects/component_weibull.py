from dataclasses import dataclass
from component_objects.base_component import BaseComponent
from math import gamma
from scipy.special import gammaincc, gamma

import numpy as np
import sympy as sp

@dataclass
class WeibullComponent(BaseComponent):
    """Component with Weibull-distributed failure times."""
    shape: float = 2.0  # k > 0

    def __post_init__(self):
        super().__post_init__()
        # Compute scale parameter from MTTF and shape
        self.scale = self.MTTF / gamma(1 + 1/self.shape)
        self._lambda = 1 / self.scale
        self.time_to_failure =  np.ceil(self.sample_failure_time() / self.dt) * self.dt # round up to nearest dt
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, distribution=Weibull(scale={self.scale}, shape={self.shape}, MTTF={self.MTTF}))"
    
    def sample_failure_time(self) -> float:
        """Sample failure time from Weibull distribution."""
        return np.random.weibull(self.shape) * self.scale

    def R_t(self, t):
        """Survivor function: system reliability over time."""
        # Symbolic
        if isinstance(t, sp.Basic):
            return sp.exp(-(self._lambda * t) ** self.shape)
        # Numeric
        return np.exp(-(self._lambda * t) ** self.shape)

    def f_t(self, t):
        """Probability density function (pdf)."""
        if isinstance(t, sp.Basic):
            return self.shape * self._lambda * (self._lambda * t) ** (self.shape - 1) * sp.exp(-(self._lambda * t) ** self.shape)
        return self.shape * self._lambda * (self._lambda * t) ** (self.shape - 1) * np.exp(-(self._lambda * t) ** self.shape)

    def z_t(self, t):
        """Hazard function: failure rate."""
        if isinstance(t, sp.Basic):
            return self.shape * self._lambda * (self._lambda * t) ** (self.shape - 1)
        return self.shape * self._lambda * (self._lambda * t) ** (self.shape - 1)        
    
    
    def medianTTF(self):
        return self.scale*(np.log(2))**(1/self.shape)
    
    # def RUL(self, t):
    #     """
    #     Remaining Useful Life (RUL) distribution given survival to time t.
    #     Returns a dict with survival, pdf, and sampler.
    #     """

    #     lam = self._lambda
    #     k = self.shape

    #     def survival(x):
    #         return np.exp(-((lam * (t + x)) ** k - (lam * t) ** k))

    #     def pdf(x):
    #         return (
    #             k * lam * (lam * (t + x)) ** (k - 1)
    #             * np.exp(-((lam * (t + x)) ** k - (lam * t) ** k))
    #         )

    #     def sample(size=1):
    #         u = np.random.rand(size)
    #         return ((-(np.log(u) - (lam * t) ** k)) ** (1 / k) - t) / lam

    #     return {
    #         "survival": survival,
    #         "pdf": pdf,
    #         "sample": sample,
    #     }        