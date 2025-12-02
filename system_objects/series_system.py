"""
series_system.py

Implements a series system model.
"""

from system_objects.base_system import BaseSystem
from dataclasses import dataclass

import sympy as sp

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
    
    
    def TTF(self):
        comp_TTFs = [c.time_to_failure for c in self.components]
        return min(comp_TTFs)

    def R_s(self, t):
        """Series reliability: product of component reliabilities evaluated at t_val"""
        R_s = 1
        for c in self.components:
            R_s *= c.R_t(t)  # assume component R(t) accepts t as input
        return R_s

    def z_s(self, t):
        """Series hazard: sum of component hazards evaluated at t_val"""
        z_s = 0
        for c in self.components:
            z_s += c.z_t(t)  # component hazard takes t as input
        return z_s

    def f_s(self, t):
        """System PDF: f(t) = z_s(t) * R_s(t) evaluated at t_val"""
        return self.z_s(t) * self.R_s(t)

    def MTTF_s(self):
        """System MTTF: ∫₀∞ R_s(t) dt"""
        t = sp.symbols("t", positive=True)
        # build symbolic R_s(t) from components then integrate it
        R_s = 1
        for c in self.components:
            R_s *= c.R_t(t)
        return sp.simplify(sp.integrate(R_s, (t, 0, sp.oo)))