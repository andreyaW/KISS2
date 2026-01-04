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

    def R_s(self, t=None):
        """
        Reliability of a SERIES system.
        Returns symbolic expression if t=None, else evaluated numeric values.
        """
        t_sym = sp.symbols("t", positive=True)

        # --- Symbolic expression ---
        if t is None or isinstance(t, sp.Symbol):
            R_sym = sp.Integer(1)
            for c in self.components:
                R_sym *= c.R_t(t_sym)             # c.R returns SymPy
            return R_sym

        # --- Numeric case ---
        # Create symbolic expression once
        R_sym = sp.Integer(1)
        for c in self.components:
            R_sym *= c.R_t(t_sym)

        R_num = sp.lambdify(t_sym, R_sym, "numpy")
        return R_num(t)
    
    def z_s(self, t=None):
        """
        System hazard function z_s(t).
        Returns symbolic expression if t=None, else numeric values.
        """
        t_sym = sp.symbols("t", positive=True)

        # symbolic
        if t is None or isinstance(t, sp.Symbol):
            z_sym = sp.Integer(0)
            for c in self.components:
                z_sym += c.z_t(t_sym)
            return sp.simplify(z_sym)

        # numeric
        z_sym = sp.Integer(0)
        for c in self.components:
            z_sym += c.z_t(t_sym)

        z_num = sp.lambdify(t_sym, z_sym, "numpy")
        return z_num(t)

    def f_s(self, t=None):
        """
        System PDF f_s(t).
        """
        t_sym = sp.symbols("t", positive=True)

        R_sym = self.R_s(t_sym)
        z_sym = self.z_s(t_sym)

        if t is None or isinstance(t, sp.Symbol):
            return sp.simplify(z_sym * R_sym)

        f_num = sp.lambdify(t_sym, z_sym * R_sym, "numpy")
        return f_num(t)


