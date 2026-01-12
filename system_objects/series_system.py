"""
series_system.py

Implements a series system model.
"""

from system_objects.base_system import BaseSystem
from dataclasses import dataclass

import numpy as np
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

    # -------------------------------------------------------------------------
    # REPAIR
    # -------------------------------------------------------------------------
    def repair(self):
        repair_times = np.zeros(len(self.components))
        for i,comp in enumerate(self.components):
            repair_times[i]= comp.repair()         
        repair_times.sort()
        sys_repair_time = min(repair_times)
        return sys_repair_time, repair_times