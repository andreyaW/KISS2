from system_objects.base_system import BaseSystem
from dataclasses import dataclass

import numpy as np
import sympy as sp

# GLOBAL CONSTANTS FOR SYSTEM STATES
WORKING_STATE = 1
FAILED_STATE = 0
REPAIR_STATE = -1

# =============================================================================
# SERIES SYSTEM CLASS
# =============================================================================
@dataclass
class SeriesSystem(BaseSystem):
    """A system where components are arranged in series."""

    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        return min(c.state for c in self.components)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"        
    
    def TTF(self):
        comp_TTFs = [c.time_to_failure for c in self.components]
        return min(comp_TTFs)

    def R_s(self, t=None):
        # --- Symbolic case ---
        t_sym = sp.symbols("t", positive=True)
        if t is None or isinstance(t, sp.Symbol):
            R_sym = 1
            for c in self.components:
                R_sym *= c.R_t(t_sym)
            return R_sym
        R_sym = 1
        for c in self.components:
            R_sym *= c.R_t(t_sym)

        # --- Numeric case ---
        R_num = sp.lambdify(t_sym, R_sym, "numpy")
        return R_num(t)

    def min_components_required(self) -> int:
        return len(self.components)