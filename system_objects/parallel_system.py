from dataclasses import dataclass
from system_objects.base_system import BaseSystem
import sympy as sp
import numpy as np

# GLOBAL CONSTANTS FOR SYSTEM STATES
WORKING_STATE = 1
FAILED_STATE = 0
REPAIR_STATE = -1

# =============================================================================
# PARALLEL SYSTEM CLASS
# =============================================================================
@dataclass
class ParallelSystem(BaseSystem):
    """A system where components are arranged in parallel. At least one component must be working for the system to work."""
    
    def structure_function(self) -> int:
        """ Returns 1 if at least one component is working. """
        if any(c.state == WORKING_STATE for c in self.components):
            return WORKING_STATE
        elif any(c.state == REPAIR_STATE for c in self.components):
            return REPAIR_STATE
        else:
            return FAILED_STATE

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, components={len(self.components)})"

    def TTF(self):
        """Parallel system failure time = max of component TTFs."""
        comp_TTFs = [c.time_to_failure for c in self.components]
        return max(comp_TTFs)

    def R_s(self, t=None):
        # --- symbolic case ---
        t_sym = sp.symbols("t", positive=True)
        if t is None or isinstance(t, sp.Symbol):
            Rs = [c.R_t(t_sym) for c in self.components]
            return sp.simplify(1 - sp.prod([1 - r for r in Rs]))
        
        # --- numeric case ---
        R_sym = self.R_s(None)
        R_num = sp.lambdify(t_sym, R_sym, "numpy")
        return R_num(t)
    
    def min_components_required(self) -> int:
        return 1