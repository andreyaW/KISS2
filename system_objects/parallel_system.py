# """
# parallel_system.py

# Implements a parallel system model.
# """

# from system_objects.base_system import BaseSystem
# from dataclasses import dataclass
# from utilities import is_numeric

# import numpy as np
# import sympy as sp

# @dataclass
# class ParallelSystem(BaseSystem):
#     """A system/subsystem where all components are arranged in parallel."""

#     def structure_function(self) -> int:
#         """Determine overall system state based on component states."""
#         if any(component.state == 1 for component in self.components):
#             return 1  # "working"
#         return 0  # "failed"

#     def __repr__(self):
#         return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"
    
    
#     def TTF(self):  
#         comp_TTFs = [c.time_to_failure for c in self.comps]
#         return max(comp_TTFs)
    
    
#     def R_s(self, t=None):
#         t_sym = sp.symbols("t", positive=True)

#         # --- symbolic case ---
#         if t is None or isinstance(t, sp.Symbol):
#             Rs = [c.R_t(t_sym) for c in self.components]   # SymPy expressions
#             R_sym = 1 - sp.prod([1 - Ri for Ri in Rs])
#             return sp.simplify(R_sym)

#         # --- numeric case ---
#         # First build symbolic expression
#         R_sym = self.R_s(None)
#         R_num = sp.lambdify(t_sym, R_sym, "numpy")
#         return R_num(t)
    
    
#     # -------------------------------------------------------------------------
#     # REPAIR
#     # -------------------------------------------------------------------------
#     def repair(self):
#         failed = [c for c in self.components if c.state == 0]
#         repair_time = min(c.repair() for c in failed)
#         return repair_time, failed




# ----------------------------------------------------------------------
# parallel_system.py
# ----------------------------------------------------------------------
from dataclasses import dataclass
from system_objects.base_system import BaseSystem
import sympy as sp
import numpy as np

@dataclass
class ParallelSystem(BaseSystem):
    def structure_function(self) -> int:
        return 1 if any(c.state == 1 for c in self.components) else 0

    def components_to_repair(self):
        # Parallel: repair at least one failed component to restore system
        failed = [c for c in self.components if c.state == 0]
        return failed[:1]  # repair only one

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, components={len(self.components)})"

    def R_s(self, t=None):
        t_sym = sp.symbols("t", positive=True)
        if t is None or isinstance(t, sp.Symbol):
            Rs = [c.R_t(t_sym) for c in self.components]
            return sp.simplify(1 - sp.prod([1 - r for r in Rs]))
        R_sym = self.R_s(None)
        R_num = sp.lambdify(t_sym, R_sym, "numpy")
        return R_num(t)


