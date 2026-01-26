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
        return max(c.state for c in self.components)

    def components_to_repair(self):
        # Parallel: repair at least one failed component to restore system (the one with shortest MTTR)
        MTTRs = {c.name: c.MTTR for c in self.components if c.state == 0}
        min_mttr = np.min(list(MTTRs.values()))
        return [c for c in self.components if c.state == 0 and c.MTTR == min_mttr][0]
    
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