"""
parallel_system.py

Implements a parallel system model.
"""

from system_objects.base_system import BaseSystem

from dataclasses import dataclass

@dataclass
class ParallelSystem(BaseSystem):
    """A system where components are arranged in parallel."""
    # self.parallels : List[tuple] = []

    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        if any(component.state == 1 for component in self.components):
            return 1  # "working"
        return 0  # "failed"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"
    
    
    def TTF(self):  
        comp_TTFs = [c.time_to_failure for c in self.comps]
        return max(comp_TTFs)
    
# import sympy as sp

# def parallel_R(f_list, R_list):
#     t = sp.Symbol("t", positive=True)
#     return 1 - sp.prod([1 - Ri for Ri in R_list])

# def parallel_f(f_list, R_list):
#     t = sp.Symbol("t", positive=True)
#     n = len(f_list)

#     return sum(
#         f_list[i] * sp.prod([1 - R_list[j] for j in range(n) if j != i])
#         for i in range(n)
#     )

# def parallel_z(f_list, R_list):
#     return sp.simplify(parallel_f(f_list, R_list) / parallel_R(f_list, R_list))
