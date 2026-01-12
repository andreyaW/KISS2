"""
k_out_of_n_system.py

Implements a general k-out-of-n system model.
"""

from system_objects.base_system import BaseSystem
from dataclasses import dataclass

import numpy as np
import sympy as sp
import itertools

@dataclass
class KOutOfNSystem(BaseSystem):
    """System where at least k of n components must be working."""

    k: int = 1  # Minimum number of working components required

    # ------------------------------------------------------------
    # Structural Logic (state-based)
    # ------------------------------------------------------------
    def structure_function(self) -> int:
        """Returns 1 if at least k components are working."""
        working = sum(c.state == 1 for c in self.components)
        return 1 if working >= self.k else 0

    def __repr__(self):
        return (f"{self.__class__.__name__}(name={self.name}, "
                f"state={self.state}, num_components={len(self.components)}, k={self.k})")

    def TTF(self):
        """k-out-of-n failure time = (n-k+1)-th order statistic of component TTFs."""
        comp_TTFs = sorted([c.time_to_failure for c in self.components])
        # system fails when the (n-k+1)-th component fails
        index = len(comp_TTFs) - self.k
        return comp_TTFs[index]

    def R_s(self, t):
        t_sym = sp.symbols("t", positive=True)

        # --- symbolic case ---
        if t is None or isinstance(t, sp.Symbol):
            Rs = [c.R_t(t_sym) for c in self.components]
            R_sym = sp.Integer(0)

            N = len(Rs)
            k = self.k

            for i in range(k, N+1):
                # sum over exactly i survivors
                for comb_idx in itertools.combinations(range(N), i):
                    prod = sp.Integer(1)
                    for j in range(N):
                        if j in comb_idx:
                            prod *= Rs[j]
                        else:
                            prod *= (1 - Rs[j])
                    R_sym += prod
            return sp.simplify(R_sym)

        # --- numeric case ---
        R_sym = self.R_s(None)
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
        sys_repair_time = repair_times[self.k-1]
        return sys_repair_time, repair_times