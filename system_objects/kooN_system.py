# """
# k-out-of-n_system.py

# Implements a k-out-of-n system model.
# """

# from system_objects.base_system import BaseSystem

# from dataclasses import dataclass

# @dataclass
# class KOutOfNSystem(BaseSystem):
#     """A system where k out of n components must be working."""

#     k: int = 1# Minimum number of working components required for system to be "working"

#     def structure_function(self) -> int:
#         """Determine overall system state based on component states."""
#         if sum(component.state == 1 for component in self.components) >= self.k:
#             return 1  # "working"
#         return 0  # "failed"

#     def __repr__(self):
#         return f"{self.__class__.__name__}(name={self.name}, state={self.state}, num_components={len(self.components)})"
    
#     def TTF(self):
#         comp_TTFs = [c.time_to_failure for c in self.comps]
#         comp_TTFs = sorted(comp_TTFs, reverse=True)
#         return comp_TTFs[self.k-1]

"""
k_out_of_n_system.py

Implements a general k-out-of-n system model.
"""

from system_objects.base_system import BaseSystem
from dataclasses import dataclass
import numpy as np
import sympy as sp


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

    # ------------------------------------------------------------
    # Time-to-Failure Simulation (component TTFs)
    # ------------------------------------------------------------
    def TTF(self):
        """k-out-of-n failure time = (n-k+1)-th order statistic of component TTFs."""
        comp_TTFs = sorted([c.time_to_failure for c in self.components])
        # system fails when the (n-k+1)-th component fails
        index = len(comp_TTFs) - self.k
        return comp_TTFs[index]

    # ------------------------------------------------------------
    # Reliability Computation: R_s(t)
    # ------------------------------------------------------------
    def R_s(self, t):
        """
        Vectorized system reliability for arbitrary components.
        Uses dynamic programming to compute:
            R_s(t) = Pr( >= k of n components survive to time t )
        Inputs:
            t : scalar or numpy array
        Returns:
            numpy array of R_s(t) values
        """

        # Step 1: Collect individual component reliabilities at time t
        R = [c.R_t(t) for c in self.components]  # each R[i] is array or scalar
        n = len(R)

        # DP[j] = probability that exactly j components survive
        # Shape: (k..n) × t-array shape
        t_shape = np.shape(R[0])
        DP = np.zeros((n + 1,) + t_shape)

        # Initially: probability 0 components working = 1
        DP[0] = np.ones(t_shape)

        # DP recursion for each component
        for Ri in R:
            new_DP = np.zeros_like(DP)
            for j in range(n):
                # component fails → contributes (1-Ri)
                new_DP[j] += DP[j] * (1 - Ri)
                # component survives → contributes Ri
                new_DP[j + 1] += DP[j] * Ri
            DP = new_DP

        # sum over j >= k
        return np.sum(DP[self.k:], axis=0)

    # ------------------------------------------------------------
    # PDF f_s(t)
    # ------------------------------------------------------------
    def f_s(self, t):
        """
        System pdf = sum over all components of:
            f_i(t) * Pr(at least k-1 of the other components survive)
        """

        n = len(self.components)
        f_vals = np.zeros_like(self.components[0].R_t(t))

        # loop components: system fails AT t when this component fails at t
        for i, ci in enumerate(self.components):
            fi = ci.f_t(t)
            other_R = [c.R_t(t) for j, c in enumerate(self.components) if j != i]
            m = len(other_R)

            # DP for other components
            t_shape = np.shape(fi)
            DP = np.zeros((m + 1,) + t_shape)
            DP[0] = np.ones(t_shape)

            for Rj in other_R:
                new_DP = np.zeros_like(DP)
                for j in range(m):
                    new_DP[j] += DP[j] * (1 - Rj)
                    new_DP[j + 1] += DP[j] * Rj
                DP = new_DP

            # at least (k-1) other components must be alive
            f_vals += fi * np.sum(DP[self.k - 1:], axis=0)

        return f_vals

    # ------------------------------------------------------------
    # Hazard Rate z_s(t) = f_s(t) / R_s(t)
    # ------------------------------------------------------------
    def z_s(self, t):
        Rs = self.R_s(t)
        fs = self.f_s(t)
        z = fs / Rs
        z[Rs == 0] = 0  # avoid division by zero
        return z

    # ------------------------------------------------------------
    # Mean Time To Failure (symbolic integration)
    # ------------------------------------------------------------
    def MTTF_s(self):
        """
        Symbolic MTTF = ∫0∞ R_s(t) dt
        Note: Only works if each c.R_t(t_symbolic) returns a SymPy expression.
        """
        t = sp.symbols("t", positive=True)
        R_s_sym = self.R_s(t)   # calls symbolic version of R_s
        return sp.integrate(R_s_sym, (t, 0, sp.oo))