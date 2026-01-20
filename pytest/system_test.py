import pytest
import numpy as np
import scipy

from scipy.integrate import quad
from component_objects.component_exponential import ExponentialComponent
from system_objects.series_system import SeriesSystem
from system_objects.parallel_system import ParallelSystem
from system_objects.kooN_system import KOutOfNSystem

# =============================================================================
# FIXTURES : variables/functions available to each test
# =============================================================================

@pytest.fixture
def num_comps():
    """Number of components in each system."""
    return 3

@pytest.fixture
def k():
    return 2

@pytest.fixture
def comps(num_comps):
    """
    Initialize a list of ExponentialComponent objects.
    Each component must have a unique name and its own failure_rate.
    """
    return [ExponentialComponent(f"Comp{i+1}", MTTF = 50*(i+1), MTTR = 1) for i in range(num_comps)]

@pytest.fixture
def series_system(comps):
    """
    A single SeriesSystem instance built from the component list.
    """
    return SeriesSystem("SeriesSystem", comps)

@pytest.fixture
def systems():
    """
    Factory: produce N independent systems of a given type.

        systems("series", comps, n=10)
        systems("parallel", comps, n=5)
        systems("kofn", comps, n=5, k=2)
    """
    def __factory__(system_type: str, comps, n: int = 5):
        sys_list = []

        for idx in range(n):
            # fresh component objects
            new_comps = [
                ExponentialComponent(c.name + f"_{idx}", c.MTTF, c.MTTR)
                for c in comps
            ]

            if system_type == "series":
                sys_list.append(SeriesSystem(f"Series_{idx}", new_comps))

            elif system_type == "parallel":
                sys_list.append(ParallelSystem(f"Parallel_{idx}", new_comps))

            elif system_type == "kooN":
                if k is None:
                    raise ValueError("k must be provided for k-of-n")
                sys_list.append(KOutOfNSystem(f"kooN_{idx}", new_comps, k))

            else:
                raise ValueError(f"Invalid system type: {system_type}")

        return sys_list

    return __factory__

@pytest.fixture
def expected():
    """
    Factory: compute the analytic expected MTTF for a system type.

        expected("series", comps)
        expected("parallel", comps)
        expected("kofn", comps, k=2)
    """
    def __factory__(system_type: str, comps, k: int = None):

        lambdas = np.array([c.failure_rate for c in comps])

        # ---- SERIES ----
        if system_type == "series":
            # Relaibility(t) = Π_i (e^{-λ_i t})
            return 1.0 / lambdas.sum()

        # ---- PARALLEL ----
        elif system_type == "parallel":
            # Reliability(t) = 1 - Π_i (1 - e^{-λ_i t})
            # MTTF = ∫ R(t) dt from 0..∞ (numerical)
            def R(t):
                return 1 - np.prod(1 - np.exp(-lambdas * t))

            mttf, _ = quad(R, 0, np.inf, limit=200)
            return mttf

        # ---- K OF N ----
        elif system_type == "kooN":

            n = len(comps)

            # reliability = sum_{j=k..n} C(n,j) (e^{-λt})^j (1 - e^{-λt})^(n-j)
            # MTTF = ∫ R(t) dt numerically
            def R(t):
                p = np.exp(-lambdas * t)
                q = 1 - p
                R_t = 0.0
                for j in range(k, n+1):
                    R_t += scipy.special.comb(n, j) * (p.mean()**j) * (q.mean()**(n-j))
                return R_t

            mttf, _ = quad(R, 0, np.inf, limit=200)
            return mttf

        else:
            raise ValueError(f"Unsupported system type: {system_type}")

    return __factory__

# =============================================================================
# TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TEST 1: Test the simulated system meets expected reliability metrics
# -----------------------------------------------------------------------------
# PARAMETERS
@pytest.mark.parametrize("system_type", ["series", "parallel", "kooN"])

# TEST FUNCTION
def test_series_system(system_type, comps, systems):
    
    """ 
    Monte-Carlo test:
    Simulate a Series System of exponential components and chech that it meets the expected:
        - MTTF
        - 

    Monte-Carlo test:
    Simulate a Series System object composed of exponential components,
    check that the sample MTTF  agrees with the analytic expression:

        MTTF = 1 / sum_i(lambda_i)  for i = 0...Num_Comps

    where each component i has failure rate lambda_i.
    """

    # ----- analytic MTTF -----
    lambdas = [c.failure_rate for c in comps]
    analytic_mttf = 1.0 / sum(lambdas)

    # ----- build several systems -----
    N_systems = 1000
    sys_list = systems(system_type, comps=comps, n=N_systems)

    # ----- Monte Carlo parameters -----
    MC_SAMPLES = 2000
    lifetimes = []

    for sys in sys_list:
        for _ in range(MC_SAMPLES):
            # system lifetime = min(component lifetimes) for series system
            comp_times = np.random.exponential(scale=1.0/np.array(lambdas))
            lifetimes.append(comp_times.min())

    simulated_mttf = np.mean(lifetimes)

    print(f"Expected MTTF: {analytic_mttf:6f} ; Simulated MTTF:{simulated_mttf:6f}" )

    # ----- Assertion -----
    assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
        f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"