# import pytest
# import numpy as np
# from sympy import Symbol, integrate


# from component_objects.component_exponential import ExponentialComponent
# from system_objects.series_system import SeriesSystem


# # -----------------------------------------------------------------------------
# # FIXTURES:
# # -----------------------------------------------------------------------------
# @pytest.fixture
# def component_factory():
#     """
#     Factory that generates a fresh list of ExponentialComponent objects.

#     Usage:
#         component_factory(n=3)   -> returns 3 new components
#         component_factory(5)     -> returns 5 new components
#     """
#     def __factory__(n: int = 3):
#         return [
#             ExponentialComponent(
#                 name=f"Comp{i+1}",
#                 MTTF=50 * (i + 1),
#                 MTTR=1
#             )
#             for i in range(n)
#         ]
#     return __factory__


# @pytest.fixture
# def series_system(component_factory):
#     """
#     A single SeriesSystem built using the component factory.
#     Produces fresh components each time the fixture is requested.
#     """
#     comps = component_factory(3)
#     return SeriesSystem("SeriesSystem", comps)


# @pytest.fixture
# def series_systems(component_factory):
#     """
#     Factory: produce N independent systems of a given type.
    
#     Usage:
#         systems("series", n=50)   -> list of 50 series systems,
#                                      each with unique components.
#     """
#     def __factory__(system_type: str, n: int = 1000):
#         sys_list = []

#         for idx in range(n):
#             # fresh components for each system
#             comps = [
#                 ExponentialComponent(
#                     name=f"Comp{i+1}_{idx}",
#                     MTTF=50 * (i + 1),
#                     MTTR=1
#                 )
#                 for i in range(3)
#             ]
            
#             # initialize a system using the new comps and add it to the systems list                
#             system = SeriesSystem(f"Series_{idx}", comps)
#             sys_list.append(system)

#         return sys_list

#     return __factory__


# @pytest.fixture
# def expected(MTTF_1, MTTF_2, MTTF_3):
#     """
#     compute the analytic z_s, R_s, and MTTF_s for a given set series system of parameters.
#     """

#     MTTF_i = np.array([MTTF_1, MTTF_2, MTTF_3])
    
#     z_s = np.sum(1/MTTF_i)
#     R_s = 
#     MTTF_s= 1/z_s
    
#     # lambdas = np.array([c.failure_rate for c in comps])

#     # # ---- SERIES ----
#     # if system_type == "series":
#     #     return 1.0 / lambdas.sum()

#     return z_s, R_s, MTTF_s

# # -----------------------------------------------------------------------------
# # PARAMETERS AND PARAMETER SETS
# # -----------------------------------------------------------------------------
# @pytest.mark.parametrize("MTTF_1, MTTF_2, MTTF_3", 
#                          [10, 10, 10],
#                          [20, 10, 10])
# # -----------------------------------------------------------------------------
# # TEST FUNCTION 1: Test the simulated system meets expected reliability metrics
# # -----------------------------------------------------------------------------
# def test_series_system_exponential(MTTF_1, MTTF_2, MTTF_3, expected, comps, systems):
    
#     """ 
#     Monte-Carlo test:
#     Simulate a Series System of exponential components and chech that it meets the expected:
#         - failure rate
#         - MTTF
#         - system reliabiity
#     """

#     # ----- analytic solutions -----
#     # lambdas = [c.failure_rate for c in comps]
#     # analytic_mttf = 1.0 / sum(lambdas)

#     # ----- build several systems -----
#     N_systems = 1000
#     sys_list = systems(system_type, comps=comps, n=N_systems)

#     # ----- Monte Carlo parameters -----
#     MC_SAMPLES = 2000
#     simulated_MTTFs = []
#     simulated_z_s= []
#     simulated_R_s= []

#     for sys in sys_list:
#         for _ in range(MC_SAMPLES):
#             # system lifetime = min(component lifetimes) for series system
#             comp_times = np.random.exponential(scale=1.0/np.array(lambdas))
#             lifetimes.append(comp_times.min())

#     simulated_mttf = np.mean(lifetimes)

#     print(f"Expected MTTF: {analytic_mttf:6f} ; Simulated MTTF:{simulated_mttf:6f}" )

#     # ----- Assertion -----
#     assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
#         f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"
        
#     assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
#         f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"
        
#     assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
#         f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"



import pytest
import numpy as np
from sympy import Symbol, exp

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent
from system_objects.series_system import SeriesSystem


# -----------------------------------------------------------------------------
# FIXTURES:
# -----------------------------------------------------------------------------
@pytest.fixture
def component_factory():
    """
    Factory that creates a list of components (Exponential or Weibull).

    Usage:
        component_factory("exp", [10, 10, 20])
        component_factory("weib", [(MTTF,k), ...])
    """
    def __factory__(dist_type="exp", params=None):
        comps = []

        if dist_type == "exp":
            # params = list of MTTFs
            for i, mttf in enumerate(params):
                comps.append(
                    ExponentialComponent(
                        name=f"Comp{i+1}",
                        MTTF=mttf,
                        MTTR=1
                    )
                )

        elif dist_type == "weib":
            # params = list of (MTTF,k) pairs
            for i, (mttf, k) in enumerate(params):
                comps.append(
                    WeibullComponent(
                        name=f"Comp{i+1}",
                        MTTF=mttf,
                        shape=k,
                        MTTR=1
                    )
                )

        return comps

    return __factory__


@pytest.fixture
def system_factory(component_factory):
    """
    Factory that produces N fresh SeriesSystem objects.

    Example:
        system_factory("exp", [10,10,10], n=1000)
        system_factory("weib", [(20,1.5),(20,1.5),(20,1.5)], n=500)
    """
    def __factory__(dist_type, params, n=1000):
        systems = []
        for i in range(n):
            comps = component_factory(dist_type, params)
            systems.append(SeriesSystem(f"Sys_{i}", comps))
        return systems

    return __factory__


@pytest.fixture
def expected_series():
    """
    Compute analytic R_s(t) and MTTF_s for a series system.

    Handles:
      - exponential components
      - weibull components (via symbolic R_i(t) product)

    Returns:
        analytic_Rs(t)   (callable)
        analytic_MTTF_s
    """
    def __factory__(dist_type, params):

        t = Symbol("t", real=True, positive=True)

        # -------- EXPONENTIAL --------
        if dist_type == "exp":
            lambdas = np.array([1/p for p in params])
            z_s = lambdas.sum()
            R_s_t = exp(-z_s * t)
            MTTF_s = 1 / z_s
            return (R_s_t, float(MTTF_s))

        # -------- WEIBULL --------
        elif dist_type == "weib":
            # params = [(MTTF, k), ...]
            R_terms = []
            for (mttf, k) in params:
                # scale parameter θ from MTTF:
                # MTTF = θ Γ(1 + 1/k)
                from scipy.special import gamma
                theta = mttf / gamma(1 + 1/k)
                R_terms.append(exp(-(t/theta)**k))

            # Series = product of R_i(t)
            R_s_t = 1
            for Ri in R_terms:
                R_s_t *= Ri

            # MTTF_s = ∫ R_s(t) dt numerically
            import mpmath as mp
            R_callable = lambda x: float(R_s_t.subs(t, x))
            MTTF_s = mp.quad(R_callable, [0, mp.inf])

            return (R_s_t, float(MTTF_s))

    return __factory__
    

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist_type, params",
    [
        ("exp", [10, 10, 10]),
        ("exp", [20, 10, 5]),
        ("weib", [(10, 1.5), (10, 1.5), (10, 1.5)]),
    ]
)
# -----------------------------------------------------------------------------
# TEST: Monte-Carlo validation of Series System reliability & MTTF
# -----------------------------------------------------------------------------
def test_series_system_mc(dist_type, params, system_factory, expected_series):
    """
    Monte-Carlo test:
    Builds 1000 series systems, simulates lifetimes, and compares:

        - system MTTF
        - system reliability R_s(t) at selected times

    against analytical expectations.
    """

    # ----- analytical target -----
    R_s_sym, MTTF_expected = expected_series(dist_type, params)

    # pick time points for R(t) comparison
    time_points = np.array([1, 5, 10])
    R_s_expected = np.array([float(R_s_sym.subs(Symbol("t"), τ)) for τ in time_points])

    # ----- build systems -----
    N_systems = 1000
    systems = system_factory(dist_type, params, n=N_systems)

    # ----- Monte-Carlo simulation -----
    N_SAMPLES = 2000
    lifetimes = []
    R_s_estimates = {τ: [] for τ in time_points}

    for sys in systems:
        for _ in range(N_SAMPLES):

            # ---- sample component lifetimes ----
            comp_times = []
            for c in sys.components:

                if dist_type == "exp":
                    comp_times.append(
                        np.random.exponential(scale=c.MTTF)
                    )
                else:  # weibull
                    k = c.shape
                    theta = c.scale
                    comp_times.append(
                        np.random.weibull(k) * theta
                    )

            # system life = min component life
            L = min(comp_times)
            lifetimes.append(L)

            # indicator R(t) ≈ P(L > t)
            for τ in time_points:
                R_s_estimates[τ].append(L > τ)

    # Monte-Carlo MTTF
    MTTF_sim = np.mean(lifetimes)

    # Monte-Carlo R(t)
    R_s_sim = np.array([np.mean(R_s_estimates[τ]) for τ in time_points])

    # ----- Assertions -----
    assert np.isclose(MTTF_sim, MTTF_expected, rtol=0.10), \
        f"Simulated MTTF={MTTF_sim:.3f} vs Analytic={MTTF_expected:.3f}"

    assert np.allclose(R_s_sim, R_s_expected, rtol=0.10), \
        f"Simulated R_s={R_s_sim} vs Analytic={R_s_expected}"
