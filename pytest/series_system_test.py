import pytest
import numpy as np
from sympy import Symbol, integrate


from component_objects.component_exponential import ExponentialComponent
from system_objects.series_system import SeriesSystem


# -----------------------------------------------------------------------------
# FIXTURES:
# -----------------------------------------------------------------------------
@pytest.fixture
def component_factory():
    """
    Factory that generates a fresh list of ExponentialComponent objects.

    Usage:
        component_factory(n=3)   -> returns 3 new components
        component_factory(5)     -> returns 5 new components
    """
    def __factory__(n: int = 3):
        return [
            ExponentialComponent(
                name=f"Comp{i+1}",
                MTTF=50 * (i + 1),
                MTTR=1
            )
            for i in range(n)
        ]
    return __factory__


@pytest.fixture
def series_system(component_factory):
    """
    A single SeriesSystem built using the component factory.
    Produces fresh components each time the fixture is requested.
    """
    comps = component_factory(3)
    return SeriesSystem("SeriesSystem", comps)


@pytest.fixture
def series_systems(component_factory):
    """
    Factory: produce N independent systems of a given type.
    
    Usage:
        systems("series", n=50)   -> list of 50 series systems,
                                     each with unique components.
    """
    def __factory__(system_type: str, n: int = 1000):
        sys_list = []

        for idx in range(n):
            # fresh components for each system
            comps = [
                ExponentialComponent(
                    name=f"Comp{i+1}_{idx}",
                    MTTF=50 * (i + 1),
                    MTTR=1
                )
                for i in range(3)
            ]
            
            # initialize a system using the new comps and add it to the systems list                
            system = SeriesSystem(f"Series_{idx}", comps)
            sys_list.append(system)

        return sys_list

    return __factory__


@pytest.fixture
def expected(MTTF_1, MTTF_2, MTTF_3):
    """
    compute the analytic z_s, R_s, and MTTF_s for a given set series system of parameters.
    """

    MTTF_i = np.array([MTTF_1, MTTF_2, MTTF_3])
    
    z_s = np.sum(1/MTTF_i)
    R_s = 
    MTTF_s= 1/z_s
    
    # lambdas = np.array([c.failure_rate for c in comps])

    # # ---- SERIES ----
    # if system_type == "series":
    #     return 1.0 / lambdas.sum()

    return z_s, R_s, MTTF_s

# -----------------------------------------------------------------------------
# PARAMETERS AND PARAMETER SETS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("MTTF_1, MTTF_2, MTTF_3", 
                         [10, 10, 10],
                         [20, 10, 10])
# -----------------------------------------------------------------------------
# TEST FUNCTION 1: Test the simulated system meets expected reliability metrics
# -----------------------------------------------------------------------------
def test_series_system_exponential(MTTF_1, MTTF_2, MTTF_3, expected, comps, systems):
    
    """ 
    Monte-Carlo test:
    Simulate a Series System of exponential components and chech that it meets the expected:
        - failure rate
        - MTTF
        - system reliabiity
    """

    # ----- analytic solutions -----
    # lambdas = [c.failure_rate for c in comps]
    # analytic_mttf = 1.0 / sum(lambdas)

    # ----- build several systems -----
    N_systems = 1000
    sys_list = systems(system_type, comps=comps, n=N_systems)

    # ----- Monte Carlo parameters -----
    MC_SAMPLES = 2000
    simulated_MTTFs = []
    simulated_z_s= []
    simulated_R_s= []

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
        
    assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
        f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"
        
    assert np.isclose(simulated_mttf, analytic_mttf, rtol=0.10), \
        f"Sample MTTF {simulated_mttf:.4f} != Analytic {analytic_mttf:.4f}"