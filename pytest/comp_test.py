import pytest
import numpy as np
import matplotlib.pyplot as plt

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent

# -----------------------------------------------------------------------------
# FIXTURES : variables/functions available to each test
# -----------------------------------------------------------------------------
@pytest.fixture
def tolerance():
    return 0.1


@pytest.fixture
def n():
    return 10000


@pytest.fixture
def comps(component_class, kwargs, n):
    # Build components
    return [component_class(name=f"comp_{i}", **kwargs, MTTR=2) for i in range(n)]

# -----------------------------------------------------------------------------
# PARAMETERS AND PARAMETER SETS
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("run", range(3))   # number of times to run each parameter set
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10}),
    (ExponentialComponent, {"MTTF": 50}),
    (ExponentialComponent, {"MTTF": 100}),

    # Weibull: mean ≠ scale, so we provide shape + scale or shape + desired MTTF
    (WeibullComponent, {"shape": 2, "MTTF": 100}),
    (WeibullComponent, {"shape": 2, "MTTF": 15}),
    (WeibullComponent, {"shape": 2, "MTTF": 250}),
])

# -----------------------------------------------------------------------------
# TEST FUNCTION 1: SAMPLE FAILURE TIME TEST
# -----------------------------------------------------------------------------
def test_sample_failure_time(tolerance, component_class, kwargs, run, comps):
    """
    Generic test that verifies the empirical MTTF of any component class.
    """
    # Extract the target MTTF
    target_MTTF = kwargs["MTTF"]

    # Collect failure times
    failure_times = np.array([c.time_to_failure for c in comps])
    mean_failure_time = failure_times.mean()

    # Allow ±3% tolerance
    assert abs(mean_failure_time - target_MTTF) < tolerance * target_MTTF


# -----------------------------------------------------------------------------
# TEST FUNCTION 2: VERIFY ALL COMPS UNIQUE
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10}),
    
    # Weibull: mean ≠ scale, so we provide shape + scale or shape + desired MTTF
    (WeibullComponent, {"shape": 10, "MTTF": 10}),
])
def test_comps_unique(component_class, kwargs, comps):
    
    # grab the failure times of all the initialized comps
    all_failure_times = np.array([c.time_to_failure for c in comps])

    first_ft = all_failure_times[0]

    # check they have a unique failure time (not all equal to the first)
    assert not np.allclose(all_failure_times, first_ft)



# -----------------------------------------------------------------------------
# TEST FUNCTION 2: VERIFY ALL COMPS UNIQUE
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10}),
    (WeibullComponent, {"shape": 10, "MTTF": 10 }) ])

def test_comp_R_s_t(component_class, kwargs, comps):
    
    # grab the failure times of all the initialized comps
    TTFs = np.array([c.time_to_failure for c in comps])

    # simulated MTTF
    MTTF_sim = np.mean(TTFs)
    
    # simulated R(t)
    time_points = np.arange(0,50,0.1)
    R_sim = np.array([np.mean(TTFs > t) for t in time_points])

    # R(t) analytical function 
    R_t = comps[0].R_t(time_points)

    # MTTF analytical solution
    MTTF_analytical = comps[0].MTTF
    
    plt.plot(time_points, R_sim, "o")
    plt.plot(time_points, R_t)
    plt.title(fr"$R_i(t): {comps[0]}$")
    plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=fr"$MTTF_sim ={MTTF_sim:.3f}$")
    plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=fr"$MTTF_analytical ={MTTF_sim:.3f}$")
    plt.show()
    