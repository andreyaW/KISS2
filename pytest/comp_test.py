import pytest
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent

# =============================================================================
# FIXTURES : variables/functions available to each test
# =============================================================================

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

# =============================================================================
# TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TEST FUNCTION 1: SAMPLE FAILURE TIME TEST
# -----------------------------------------------------------------------------
# PARAMETERS:
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

# TEST FUNCTION
def test_sample_failure_time(run, tolerance, kwargs, comps):
    """
    Generic test that verifies the empirical MTTF of any component class.
    """
    # Extract the target MTTF
    target_MTTF = kwargs["MTTF"]

    # Collect failure times
    failure_times = np.array([c.time_to_failure for c in comps])
    mean_failure_time = failure_times.mean()
    # print(comps[0].__class__.__name__) # Uncomment to see which component type is being tested

    # Allow ±3% tolerance
    assert abs(mean_failure_time - target_MTTF) < tolerance * target_MTTF
    
# -----------------------------------------------------------------------------
# TEST 2: Verify components have unique failure times
# -----------------------------------------------------------------------------
# PARAMETERS: 
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10}),
    
    # Weibull: mean ≠ scale, so we provide shape + scale or shape + desired MTTF
    (WeibullComponent, {"shape": 10, "MTTF": 10}),
])

# TEST FUNCTION
def test_comps_unique(comps):
    
    # grab the failure times of all the initialized comps
    all_failure_times = np.array([c.time_to_failure for c in comps])

    first_ft = all_failure_times[0]

    # check they have a unique failure time (not all equal to the first)
    assert not np.allclose(all_failure_times, first_ft)

# -----------------------------------------------------------------------------
# TEST 3: Test component reliability R(t) matches analytical solution
# -----------------------------------------------------------------------------
# PARAMETERS: (use same parameters as test 2)
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10}),
    
    # Weibull: mean ≠ scale, so we provide shape + scale or shape + desired MTTF
    (WeibullComponent, {"shape": 10, "MTTF": 10}),
])

# TEST FUNCTION
def test_comp_R_s_t(comps, plot_enabled):
    
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
    
    # Assertions
    assert abs(MTTF_sim - MTTF_analytical) < 0.1 * MTTF_analytical
    assert np.allclose(R_sim, R_t, atol=0.05)
    
    # Optional plotting
    if plot_enabled:
        plt.plot(time_points, R_sim, "o")
        plt.plot(time_points, R_t)
        plt.title(fr"$R_i(t): {comps[0]}$")
        plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=fr"$MTTF_sim ={MTTF_sim:.3f}$")
        plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=fr"$MTTF_analytical ={MTTF_sim:.3f}$")
        plt.show()
    
# -----------------------------------------------------------------------------
# TEST 4: Test exponential component availability matches analytical solution 
# -----------------------------------------------------------------------------    
# PARAMETERS:
@pytest.mark.parametrize("component_class, kwargs", [
    # Exponential: uses only MTTF
    (ExponentialComponent, {"MTTF": 10, "MTTR" : 4, "repairable": True}),
])

# TEST FUNCTION
def test_comp_availability(kwargs, comps):
    
    """ Determine if the availability of a repairable component matches the analytical solution. 
        (based on example 6.13 from 'System Reliability Theory' by Rausand and Hoyland) """
        
    # time array 
    total_time = 1000
    dt = 1
    time_array = np.arange(0, total_time, dt)
        
    # analytical availability
    MTTF = kwargs["MTTF"]
    MTTR = kwargs["MTTR"]
    failure_rate = 1 / MTTF
    repair_rate = 1 / MTTR
    A_t_analytical = repair_rate / (failure_rate + repair_rate) + (failure_rate / (failure_rate + repair_rate)) * sp.exp(-(failure_rate + repair_rate) * time_array)
    A_t_limit = repair_rate / (failure_rate + repair_rate) 
    
    # simulated availability
    comps.simulate(total_time, dt, repairable=True)
    
    working_state = max(comps[0].history)  # assuming history records 1 for working, 0 for failed
    failed_state = min(comps[0].history)

    # deterimine mean up time and down time
