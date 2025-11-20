import pytest
import numpy as np

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent

# -----------------------------------------------------------------------------
# Define fixtures (variables/functions available to each test)
# -----------------------------------------------------------------------------

@ pytest.fixture
def comps(component_class, kwargs, n):
    # Build components
    return [ component_class(name=f"comp_{i}",**kwargs, MTTR=2) for i in range(n) ]

# -----------------------------------------------------------------------------
# Define parameters and parameter sets
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("tolerance, component_class, kwargs", [
    # Exponential: uses only MTTF
    (0.03, ExponentialComponent, {"MTTF": 10}),
    (0.03, ExponentialComponent, {"MTTF": 50}),
    (0.03, ExponentialComponent, {"MTTF": 100}),

    # Weibull: mean ≠ scale, so we provide shape + scale or shape + desired MTTF
    (0.03, WeibullComponent, {"shape": 1.0, "MTTF": 100}),  
    (0.03, WeibullComponent, {"shape": 1.0, "MTTF": 15}),
    (0.03, WeibullComponent, {"shape": 1.0, "MTTF": 250}),
])
@pytest.mark.parametrize("run", range(3))   # number of times to run each parameter set
@pytest.mark.parametrize("n", [10000])       # number of comps per test run

def test_sample_failure_time(tolerance, component_class, kwargs, run, n, comps):
    """
    Generic test that verifies the empirical MTTF of any component class.
    """
    # Extract the target MTTF
    target_MTTF = kwargs["MTTF"]

    # Collect failure times
    failure_times = np.array([c.time_to_failure for c in comps])    # uses the comps fixture
    mean_failure_time = failure_times.mean()

    # Allow ±3% tolerance
    assert abs(mean_failure_time - target_MTTF) < tolerance * target_MTTF