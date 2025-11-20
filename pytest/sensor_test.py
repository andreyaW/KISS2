from sensor_objects import quality_based_sensor
from component_objects import component_exponential

import pytest
import numpy as np

# -----------------------------------------------------------------------------
# Define fixtures (variables/functions available to each test)
# -----------------------------------------------------------------------------
@pytest.fixture()
def comp():
    return component_exponential(name="test_comp", MTTF = 100, MTTR= 10)

@pytest.fixture
def num_runs():
    return 1000

# @ pytest.fixture
# def sensor(sensor_class, kwargs, n):
#     # Build sensor objects
#     return [ sensor_class(name=f"s{i}",**kwargs) for i in range(n) ]

# -----------------------------------------------------------------------------
# Define parameters and parameter sets
# -----------------------------------------------------------------------------
# @pytest.mark.parametrize("quality", [
#     ('good'), 
#     ('moderate'),
#     ('bad'),
# ])

# @pytest.mark.parametrize("tolerance", [0.03])
# @pytest.mark.parametrize("n", [1])
# -----------------------------------------------------------------------------
# TEST FUNCTIONS
# -----------------------------------------------------------------------------

# def test_sensor_accuracy(num_runs):
#     """
#     Generic test that verifies the simulated accuracy of a single sensor matches its input parameters
#     """
#     pass
