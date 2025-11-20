from sensor_objects import quality_based_sensor
from component_objects import component_exponential

import pytest
import numpy as np


@pytest.mark.parametrize("quality", [
    """ single sensor testing """
    ('good'), 
    ('moderate'),
    ('bad'),
    
])

@pytest.mark.parametrize("num_runs", 1000)
@pytest.mark.parametrize("tolerance", 0.03)
def test_sensor_accuracy(num_runs):
    """
    Generic test that verifies the simulated accuracy of a single sensor matches its input parameters
    """
    
    # initialize a comp
    test_comp = component_exponential(name="test_comp", MTTF = 100, MTTR= 10) 

    # attach a sensor to the component 
    sensor_accuracies = np.empty(num_runs)

    # simulate and sense with each initialized sensor
    for i in range(num_runs):
        
        
        
    # ensure all the sensors are not rpoducing the same history (identical random seeds)
        


    
    