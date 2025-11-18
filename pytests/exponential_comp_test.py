# from component_objects import component_weibull
from component_objects.component_exponential import ExponentialComponent

import pytests
import numpy as np


def test_sample_failure_time(n, MTTF):
    
    # initialize n exponential components
    comps =[ExponentialComponent(MTTF=MTTF) for i in range(n)]
    comps_failure_times = [comp.time_to_failure for comp in comps]
    comps_failure_times = np.array(comps_failure_times)
    mean_failure_time = comps_failure_times.mean()
    
    # check that the simulated MTTF is within 10% of the set MTTF
    assert(abs(mean_failure_time-MTTF) < 0.1*MTTF)
    