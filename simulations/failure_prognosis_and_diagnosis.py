from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent

from sensor_objects.DiagnosisSensor import DiagnosisSensor
from sensor_objects.PrognosisSensor import PrognosisSensor

from sensed_objects import majority_vote_sensed_comp

import numpy as np


# initialize n component objects and add k sensors to each 
def initialize_comps_and_sensors(n: int, comp_type: str,  k: int):
    
    # default variables
    MTTF = 10
    MTTR = 2
    k = 4 # weibull comps will exhibit increasing failure rate (wear out)
    
    # add n comps to list object
    comps = [None for i in range(n)]
    for i in range(n):
        comp_name = f'Comp #{i}'
        if comp_type == 'exponential':
            comps[i] = ExponentialComponent(comp_name, MTTF, MTTR)
            
        elif comp_type == 'weibull':
            comps[i] = WeibullComponent(comp_name, MTTF, MTTR, k)      
    
    # add k sensors to each comp 
    sensed_comps = [None for i in range(n)]
    for comp in comps: 
        # sensors have 100% prognosis ability (to start)
        sensors = [PrognosisSensor(f"prog_sensor # {i+1}", comp, 1.0) for i in range(k)]
        sensed_comps[i] = majority_vote_sensed_comp(comp, sensors) 
        
    return sensed_comps

def predict_comp_state():
    pass

def main():
    sensed_comps = initialize_comps_and_sensors()
    
    t_array = np.linspace(0,100)
    for t in t_array():
        for sc in sensed_comps:
            # at a given time step, predict the components future failure time using its prognosisc sensors
            predict_comp_state(sc, t)
    

if __name__ == "__main__":
    main()
    
    