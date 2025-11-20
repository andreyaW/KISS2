from dataclasses import dataclass
from sensed_objects.base_sensed_comp import BaseSensedComponent
import numpy as np
import pandas as pd

@dataclass
class MajorityVoteSensedComp(BaseSensedComponent):
    sensor_fusion_method: str = 'majority_vote'

    # -------------------------------------------------------------------------
    # AGGREGATE SENSOR READINGS
    # -------------------------------------------------------------------------
    def fuse_sensor_readings(self, t: float, dt: float = 1.0, method: str = 'majority_vote'):
        """
        Compute majority vote of the last reading of all attached sensors
        """
        # gather the last reading of each sensor (state only)
        readings = np.array([sensor.sensed_history[-1, 1] for sensor in self.sensors], dtype=int)

        # determine mode
        unique_vals, counts = np.unique(readings, return_counts=True)
        fused_state = unique_vals[np.argmax(counts)]

        return fused_state
    
    def __post_init__(self):
        "Defining the all states array to hold comp true and sensed states and true states and readings from all sensors"
        column_names = ['t', self.comp.name+' true state at time t', self.comp.name+' sensed state at time t'] + [sensor.name+' reading for time t' for sensor in self.sensors] + [sensor.name+' true state at time t' for sensor in self.sensors]
        
        self.all_states = pd.DataFrame(columns=column_names)

    # -------------------------------------------------------------------------
    # OPTIONAL: REPR
    # -------------------------------------------------------------------------
    def __repr__(self):
        return f"<MajorityVoteSensedComp: {self.comp.name}, method={self.sensor_fusion_method}>"
