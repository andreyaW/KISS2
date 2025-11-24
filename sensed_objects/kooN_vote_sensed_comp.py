from dataclasses import dataclass
from sensed_objects.base_sensed_comp import BaseSensedComponent
import numpy as np
import pandas as pd

@dataclass
class KooNVoteSensedComp(BaseSensedComponent):
    k: int
    sensor_fusion_method: str = 'k out of n vote'

    # -------------------------------------------------------------------------
    # AGGREGATE SENSOR READINGS
    def fuse_sensor_readings(self, t: float, dt: float = 1.0):
        """
        Fuse sensor readings using k-out-of-N voting.
        
        - If no state reaches k → choose randomly among tied highest-count states.
        - Handles edge cases like ties (e.g., [1,1,0,0]).
        """
        # grab the last reading from each sensor
        readings = np.array([sensor.sensed_history[-1, 1] for sensor in self.sensors], dtype=int)

        # count occurrences
        unique_vals, counts = np.unique(readings, return_counts=True)

        # find all states that meet or exceed k
        valid_states = unique_vals[counts >= self.k]

        # Case 1: One or more states meet k-out-of-n
        if len(valid_states) == 1:
            return int(valid_states[0])

        # Case 2: No state meets k → tie or dispersed readings
        # Example: [1,1,0,0] with N=4, k=3 → no one reaches 3
        # pick randomly between the tied most frequent readings
        max_count = np.max(counts)
        tied_states = unique_vals[counts == max_count]

        # choose randomly among tied states
        return int(np.random.choice(tied_states))
    
    def __post_init__(self):
        "Defining the all states array to hold comp true and sensed states and true states and readings from all sensors"
        column_names = ['t', self.comp.name+' true state at time t', self.comp.name+' sensed state at time t'] + [sensor.name+' reading for time t' for sensor in self.sensors] + [sensor.name+' true state at time t' for sensor in self.sensors]
        self.all_states = pd.DataFrame(columns=column_names)
        self.N = len(self.sensors)

    # -------------------------------------------------------------------------
    # OPTIONAL: STRING REPRESENTATION OF THE COMP
    # -------------------------------------------------------------------------
    def __repr__(self):
        return f"<SensedComp: {self.comp.name}, method={self.sensor_fusion_method}>"
