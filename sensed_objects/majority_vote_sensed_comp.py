from sensed_objects.base_sensed_comp import BaseSensedComponent
from dataclasses import dataclass

import numpy as np

@dataclass
class MajorityVoteSensedComp(BaseSensedComponent):
    sensor_fusion_method: str = 'majority vote'
    
    def __repr__(self):
        pass
        # return super().__repr__()

    def aggregate_sensor_readings(self):
    
        all_sensor_readings = np.empty(len(self.sensors))
        
        # grab the last reading from each sensor
        for i,sensor in enumerate(self.sensors):
            all_sensor_readings[i] = sensor.sensed_history[-1]

        # determine the majority reading (mode of readings)
        unique_readings, counts = np.unique(all_sensor_readings, return_counts=True)
        mode_index = np.argmax(counts)
        mode_value = unique_readings[mode_index]
        aggregated_reading = mode_value
        
        print(f"The mode is: {mode_value}")
            