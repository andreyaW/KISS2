# from sensor_objects.diagnostic_sensors.diagnostic_sensor import DiagnosticSensor
# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class BinaryDiagnosticSensor(DiagnosticSensor):
#     """
#     Binary diagnostic sensor
#     """
#     def __init__(self, sensor_reliability: float):    
#         self.sensor_reliability = sensor_reliability
#         # self.history = [0, self.comp.state]
#         self.history = np.empty((0, 2))

#     def diagnose(self, true_state: int, t: float) -> int:
#         if np.random.rand() < self.sensor_reliability:
#             return true_state
#         diagnosed_state = 1 - true_state
        
#         self.history = np.vstack([self.history, 
#                                  [t, diagnosed_state]])
        
#         return diagnosed_state
    

from dataclasses import dataclass
from diagnostic_sensor import BaseDiagnosticSensor
import numpy as np

@dataclass
class BinaryDiagnosticSensor(BaseDiagnosticSensor):

    def diagnose(self, true_state: int, t: float) -> int:
        if np.random.rand() < self.reliability:
            return true_state
        return 1 - true_state
