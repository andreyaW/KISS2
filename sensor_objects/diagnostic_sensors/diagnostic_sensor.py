# from sensor_objects.base_sensor import BaseSensor
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class DiagnosticSensor(BaseSensor):
#     def __init__(self, sensor_reliability: float):
#         self.sensor_reliability = sensor_reliability

#     @abstractmethod
#     def diagnose(self, true_state:int, t:float):
#         "Return a sensor diagnosis based on sensor reliability and true state at time t"
#         pass

#     def sensorLogic(self, true_state, t):
#         return self.diagnose()
    
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class BaseDiagnosticSensor(ABC):
    reliability: float
    history: np.ndarray = field(default_factory=lambda: np.empty((0,2)))

    @abstractmethod
    def diagnose(self, true_state: int, t: float) -> int:
        pass

    def step(self, true_state: int, t: float) -> int:
        sensed = self.diagnose(true_state, t)
        self.history = np.vstack([self.history, [t, sensed]])
        return sensed
    

