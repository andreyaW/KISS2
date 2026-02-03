from sensor_objects.base_sensor import BaseSensor
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class DiagnosisSensor(BaseSensor):
    def __init__(self, sensor_reliability: float):
        self.sensor_reliability = sensor_reliability

    @abstractmethod
    def diagnose(self, true_state:int, t:float):
        "Return a sensor diagnosis based on sensor reliability and true state at time t"
        pass

    def sensorLogic(self, true_state, t):
        return self.diagnose()
