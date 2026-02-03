from sensor_objects.base_sensor import BaseSensor
from dataclasses import dataclass
import numpy as np

@dataclass
class DiagnosisSensor(BaseSensor):
    def __init__(self, sensor_reliability: float):
        self.sensor_reliability = sensor_reliability

    def diagnose(self, true_state: int) -> int:
        if np.random.rand() < self.sensor_reliability:
            return true_state
        return 1 - true_state