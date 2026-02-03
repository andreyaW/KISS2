from sensor_objects.diagnostic_sensor import DiagnosticSensor
from dataclasses import dataclass
import numpy as np

@dataclass
class BinaryDiagnosticSensor(DiagnosticSensor):
    """
    Binary diagnostic sensor
    """

    def __init__(self, sensor_reliability: float):    
        self.sensor_reliability = sensor_reliability

    def diagnose(self, true_state: int) -> int:
        if np.random.rand() < self.sensor_reliability:
            return true_state
        return 1 - true_state