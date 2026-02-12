# from sensor_objects.base_sensor import BaseSensor
# from dataclasses import dataclass
# import numpy as np
        
# @dataclass
# class PrognosticSensor(BaseSensor):
#     def __init__(self):
#         sensor_skill: float # probability that the sensor will estimate the correct TTF

#     def sensorLogic(self, t, diagnostic_readings) -> float:
#         """
#         Generate a RUL prognosis at a given timestep.
#         """
#         return self.predictFailure(t, diagnostic_readings) 

from abc import ABC, abstractmethod

class BasePrognosticSensor(ABC):

    @abstractmethod
    def step(self, t: float, diagnostic_readings: list[int]) -> float:
        pass