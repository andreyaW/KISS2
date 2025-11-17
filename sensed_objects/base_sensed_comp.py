"""
sensed_component.py

Defines the logic for a component object with multiple attached sensors
"""

from abc import ABC, abstractmethod
from objects import BasicObject
from dataclasses import dataclass

from component_objects import base_component
from sensor_objects import base_sensor

import numpy as np

@dataclass
class BaseSensedComponent(BasicObject, ABC):
    comp: base_component
    sensors: base_sensor
    history: np.ndarray
    sensed_history: np.ndarray
    
    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def aggregate_sensor_readings(self, t:int , dt:float, method:str ='majority_vote'): 
        pass
        # elif method=='KooN':
        #     # determine the KooN reading from the sensor readings
        #     pass
        # return aggregated_reading

    # # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    # def grab_truth_readings(self, t:int, dt:float):
    #     """ Grab a specific set of sensor readings over time from all the components sensors """
    #     for sensor in self.sensors:
             

    def step(self):
        # step the component
        self.comp.step()
        
        # read the component state with each sensor
        for sensor in self.sensors:
            sensor.read()
            
        # aggregate the sensor readings using a decided logic 
        self.sensed_history.append(self.aggregate_sensor_readings())
        self.history.append(comp.history[-1])    