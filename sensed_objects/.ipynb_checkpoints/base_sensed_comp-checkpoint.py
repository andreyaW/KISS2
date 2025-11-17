"""
sensed_component.py

Defines the logic for a component object with multiple attached sensors
"""

from abc import ABC, abstractmethod
from objects import BasicObject
from dataclasses import dataclass, field

from component_objects import base_component
from sensor_objects import base_sensor

import numpy as np

@dataclass
class BaseSensedComponent(BasicObject, ABC):
    comp: base_component
    sensors: base_sensor
    history: np.ndarray[tuple[float, int]] = field(default_factory=list, init=False)
    sensed_history: np.ndarray[tuple[float, int]] = field(default_factory=list, init=False)

    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def aggregate_sensor_readings(self, t:int , dt:float, method:str ='majority_vote'): 
        pass

    # COMMON METHODS = methods shared by all subclasses (inherited as-is)           
    def step():
        pass


        
    def simulate(self, t_end: float, dt: float =1): #overwriting the basic object simulate
        """
        Run a time-based simulation loop for a duration `t_end` with time step `dt`.

        Parameters
        ----------
        t_end : float
            Total simulation time (in hours)
        dt : float
            Simulation step size. Determines numerical resolution.
        """

        # Calculate number of steps 
        num_steps = int(t_end // dt)
        
        # Initialize simulation
        current_time = 0.0
        self.history.append((current_time, self.comp.state)) # first state is the same as the components
        BasicObject.logger.info(f"Starting simulation for {self.comp.name}: duration={t_end}, dt={dt}, steps={num_steps}")

        # Initialize empty history arrays
        new_history = np.empty((num_steps, 2))
        new_sensor_readings = np.empty((num_steps,2)) 

        # Main simulation        
        comp.simulate(t_end,dt)
        new_history = comp.history
        self.
            
        # Append new history records
        self.history = np.append(self.history, new_history, axis=0) 
        self.sensed_history = np.append(self.sensed_history, new_sensor_readings, axis=0) 
        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")











