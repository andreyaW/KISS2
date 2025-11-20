from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from objects import BasicObject
from component_objects.base_component import BaseComponent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class BaseSensedComponent(ABC):
    comp: BaseComponent
    sensors: list
    sensed_state :int = field(init=False)
    all_states: pd.DataFrame = field(init=False)
    # intialize an empty history arrays with correct size
    history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)
    sensed_history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)

    @abstractmethod
    def fuse_sensor_readings(self, t: float) -> int:
        """Return fused sensor reading from sensors at timestep t."""
        pass

    @abstractmethod
    def __post_init__(self):
        """ After defining the comp and it sensors, initialize the all states dataframe"""

    def step(self, t: float, dt: float = 1.0):
        # step the component and it's sensors
        self.comp.step(dt)
        for sensor in self.sensors:
            sensor.step(t)
        
        # use child class logic to fuse readings from multiple sensors
        fused_state = self.fuse_sensor_readings(t)
        
        # update history arrays
        self.history = np.vstack([self.history, [t, self.comp.state]])
        self.sensed_history = np.vstack([self.sensed_history, [t, fused_state]])
        
        # update sensed state
        self.sensed_state = self.sensed_history[-1,1]
        
        # update all states dataframe
        row = [t, self.comp.state, self.sensed_state]
        row = row + [sensor.sensed_history[-1,1] for sensor in self.sensors] + [sensor.sensed_history[-1,1] for sensor in self.sensors]
        self.all_states.loc[len(self.all_states)]=row # t, comp truth, comp sensed, sensor_i reading for i=1,2.. , sensor_i true state for i=1,2...
        
    # -------------------------------------------------------------------------
    # SIMULATION FUNCTION
    # -------------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        num_steps = int(t_end // dt)
        current_time = 0.0

        # Initialize history arrays
        self.history = np.empty((0,2))
        self.sensed_history = np.empty((0,2))

        for step_idx in range(num_steps+1):
            t = step_idx * dt
            self.step(t, dt)

    def plot_history(self):
        
        plt.plot(self.history[:,0], self.history[:,1])        
        plt.plot(self.sensed_history[:,0], self.sensed_history[:,1], '--', color = 'orange')

