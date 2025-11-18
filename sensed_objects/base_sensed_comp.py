from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from objects import BasicObject
from component_objects.base_component import BaseComponent
# from sensor_objects import base_sensor
import numpy as np

@dataclass
class BaseSensedComponent(ABC):
    comp: BaseComponent
    sensors: list
    history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)
    sensed_history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)

    @abstractmethod
    def fuse_sensor_readings(self, t: float) -> int:
        """Return fused state from child sensors at timestep t."""
        pass

    def step(self, t: float, dt: float = 1.0):
        self.comp.step(dt)
        for sensor in self.sensors:
            sensor.step(t)
        fused_state = self.fuse_sensor_readings(t)
        self.history = np.vstack([self.history, [t, self.comp.state]])
        self.sensed_history = np.vstack([self.sensed_history, [t, fused_state]])

    # -------------------------------------------------------------------------
    # SIMULATION LOOP
    # -------------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        num_steps = int(t_end // dt)
        current_time = 0.0

        # Initialize history arrays
        self.history = np.empty((0,2))
        self.sensed_history = np.empty((0,2))

        for step_idx in range(num_steps):
            t = step_idx * dt
            self.step(t, dt)
