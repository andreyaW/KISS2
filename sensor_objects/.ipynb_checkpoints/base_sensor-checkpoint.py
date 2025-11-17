"""
base_sensor.py

Defines an abstract base class for ship system sensors.
Provides a clear API for sensor initialization, reading, and performance tracking.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from objects import BasicObject


@dataclass
class BaseSensor(ABC):
    """
    Abstract base class for sensors used in reliability simulations.
    Sensors measure the state of components or systems with some uncertainty.

    Attributes
    ----------
    name : str
        Unique identifier for the sensor.
    history : list[tuple[float, int]]
        Records sensor correctness over time (1 = correct, 0 = incorrect).
    sensed_history : list[tuple[float, int]]
        Records raw sensor readings over time.
    """

    name: str
    attached_object : BasicObject
    history: np.ndarray[tuple[float, int]] = field(default_factory=list, init=False)
    sensed_history: np.ndarray[tuple[float, int]] = field(default_factory=list, init=False)

    # -------------------------------------------------------------------------
    # ABSTRACT READ METHOD
    # -------------------------------------------------------------------------
    @abstractmethod
    def sensorLogic(self, n_true_states):
        """ Applies a sensor logic to generate n readings from n given true states
        """
        pass
        
    
    def read(self, t_end:int, dt:float) -> int:
        """
        Generate a n sensor readings given the true state.

        Parameters
        ----------
        true_states : int
            Actual states of the observed system/component (e.g., 0, 1, or 2) over a number of steps

        Returns
        -------
        int
            Observed state (possibly incorrect) according to sensor probabilities.
        """
        """ Sense the state of the object the sensor is attached to. 
            Reading will be right or wrong depending on the sensors quality """
        
        num_steps = int(t_end/dt)

        # grab true states and times seperately
        true_states = self.attached_object.history[-num_steps:, 1]
        true_times = self.attached_object.history[-num_steps:, 0]

        # Set the shape of the true and sensed histories
        self.sensed_history = np.empty(shape=(0, 2))
        self.history = np.empty(shape=(0, 2))
        
        # Use child class' sensorLogic to generate readings
        sensed_history, sensor_state_history = self.sensorLogic(true_states, true_times)
        
        # Update histories
        self.sensed_history = np.append(self.sensed_history, sensed_history, axis=0)
        self.history = np.append(self.history, sensor_state_history, axis=0)


    # -------------------------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------------------------
    def plot_history(self, sensed_or_truth='sensed'):
        """Plot the sensor’s recorded sensed states."""

        if sensed_or_truth == 'sensed': 
            hist = np.array(self.sensed_history)
            title = f"Sensor Readings from {self.name.title()}"
        elif sensed_or_truth == 'truth':
            hist= np.array(self.history)
            title = f"State History of Sensor: {self.name.title()}"
        else: 
            print(f"please specify which history to plot for this sensor {{self.name}}, 'sensed' or 'truth' ")
        plt.plot(hist[:, 0], hist[:, 1], marker="*", linestyle="--", label=f"{self.name} ({self.quality})")
        
        plt.xlabel("Time")
        plt.ylabel("Sensed State")        
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        # set y-axis to always show the three possible states
        y_ticks = list(self.attached_object.states.keys())
        y_labels = [self.attached_object.states[i] for i in y_ticks]
        plt.yticks(y_ticks, y_labels)
