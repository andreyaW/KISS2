from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from objects import BasicObject


@dataclass
class BaseSensor(ABC):
    name: str
    attached_object: BasicObject
    history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)
    sensed_history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)

    @abstractmethod
    def sensorLogic(self, true_state: int, t: float) -> int:
        """Return sensed state from true state at one timestep."""
        pass

    def step(self, t: float):
        true_state = self.attached_object.state
        sensed_state = self.sensorLogic(true_state, t)
        correct = int(sensed_state == true_state)
        self.sensed_history = np.vstack([self.sensed_history, [t, sensed_state]])
        self.history = np.vstack([self.history, [t, correct]])

    def simulate(self, t_end: float, dt: float = 1.0):
        """
        Run the sensor over time, stepping in increments of dt.

        Parameters
        ----------
        t_end : float
            Total simulation time
        dt : float
            Timestep size
        """
        num_steps = int(t_end // dt)
        for step_idx in range(num_steps):
            t = step_idx * dt
            # # ensure the attached object updates before reading
            # if hasattr(self.attached_object, 'step'):
            #     self.attached_object.step(dt)
            self.step(t)

    # -------------------------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------------------------
    def plot_history(self, sensed_or_truth='sensed'):
        if sensed_or_truth == 'sensed': 
            hist = np.array(self.sensed_history)
            title = f"Sensor Readings from {self.name.title()}"
        elif sensed_or_truth == 'truth':
            hist = np.array(self.history)
            title = f"State History of Sensor: {self.name.title()}"
        else: 
            print(f"please specify which history to plot for this sensor {{self.name}}, 'sensed' or 'truth' ")
            return
        plt.plot(hist[:, 0], hist[:, 1], linestyle="--", label=f"{self.name}")
        plt.xlabel("Time")
        plt.ylabel("Sensed State")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        y_ticks = list(self.attached_object.states.keys())
        y_labels = [self.attached_object.states[i] for i in y_ticks]
        plt.yticks(y_ticks, y_labels)
