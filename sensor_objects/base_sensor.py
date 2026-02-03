from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from objects import BasicComponent


@dataclass
class BaseSensor(ABC):
    name: str
    comp: BasicComponent
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    sensed_history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)

    @abstractmethod
    def sensorLogic(self, true_state: int, t: float) -> int:
        """Return sensed state from true state at one timestep."""
        pass

    def step(self, t: float, true_state: int):
        """Perform one timestep of sensing, given true_state at time t."""
        sensed_state = self.sensorLogic(true_state, t)
        correct = int(sensed_state == true_state)

        self.sensed_history = np.vstack([self.sensed_history, [t, sensed_state]])
        self.history = np.vstack([self.history, [t, correct]])

    def simulate(self):
        """
        Simulate sensor output using the *already generated* component history.
        Assumes attached_object.history exists with columns [time, state].
        Sensor readings start after initial state (skip first row of comp history).
        """
        if not hasattr(self.attached_object, "history") or len(self.attached_object.history) == 0:
            raise ValueError(f"Component {self.attached_object.name} has no history. "
                             f"Run component.simulate(...) first.")

        comp_hist = self.attached_object.history   # shape (T, 2)
        if comp_hist.shape[0] < 2:
            raise ValueError("Component history must have at least 2 rows (initial state + timesteps).")

        # Skip the initial state (first row)
        times = comp_hist[1:, 0]
        true_states = comp_hist[1:, 1].astype(int)

        for t, true_state in zip(times, true_states):
            self.step(t, true_state)

    def plot_history(self, sensed_or_truth='sensed'):
        if sensed_or_truth == 'sensed':
            hist = np.array(self.sensed_history)
            title = f"Sensor Readings from {self.name.title()}"
        elif sensed_or_truth == 'truth':
            hist = np.array(self.history)
            title = f"State History of Sensor: {self.name.title()}"
        else:
            print(f"please specify 'sensed' or 'truth' for sensor {self.name}")
            return

        plt.plot(hist[:, 0], hist[:, 1], linestyle="--", label=f"{self.name}")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        y_ticks = list(self.attached_object.states.keys())
        y_labels = [self.attached_object.states[i] for i in y_ticks]
        plt.yticks(y_ticks, y_labels)
