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
class BaseSensor(BasicObject, ABC):
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
    quality: str | float = "Good"
    observation_probs: np.ndarray = field(init=False)
    history: list[tuple[float, int]] = field(default_factory=list, init=False)
    sensed_history: list[tuple[float, int]] = field(default_factory=list, init=False)

    # -------------------------------------------------------------------------
    # OBSERVATION PROBABILITY MATRIX
    # -------------------------------------------------------------------------
    def set_observation_probs(self) -> np.ndarray:
        """
        Set the observation probability matrix based on sensor quality.
        Each row corresponds to a true state; columns are sensed states.
        """
        if isinstance(self.quality, (float, int)):
            prob_correct = float(self.quality)
        else:
            q = self.quality.lower()
            if q == "good":
                prob_correct = 0.98
            elif q == "moderate":
                prob_correct = 0.75
            elif q == "bad":
                prob_correct = 0.5
            else:
                raise ValueError(f"Unknown sensor quality: {self.quality}")

        prob_incorrect = (1 - prob_correct) / 2
        obs = np.array([
            [prob_correct, prob_incorrect, prob_incorrect],
            [prob_incorrect, prob_correct, prob_incorrect],
            [prob_incorrect, prob_incorrect, prob_correct],
        ])
        return obs

    # -------------------------------------------------------------------------
    # ABSTRACT READ METHOD
    # -------------------------------------------------------------------------
    @abstractmethod
    def read(self, true_state: int) -> int:
        """
        Generate a single sensor reading given the true state.

        Parameters
        ----------
        true_state : int
            Actual state of the observed system/component (e.g., 0, 1, or 2)

        Returns
        -------
        int
            Observed (possibly incorrect) state according to sensor probabilities.
        """
        pass

    # -------------------------------------------------------------------------
    # RESET FUNCTION
    # -------------------------------------------------------------------------
    def reset(self):
        """Reset the sensor’s internal histories."""
        self.history.clear()
        self.sensed_history.clear()

    # -------------------------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------------------------
    def plot_history(self):
        """Plot the sensor’s recorded sensed states."""
        if len(self.sensed_history) == 0:
            print(f"No readings available for {self.name}")
            return

        hist = np.array(self.sensed_history)
        plt.plot(hist[:, 0], hist[:, 1], marker="*", linestyle="--", label=f"{self.name} ({self.quality})")
        plt.xlabel("Time")
        plt.ylabel("Sensed State")
        plt.title(f"Sensor Readings: {self.name}")
        plt.legend()
        plt.tight_layout()

    # -------------------------------------------------------------------------
    # ERROR ANALYSIS HELPER
    # -------------------------------------------------------------------------
    def compare_to_truth(self, true_history: np.ndarray):
        """
        Compare sensor readings to true states for diagnostic metrics.
        Returns counts of mismatches and classification errors.
        """
        sensed = np.array([s for _, s in self.sensed_history])
        true_states = np.array(true_history)

        sm = (sensed != true_states).sum()       # total mismatches
        fn = ((sensed == 0) & (true_states == 2)).sum()
        fp = ((sensed == 2) & (true_states == 0)).sum()
        fa = ((sensed == 1) & (true_states == 2)).sum()
        ma = ((sensed == 2) & (true_states == 1)).sum()

        return {"SM": sm, "FN": fn, "FP": fp, "FA": fa, "MA": ma}