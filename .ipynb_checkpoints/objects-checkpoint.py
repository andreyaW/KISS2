from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import logging
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# GLOBAL LOGGER CONFIGURATION
# -------------------------------------------------------
def get_global_logger():
    """Return a shared logger for all simulation objects."""
    logger = logging.getLogger("SimulationLogger")

    if not logger.handlers:  # Prevent duplicate handlers if imported multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# -------------------------------------------------------
# BasicObject CLASS DEFINITION
# -------------------------------------------------------
class BasicObject(ABC):
    """
    Defines the base class for all simulation entities (components, subsystems, systems, etc.).
    Provides consistent global logger and a time-based simulation loop.
    """
    logger = get_global_logger()    # Shared class-level logger

    def __init__(self, name: str):
        self.name = name
        self.history = np.array([])  # Records state changes over time (time, state)

    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def step(self, dt: float = 1.0):
        """
        Advance object state by dt time units.
        Each subclass must implement this.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return string representation of the object."""
        pass


    # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    def simulate(self, t_end: float, dt: float = 1.0):
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
        self.history.append((current_time, self.state))
        BasicObject.logger.info(f"Starting simulation for {self.name}: duration={t_end}, dt={dt}, steps={num_steps}")

        # Main simulation loop
        new_history = np.empty((num_steps, 2))  # Initialize empty history array
        for i in range(1, num_steps + 1):
            self.step(dt)
            current_time += dt
            new_history[i-1] = (current_time, self.state)
        self.history = np.append(self.history, new_history, axis=0) # Append new history records
        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")

    def plot_history(self):
        """Plot the history of the object's state over time."""
        times = self.history[:, 0]
        states = self.history[:, 1]
        plt.plot(times, states, label=self.name) #drawstyle='steps-post',
        plt.title(f"State History of {self.name.title()}")
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.ylim(-0.1, 1.1)
        plt.grid()