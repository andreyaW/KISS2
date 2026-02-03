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
    
    @abstractmethod
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
        pass

    # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    def health_reset(self, repair_duration: float):
        """Reset the object's health/state to initial conditions."""
        t_repair = 0.0
        while t_repair < repair_duration:
            self.current_time += self.dt
            t_repair += self.dt
            self.history = np.vstack([
                self.history,
                [self.current_time, -1]
            ])

        # Reset for new life
        self.state = 1
        self.history[-1] = [self.current_time, self.state]
        
        # sample a new TTF from failure distribution
        self.time_to_failure = self.sample_failure_time()



    def plot_history(self, ax=None):
        """Plot the history of the object's state over time."""
        if ax is None:
            ax = plt.gca()
    
        times = self.history[:, 0]
        states = self.history[:, 1]
        ax.plot(times, states, '--o', label=self.name) #drawstyle='steps-post',
        ax.set_title(f"State History of {self.name.title()}")
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fancybox=True,
            shadow=True
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.grid()
        
        return ax
        