"""
base_system.py

Defines an abstract base class for ship systems.
Provides a clear API for different types of ship systems.
"""
from component_objects.base_component import BaseComponent
from objects import BasicObject
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class BaseSystem(BasicObject, ABC):
    """
    Abstract base class for systems in long-term reliability simulation.
    """
    name: str
    components : list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)                     # overall system state: 1 = working, 0 = failed
    history: list[int] = field(default_factory=list, init=False)  # record of system states over time

    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def structure_function(self) -> int:
        """Determine overall system state based on component states."""
        pass        
    
    # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    def step(self, dt: float = 1.0) -> int:
        """Advance the simulation by dt time units and return system state."""
        for component in self.components:
            component.step(dt)
        if self.state == 0:      # failed
            self.logger.info(f"{self.name}: System has failed.")
            return
        self.state = self.structure_function()

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
        for comp in self.components:
            comp.history.append((current_time, comp.state))
        BasicObject.logger.info(f"Starting simulation for {self.name}: duration={t_end}, dt={dt}, steps={num_steps}")

        # Main simulation loop
        new_history = np.empty((num_steps, 2))  # Initialize empty history array
        comp_history = {comp.name: np.empty((num_steps, 2)) for comp in self.components}  # Initialize component history
        for i in range(1, num_steps + 1):
            self.step(dt)
            current_time += dt
            new_history[i-1] = (current_time, self.state)
            for comp in self.components:
                comp_history[comp.name][i-1] = (current_time, comp.state)
        
        # append new history to records
        self.history = np.append(self.history, new_history, axis=0) # Append new history records
        for comp in self.components:
            comp.history = np.append(comp.history, comp_history[comp.name], axis=0)
        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")


    def plot_history(self, plot_comps: bool = False):
        if plot_comps:
            for component in self.components:
                component.plot_history()
        # set the linestyle and label for the system line
        plt.plot(self.history[:, 0], self.history[:, 1], linestyle='--', color = 'black', label=self.name)
        # super().plot_history()
        plt.legend()