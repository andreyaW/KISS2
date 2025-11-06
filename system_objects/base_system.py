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
    Handles both components and subsystems.
    """

    name: str
    components: list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)                     # overall system state: 1 = working, 0 = failed
    history: list[tuple[float, int]] = field(default_factory=list, init=False)

    # -------------------------------------------------------------------------
    # ABSTRACT METHOD -- all subclasses must implement
    # -------------------------------------------------------------------------
    @abstractmethod
    def structure_function(self) -> int:
        """
        Determine overall system state based on component (and subsystem) states.
        Must be implemented by subclasses (e.g., SeriesSystem, ParallelSystem, etc.)
        """
        pass

    # -------------------------------------------------------------------------
    # STEP FUNCTION
    # -------------------------------------------------------------------------
    def step(self, dt: float = 1.0) -> int:
        """
        Advance the simulation by dt time units.
        Updates all components (and subsystems), then evaluates system state.
        """
        for comp in self.components:
            comp.step(dt)   # Includes both BaseComponent and BaseSystem subclasses

        # Recalculate overall system state based on updated component/subsystem states
        self.state = self.structure_function()

        if self.state == 0:
            self.logger.info(f"{self.name}: System has failed.")

        return self.state

    # -------------------------------------------------------------------------
    # SIMULATE FUNCTION
    # -------------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        """
        Run a time-based simulation loop for a duration `t_end` with step size `dt`.
        Handles nested subsystems recursively in lockstep with parent system.
        """
        num_steps = int(t_end // dt)
        current_time = 0.0

        # Initialize histories
        self.history = [(current_time, self.state)]
        for comp in self.components:
            comp.history = [(current_time, comp.state)]

        BasicObject.logger.info(
            f"Starting simulation for {self.name}: duration={t_end}, dt={dt}, steps={num_steps}"
        )

        # --- Time stepping loop ---
        for step_idx in range(1, num_steps + 1):
            current_time += dt
            self.step(dt)

            # Record state histories
            self.history.append((current_time, self.state))
            for comp in self.components:
                comp.history.append((current_time, comp.state))

        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")

        # Convert histories to numpy arrays for easy plotting
        self.history = np.array(self.history)
        for comp in self.components:
            comp.history = np.array(comp.history)

    # -------------------------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------------------------
    def plot_history(self, plot_comps: bool = False):
        """
        Plot the state history of the system (and optionally its components/subsystems).
        """
        if plot_comps:
            for comp in self.components:
                times = comp.history[:, 0]
                states = comp.history[:, 1]
                plt.plot(times, states, label=comp.name)

        plt.plot(
            self.history[:, 0],
            self.history[:, 1],
            linestyle='--',
            color='black',
            label=self.name,
        )

        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            fancybox=True,
            shadow=True
        )
        plt.title(f"{self.name.title()} History")
        plt.xlabel("Time")
        plt.ylabel("State (1=Working, 0=Failed)")
        plt.tight_layout()
