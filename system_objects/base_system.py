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

    # ABSTRACT METHOD: each subclass must define how system state depends on components
    @abstractmethod
    def structure_function(self) -> int:
        """Determine overall system state based on component (and subsystem) states."""
        pass

    # -------------------------------------------------------
    # STEP FUNCTION
    # -------------------------------------------------------
    def step(self, dt: float = 1.0) -> int:
        """
        Advance the simulation by dt time units.
        Updates all components (and subsystems), then evaluates system state.
        """
        # Step all *components* that are not systems
        for comp in self.components:
            if isinstance(comp, BaseSystem):
                # Subsystems manage their own time evolution
                continue
            comp.step(dt)

        # Recalculate overall system state based on updated component/subsystem states
        self.state = self.structure_function()

        # Log system failure if applicable
        if self.state == 0:
            self.logger.info(f"{self.name}: System has failed.")

        return self.state

    # -------------------------------------------------------
    # SIMULATE FUNCTION
    # -------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        """
        Run a time-based simulation loop for a duration `t_end` with step size `dt`.
        Handles nested subsystems recursively.
        """
        num_steps = int(t_end // dt)
        current_time = 0.0

        # Initialize history
        self.history = [(current_time, self.state)]
        for comp in self.components:
            comp.history = [(current_time, comp.state)]

        BasicObject.logger.info(
            f"Starting simulation for {self.name}: duration={t_end}, dt={dt}, steps={num_steps}"
        )

        # --- Step 1: Simulate any subsystems recursively ---
        for comp in self.components:
            if isinstance(comp, BaseSystem):
                BasicObject.logger.info(f"Simulating subsystem {comp.name} under {self.name}")
                comp.simulate(t_end, dt)

        # --- Step 2: Simulate this system’s own components ---
        for step_idx in range(1, num_steps + 1):
            current_time += dt
            self.step(dt)

            # Record state histories
            self.history.append((current_time, self.state))
            for comp in self.components:
                # Only record if component is not a subsystem (those already have their own histories)
                if not isinstance(comp, BaseSystem):
                    comp.history.append((current_time, comp.state))

        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")

        # Convert to numpy arrays for plotting convenience
        self.history = np.array(self.history)
        for comp in self.components:
            comp.history = np.array(comp.history)

    # -------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------
    def plot_history(self, plot_comps: bool = False):
        """
        Plot the state history of the system (and optionally its components/subsystems).
        """
        
        if plot_comps:
            for comp in self.components:
                times = comp.history[:, 0]
                states = comp.history[:, 1]
                plt.plot(times, states, label=comp.name) #drawstyle='steps-post',
                
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
        plt.title(f"{self.name.capitalize()} History")