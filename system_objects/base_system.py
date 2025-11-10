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
import pandas as pd

@dataclass
class BaseSystem(BasicObject, ABC):
    """
    Abstract base class for systems in long-term reliability simulation.
    Handles both components and subsystems.
    """

    name: str
    components: list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)  # overall system state: 1 = working, 0 = failed
    history: list[tuple[float, int]] = field(default_factory=list, init=False)
    all_histories: pd.DataFrame | None = field(default=None, init=False)

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
    def step(self, dt: float = 1.0, current_time: float | None = None) -> int:
        """
        Advance the simulation by dt time units.
        Updates all components and subsystems recursively, then records current state.
        """
        # --- Step components and subsystems ---
        for comp in self.components:
            if isinstance(comp, BaseSystem):
                comp.step(dt, current_time=current_time)
            else:
                comp.step(dt)

        # --- Update own state ---
        self.state = self.structure_function()

        # --- Record history for this step ---
        self.history.append((current_time + dt, self.state))
        for comp in self.components:
            if isinstance(comp, BaseComponent):
                comp.history.append((current_time + dt, comp.state))

        return self.state

    # -------------------------------------------------------------------------
    # SIMULATE FUNCTION
    # -------------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        """
        Run a time-based simulation loop for a duration `t_end` with step size `dt`.
        Uses the system's step() method to advance time and record history.
        Builds a pandas DataFrame (self.all_histories) containing all state histories.
        """
        num_steps = int(t_end // dt)
        current_time = 0.0

        # Initialize history for this system and all components/subsystems
        self.history = [(current_time, self.state)]
        for comp in self.components:
            comp.history = [(current_time, comp.state)]
            if isinstance(comp, BaseSystem):
                for subsys_comp in comp.components:
                    subsys_comp.history = [(current_time, subsys_comp.state)]

        BasicObject.logger.info(
            f"Starting simulation for {self.name}: duration={t_end}, dt={dt}, steps={num_steps}"
        )

        # --- Main simulation loop ---
        for _ in range(num_steps):
            self.step(dt, current_time=current_time)
            
            # --- Optional logging ---
            if self.history[-2][1] ==1 and self.state == 0:
                self.logger.info(f"{self.name}: System has failed at timestep {current_time}")
            current_time += dt

        BasicObject.logger.info(f"Completed simulation for {self.name} at t={current_time}")

        # Convert individual histories to numpy arrays
        self.history = np.array(self.history)
        for comp in self.components:
            comp.history = np.array(comp.history)

        # --- Build the all_histories DataFrame ---
        self._build_all_histories()
        for comp in self.components:
            if isinstance(comp, BaseSystem):
                for subsys_comp in comp.components:
                    subsys_comp.history = np.array(subsys_comp.history)
                comp._build_all_histories()

    # -------------------------------------------------------------------------
    # BUILD COMBINED HISTORY DATAFRAME
    # -------------------------------------------------------------------------
    def _build_all_histories(self):
        """
        Combine all component/subsystem histories into a single pandas DataFrame.
        Columns:
            time | <system_name> | <component_1_name> | <component_2_name> | ...
        """
        # Ensure histories exist
        if len(self.history) == 0:
            raise ValueError("No simulation history available. Run simulate() first.")

        # Create base DataFrame with system times
        times = self.history[:, 0]
        df = pd.DataFrame({"time": times, self.name: self.history[:, 1]})

        # Add component/subsystem histories as columns
        for comp in self.components:
            if hasattr(comp, "history") and len(comp.history) > 0:
                comp_states = comp.history[:, 1]
                df[comp.name] = comp_states

        self.all_histories = df

    # -------------------------------------------------------------------------
    # PLOTTING FUNCTION
    # -------------------------------------------------------------------------
    def plot_history(self, plot_comps: bool = False):
        """
        Plot the state history of the system (and optionally its components/subsystems).
        """
        # Use the DataFrame for plotting all comps
        for col in self.all_histories.columns[2:]:
            if col == self.name and not plot_comps:
                continue
            plt.plot(self.all_histories["time"], self.all_histories[col], label=col)
    
        # Use the DataFrame to plot the system overall history 
        plt.plot(self.all_histories["time"], self.all_histories[self.name], '--k', linewidth=2, label =self.name)
        
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
