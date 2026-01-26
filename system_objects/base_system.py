# ----------------------------------------------------------------------
# base_system.py
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
from utilities import is_numeric
from component_objects.base_component import BaseComponent

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

# GLOBAL CONSTANTS FOR SYSTEM STATES
WORKING_STATE = 1
FAILED_STATE = 0
REPAIR_STATE = -1

# =============================================================================
# BASE SYSTEM CLASS
# =============================================================================
@dataclass
class BaseSystem(BasicObject, ABC):
    # -------------------------------------------------------------------------
    # DATA ATTRIBUTES
    name: str
    components: list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)
    current_time: float = field(default=0.0, init=False)
    dt: float = 1.0

    # histories
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    all_histories: pd.DataFrame | None = field(default=None, init=False)

    # -------------------------------------------------------------------------
    # INITIALIZATION
    def __post_init__(self):
        self.state = 1
        self.current_time = 0.0
        self.history = np.array([[0.0, self.state]], dtype=float)
        for comp in self.components:
            if hasattr(comp, "history") and comp.history.size == 0:
                comp.history = np.array([[0.0, comp.state]], dtype=float)

    # -------------------------------------------------------------------------
    # ABSTRACT METHODS
    @abstractmethod
    def structure_function(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def R_s(self, t=None):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # OPERATION STEP
    def step(self, dt: float = 1.0):
        """Advance time by dt: step all components and update system state."""
        self.current_time += dt

        # Step all components
        for comp in self.components:
            comp.step(dt)

        # Update system state
        self.state = self.structure_function()

        # Record system history
        self.history = np.vstack([self.history, [self.current_time, self.state]])

        return self.state

    # -------------------------------------------------------------------------
    # REPAIR LOGIC
    def repair(self):
        """Start repairs on failed components (no time advancement)."""
        for comp in self.components:
            if comp.state == FAILED_STATE:
                comp.start_repair(
                    cv=getattr(comp, "CV_repair", 0.25),
                    min_time=1.0
                )

    '''
    def repair(self, dt: float = 1.0):
        # """ Sample and queue repair for the system and its components. """
        comps_to_repair = [comp for comp in self.components if comp.state == 0]
        if not comps_to_repair:
            return  # No components need repair
        comps_not_under_repair = [comp for comp in self.components if comp not in comps_to_repair]        

        # Sample repair times for components
        repair_times = {}
        for comp in comps_to_repair:
            cv = getattr(comp, "CV_repair", 0.0)
            # min_time = getattr(comp, "min_repair_time", 1.0)
            repair_time = comp.sample_repair_time(cv, 1.0)
            repair_time = np.ceil(repair_time / dt) * dt  # round up to nearest dt
            repair_times[comp.name] = repair_time
        max_repair_time = max(repair_times.values())
        print(repair_times)

        # Set components to repair state over the repair duration
        repair_end_time = self.current_time + max_repair_time
        while self.current_time < repair_end_time+dt:
            self.current_time += dt
            for comp in self.components:
                
                # Update the state of components under repair
                if comp in comps_to_repair:
                    comp.state = REPAIR_STATE
                    comp.current_time = self.current_time
                    
                    # Check if repair is complete
                    if self.current_time >= (repair_end_time - repair_times[comp.name]):
                        comp.state = WORKING_STATE
                     
                    # Append row [time, state] to history
                    new_row = np.array([[comp.current_time, comp.state]])
                    comp.history = np.vstack([comp.history, new_row])
                   
                # Step components not under repair
                else:
                    comp.step(dt)

            # Update system state during repair
            self.state = self.structure_function()
            self.history = np.vstack([self.history, [self.current_time, self.state]])
                    
        # After repairs, update system state to reflect repaired components
        self.state = self.structure_function()
        self.history[-1] = [self.current_time, self.state]  # update last entry to current state
        '''
      
    # -------------------------------------------------------------------------
    # SIMULATION LOOP
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
        """Simulate the system operating for t_end/dt time steps. If repairable the system will undergo repair after failure, if not repairable the system will remain failed. 

        Args:
            t_end (float): last time of simulation
            dt (float, optional): time step (delta_t). Defaults to 1.0.
            repairable (bool, optional): Determines if the system is repairable. Defaults to False.
        """
        
        self.dt = dt
        end_time = self.current_time + t_end
        while self.current_time < end_time:
            self.step(dt)
            if repairable and self.state == 0:
                self.repair()                
        # After simulation, build all histories as a DataFrame
        self._build_all_histories()

    # -------------------------------------------------------------------------
    # HISTORY BUILDER
    def _build_all_histories(self):
        times = self.history[:, 0]
        df = pd.DataFrame({"time": times, self.name: self.history[:, 1]})
        for comp in self.components:
            if hasattr(comp, "history"):
                df[comp.name] = comp.history[:, 1]
        self.all_histories = df

    # -------------------------------------------------------------------------
    # RELIABILITY MODELING
    def f_s(self, t=None):
        """
        System PDF: f_s(t) = -dR/dt
        """

        # --- symbolic ---
        if t is None:
            ts = sp.symbols("t", positive=True)
            R_expr = self.R_s(ts)       # symbolic
            f_expr = -sp.diff(R_expr, ts)
            return sp.simplify(f_expr)

        # --- numeric ---
        if is_numeric(t):
            ts = sp.symbols("t", positive=True)
            f_expr = self.f_s(None)     # symbolic form
            f_num = sp.lambdify(ts, f_expr, "numpy")
            return f_num(t)
        
        raise TypeError("t must be None, float, or numpy array")

    def z_s(self, t=None):
        """
        System hazard function: z(t) = f(t) / R(t)
        """

        # --- symbolic ---
        if t is None:
            ts = sp.symbols("t", positive=True)
            f_expr = self.f_s(None)
            R_expr = self.R_s(None)
            return sp.simplify(f_expr / R_expr)

        # --- numeric ---
        if is_numeric(t):
            eps = 1e-30
            f = self.f_s(t)
            R = self.R_s(t)
            return f / np.maximum(R, eps)

        raise TypeError("t must be None, float, or numpy array")
    
    def MTTF_s(self):
        """
        MTTF = ∫₀^∞ R_s(t) dt
        """
        ts = sp.symbols("t", positive=True)
        R_expr = self.R_s(ts)
        return sp.integrate(R_expr, (ts, 0, sp.oo))

    # -------------------------------------------------------------------------
    # PLOTTING
    def plot_history(self, ax=None, plot_comps: bool = False):
        if ax is None:
            ax = plt.gca()
        if self.all_histories is None:
            raise ValueError("Run simulate() before plotting.")
        df = self.all_histories
        ax.plot(df["time"], df[self.name], "--k", lw=2, label=self.name)
        if plot_comps:
            for col in df.columns[1:]:
                if col != self.name:
                    ax.plot(df["time"], df[col], ":", label=col)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title(f"{self.name} State History")