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

REPAIR_STATE = -1  # global constant for repair

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
    def components_to_repair(self):
        """Return list of components that must be repaired to restore system."""
        return [comp for comp in self.components if comp.state == 0]
    
    def _advance_time_during_repair(self,max_repair_time, comps_not_under_repair, dt):
        """Advance time for the system and the components not being repaired in discrete steps while components are under repair."""
        
        # add steps to each component not under repair
        t_advanced = 0.0
        while t_advanced < max_repair_time:
            step_time = min(dt, max_repair_time - t_advanced)
            for comp in comps_not_under_repair:
                comp.step(step_time)
            t_advanced += step_time
            self.current_time += step_time
            self.history = np.vstack([
                self.history,
                [self.current_time, self.state]
            ])
        

    def repair(self, dt: float = 1.0):
        """ Sample and queue repair for the system and its components. """

        # determine which components need repair and how long it will take
        comps_to_repair = self.components_to_repair()
        comps_not_under_repair = [c for c in self.components if c not in comps_to_repair]
        repair_times = [(c, c.sample_repair_time()) for c in comps_to_repair]
        
        # use the max repair time and the repair limit to determine how components are being repaired
        max_repair_time = max(rt for c, rt in repair_times) if repair_times else 0.0
        repair_limit = 2 # arbitrary limit for simultaneous repairs
        
        if len(repair_times) <= repair_limit:
            # repair all components simultaneously
            for comp, repair_time in repair_times:
                comp.repair(repair_time)
                
            self._advance_time_during_repair(max_repair_time, comps_not_under_repair, dt)
    
        # else: 
        #     # repair components in batches according to the repair limit
        #     for i in range(0, len(repair_times), repair_limit):
        #         batch = repair_times[i:i+repair_limit]
        #         for comp, repair_time in batch:
        #             comp.repair(repair_time)                   
                                    
            # self._advance_time_during_repair(repair_times, dt)
        
        
    # -------------------------------------------------------------------------
    # SIMULATION LOOP
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
        
        # mark all the components as repairable if the system is repairable
        if repairable:
            for comp in self.components:
                comp.repairable = True
        
        self.dt = dt
        while self.current_time < t_end:
            self.step(dt)
            if repairable and self.state == 0:
                self.repair(dt)
                
            # print(f"  System State: {self.state}")
            # for comp in self.components:
            #     print(f"    {comp.name} State: {comp.state}")
            #     print(f" {comp.name} History Length: {len(comp.history)}")
                        
        # self._build_all_histories()

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