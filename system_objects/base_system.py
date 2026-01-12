"""
base_system.py

System base class rewritten to match BaseComponent’s simulation API:
- step(dt)
- simulate(t_end, dt)
- proper history as numpy array
- initial state = 1
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
from component_objects.base_component import BaseComponent
from utilities import is_numeric

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class BaseSystem(BasicObject, ABC):
    """
    Abstract base class for systems in reliability simulation.
    Mirrors the design of BaseComponent:
    - state (0/1)
    - current_time
    - history array [[time, state], ...]
    - step() and simulate()
    """

    name: str
    components: list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)                           # system state: 1 = working, 0 = failed
    current_time: float = field(default=0.0, init=False)

    # history will be a numpy array [[t, state], ...]
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    all_histories: pd.DataFrame | None = field(default=None, init=False)

    def __post_init__(self):
        # system starts working
        self.state = 1
        self.current_time = 0.0
        self.history = np.array([[0.0, self.state]], dtype=float)

        # components already have their own history initialized in BaseComponent
        for comp in self.components:
            if hasattr(comp, "history") and comp.history.size == 0:
                comp.history = np.array([[0.0, comp.state]], dtype=float)

    # -------------------------------------------------------------------------
    # ABSTRACT METHODS: all subclasses must implement these
    # -------------------------------------------------------------------------
    @abstractmethod
    def structure_function(self) -> int:
        """
        Return the system state (1 or 0) based on component states.
        Example:
            Series → min(component states)
            Parallel → max(component states)
            K-of-N → int(sum(states) >= k)
        """
        raise NotImplementedError
    
    @abstractmethod
    def R_s(self, t=None):
       """
        Compute system reliability symbolically (t=None) or numerically.
        Works for scalar t or numpy arrays.
       """
       raise NotImplementedError 

    # -------------------------------------------------------------------------
    # STEP AND SIMULATE FUNCTIONS
    # -------------------------------------------------------------------------
    def step(self, dt: float = 1.0):
        """
        Advance time by dt:
        - Step all components
        - Recompute system state using structure_function()
        - Append to history
        """
        self.current_time += dt

        # --- Step all components ---
        for comp in self.components:
            comp.step(dt)

        # --- Update system state ---
        self.state = self.structure_function()

        # --- Record system history ---
        new_row = np.array([[self.current_time, self.state]])
        self.history = np.vstack([self.history, new_row])

        return self.state

    # Simulation loop
    def simulate(self, t_end: float, dt: float = 1.0):
        t = 0.0
        while t < t_end:
            self.step(dt)
            t += dt

        # After simulation, build combined history DataFrame
        self._build_all_histories()
        
        # for the subsystems, also build their histories
        for comp in self.components:
            if hasattr(comp, "_build_all_histories"):
                comp._build_all_histories()
    
    
    def _build_all_histories(self):
        """
        Combine system and component histories into a single DataFrame:
        columns = time, <system>, <component1>, ...
        """

        times = self.history[:, 0]
        df = pd.DataFrame({"time": times, self.name: self.history[:, 1]})

        for comp in self.components:
            if hasattr(comp, "history"):
                comp_states = comp.history[:, 1]
                df[comp.name] = comp_states

        self.all_histories = df

    # -------------------------------------------------------------------------
    # RELIABILITY MODELING
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    def plot_history(self, ax=None, plot_comps: bool = False):
        """
        Plot system and optionally component histories.
        """
        if ax is None:
            ax = plt.gca()
        
        if self.all_histories is None:
            raise ValueError("Run simulate() before plotting.")

        df = self.all_histories
        
        # plot system
        ax.plot(df["time"], df[self.name], "--k", lw=2, label=self.name)

        # plot components
        if plot_comps:
            for col in df.columns[2:]:
                ax.plot(df["time"], df[col], ":",label=col)   
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fancybox=True,
            shadow=True
        )
        ax.set_title(f"{self.name} State History")
        ax.set_xlabel("Time")
        ax.set_ylabel("State")

               