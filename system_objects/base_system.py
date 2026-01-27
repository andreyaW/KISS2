# ----------------------------------------------------------------------
# base_system.py
# Implements a basic system model.
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
from component_objects.base_component import BaseComponent
from utils.helper_functions import is_numeric
from utils.enums import RepairMode

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
    
    # repair attributes
    repair_mode: RepairMode = RepairMode.ALL
    max_simultaneous_repairs: int | None = None  # None = unlimited
    repair_queue: list[BaseComponent] = field(default_factory=list, init=False)

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

    @abstractmethod
    def min_components_required(self) -> int:
        """Minimum number of working components required for system operation."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # OPERATION STEP
    def step(self, dt: float = 1.0, repairable: bool = False):
        self.current_time += dt

        for comp in self.components:
            comp.step(dt)

        self.state = self.structure_function()
        self.history = np.vstack([
            self.history,
            [self.current_time, self.state]
        ])
        
        if repairable:
            self.schedule_repairs() # will schedule repairs if needed


    # -------------------------------------------------------------------------
    # REPAIR LOGIC
    def components_needed_for_functionality(self) -> list[BaseComponent]:
        """
        Return the minimal set of failed components that must be repaired
        to restore system functionality, preferring lowest MTTR.
        """
        working = [c for c in self.components if c.state == WORKING_STATE]
        failed = [c for c in self.components if c.state == FAILED_STATE]

        required = self.min_components_required()
        deficit = max(0, required - len(working))

        if deficit == 0:
            return []

        failed_sorted = sorted(failed, key=lambda c: c.MTTR)
        return failed_sorted[:deficit]

    def select_repair_targets(self) -> list[BaseComponent]:
        failed = [c for c in self.components if c.state == FAILED_STATE]

        if self.repair_mode == RepairMode.ALL:
            return sorted(failed, key=lambda c: c.MTTR)

        if self.repair_mode == RepairMode.FUNCTIONAL:
            return self.components_needed_for_functionality()

        raise ValueError(f"Unknown repair mode: {self.repair_mode}")

        
    def schedule_repairs(self):
        """Schedule repairs based on repair mode and max simultaneous repairs."""

        # Select targets based on policy
        targets = self.select_repair_targets()

        # Add eligible targets to queue
        for comp in targets:
            if comp.state == FAILED_STATE and comp not in self.repair_queue:
                self.repair_queue.append(comp)

        # Count active repairs
        active_repairs = [
            c for c in self.components if c.state == REPAIR_STATE
        ]

        if self.max_simultaneous_repairs is None:
            available_slots = len(self.repair_queue)
        else:
            available_slots = max(
                0,
                self.max_simultaneous_repairs - len(active_repairs)
            )

        # Start repairs (MTTR-prioritized)
        for _ in range(available_slots):
            if not self.repair_queue:
                break

            comp = self.repair_queue.pop(0)

            # Safety guard
            if comp.state != FAILED_STATE:
                continue

            comp.start_repair(
                cv=getattr(comp, "CV_repair", 0.25),
                min_time=1.0
            )
   
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
            self.step(dt, repairable)

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
        ax.plot(df["time"], df[self.name], "*--k", lw=2, label=self.name)
        if plot_comps:
            for col in df.columns[1:]:
                if col != self.name:
                    ax.plot(df["time"], df[col], ":.", lw=.75, label=col)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title(f"{self.name} State History")