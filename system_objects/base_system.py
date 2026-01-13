# """
# base_system.py

# System base class rewritten to match BaseComponent’s simulation API:
# - step(dt)
# - simulate(t_end, dt)
# - proper history as numpy array
# - initial state = 1
# """

# from dataclasses import dataclass, field
# from abc import ABC, abstractmethod
# from objects import BasicObject
# from component_objects.base_component import BaseComponent
# from pytest.sensed_comp_test import dt
# from utilities import is_numeric

# import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt
# import pandas as pd

# @dataclass
# class BaseSystem(BasicObject, ABC):
#     """
#     Abstract base class for systems in reliability simulation.
#     Mirrors the design of BaseComponent:
#     - state (0/1)
#     - current_time
#     - history array [[time, state], ...]
#     - step() and simulate()
#     """

#     name: str
#     components: list[BaseComponent] = field(default_factory=list)
#     state: int = field(default=1, init=False)                           # system state: 1 = working, 0 = failed
#     current_time: float = field(default=0.0, init=False)

#     # history will be a numpy array [[t, state], ...]
#     history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
#     all_histories: pd.DataFrame | None = field(default=None, init=False)

#     def __post_init__(self):
#         # system starts working
#         self.state = 1
#         self.current_time = 0.0
#         self.history = np.array([[0.0, self.state]], dtype=float)

#         # components already have their own history initialized in BaseComponent
#         for comp in self.components:
#             if hasattr(comp, "history") and comp.history.size == 0:
#                 comp.history = np.array([[0.0, comp.state]], dtype=float)

#     # -------------------------------------------------------------------------
#     # ABSTRACT METHODS: all subclasses must implement these
#     # -------------------------------------------------------------------------
#     @abstractmethod
#     def structure_function(self) -> int:
#         """
#         Return the system state (1 or 0) based on component states.
#         Example:
#             Series → min(component states)
#             Parallel → max(component states)
#             K-of-N → int(sum(states) >= k)
#         """
#         raise NotImplementedError
    
#     @abstractmethod
#     def R_s(self, t=None):
#        """
#         Compute system reliability symbolically (t=None) or numerically.
#         Works for scalar t or numpy arrays.
#        """
#        raise NotImplementedError 

#     # -------------------------------------------------------------------------
#     # STEP AND SIMULATE FUNCTIONS
#     # -------------------------------------------------------------------------
#     def step(self, dt: float = 1.0):
#         """
#         Advance time by dt:
#         - Step all components
#         - Recompute system state using structure_function()
#         - Append to history
#         """
#         self.current_time += dt

#         # --- Step all components ---
#         for comp in self.components:
#             comp.step(dt)

#         # --- Update system state ---
#         self.state = self.structure_function()

#         # --- Record system history ---
#         new_row = np.array([[self.current_time, self.state]])
#         self.history = np.vstack([self.history, new_row])

#         return self.state

#     def _advance_time_during_repair(self, repair_time):
#         self.current_time += repair_time

#         # system remains failed during repair
#         self.history = np.vstack([
#             self.history,
#             [self.current_time, 0]
#         ])

#         # after repair, recompute system state
#         self.state = self.structure_function()        

#     # Simulation loop
#     def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
#         """
#         Simulate system operation until t_end with time step dt.
#         If repairable=True, system repairs immediately upon failure.
#         """
#         while self.current_time < t_end:

#             # normal stepping always allowed
#             self.step(dt)

#             # check for repair condition
#             if repairable and self.state == 0:
#                 repair_time, _ = self.repair()
#                 self._advance_time_during_repair(repair_time)

#         self._build_all_histories()
    
    
#     def _build_all_histories(self):
#         """
#         Combine system and component histories into a single DataFrame:
#         columns = time, <system>, <component1>, ...
#         """

#         times = self.history[:, 0]
#         df = pd.DataFrame({"time": times, self.name: self.history[:, 1]})

#         for comp in self.components:
#             if hasattr(comp, "history"):
#                 comp_states = comp.history[:, 1]
#                 df[comp.name] = comp_states

#         self.all_histories = df

#     # -------------------------------------------------------------------------
#     # REPAIR FUNCTION
#     # -------------------------------------------------------------------------
#     def repair(self):
#         """
#         Repair the system.
#         """
#         for comp in self.components: 
#             if comp.state == 0:
#                 repair_time = comp.repair()
#         self.health_reset(repair_time)
#         return repair_time
        
        
#     # -------------------------------------------------------------------------
#     # RELIABILITY MODELING
#     # -------------------------------------------------------------------------
#     def f_s(self, t=None):
#         """
#         System PDF: f_s(t) = -dR/dt
#         """

#         # --- symbolic ---
#         if t is None:
#             ts = sp.symbols("t", positive=True)
#             R_expr = self.R_s(ts)       # symbolic
#             f_expr = -sp.diff(R_expr, ts)
#             return sp.simplify(f_expr)

#         # --- numeric ---
#         if is_numeric(t):
#             ts = sp.symbols("t", positive=True)
#             f_expr = self.f_s(None)     # symbolic form
#             f_num = sp.lambdify(ts, f_expr, "numpy")
#             return f_num(t)
        
#         raise TypeError("t must be None, float, or numpy array")

#     def z_s(self, t=None):
#         """
#         System hazard function: z(t) = f(t) / R(t)
#         """

#         # --- symbolic ---
#         if t is None:
#             ts = sp.symbols("t", positive=True)
#             f_expr = self.f_s(None)
#             R_expr = self.R_s(None)
#             return sp.simplify(f_expr / R_expr)

#         # --- numeric ---
#         if is_numeric(t):
#             eps = 1e-30
#             f = self.f_s(t)
#             R = self.R_s(t)
#             return f / np.maximum(R, eps)

#         raise TypeError("t must be None, float, or numpy array")
    
#     def MTTF_s(self):
#         """
#         MTTF = ∫₀^∞ R_s(t) dt
#         """
#         ts = sp.symbols("t", positive=True)
#         R_expr = self.R_s(ts)
#         return sp.integrate(R_expr, (ts, 0, sp.oo))

#     # -------------------------------------------------------------------------
#     # PLOTTING
#     # -------------------------------------------------------------------------
#     def plot_history(self, ax=None, plot_comps: bool = False):
#         """
#         Plot system and optionally component histories.
#         """
#         if ax is None:
#             ax = plt.gca()
        
#         if self.all_histories is None:
#             raise ValueError("Run simulate() before plotting.")

#         df = self.all_histories
        
#         # plot system
#         ax.plot(df["time"], df[self.name], "--k", lw=2, label=self.name)

#         # plot components
#         if plot_comps:
#             for col in df.columns[2:]:
#                 ax.plot(df["time"], df[col], ":",label=col)   
#         ax.legend(
#             loc="upper left",
#             bbox_to_anchor=(1.05, 1),
#             fancybox=True,
#             shadow=True
#         )
#         ax.set_title(f"{self.name} State History")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("State")

               
               
               
               
# ----------------------------------------------------------------------
# base_system.py
# ----------------------------------------------------------------------
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
from component_objects.base_component import BaseComponent
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from utilities import is_numeric

REPAIR_STATE = -1  # global constant for repair

@dataclass
class BaseSystem(BasicObject, ABC):
    name: str
    components: list[BaseComponent] = field(default_factory=list)
    state: int = field(default=1, init=False)
    current_time: float = field(default=0.0, init=False)
    dt: float = 1.0

    # histories
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    all_histories: pd.DataFrame | None = field(default=None, init=False)

    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.state = 1
        self.current_time = 0.0
        self.history = np.array([[0.0, self.state]], dtype=float)
        for comp in self.components:
            if hasattr(comp, "history") and comp.history.size == 0:
                comp.history = np.array([[0.0, comp.state]], dtype=float)

    # -------------------------------------------------------------------------
    @abstractmethod
    def structure_function(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def R_s(self, t=None):
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # OPERATING STEP
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
    # REPAIR SELECTION
    def components_to_repair(self):
        """Return list of components that must be repaired to restore system."""
        return [c for c in self.components if c.state == 0]

    # -------------------------------------------------------------------------
    # REPAIR LOGIC
    def _advance_time_during_repair(self, repair_times: dict, dt: float):
        """Advance time in discrete steps while components are under repair."""
        while max(repair_times.values()) > 0:
            self.current_time += dt
            self.state = REPAIR_STATE

            # Update repair clocks and component states
            for comp, remaining in repair_times.items():
                repair_times[comp] = max(0.0, remaining - dt)
                if remaining > 0:
                    comp.state = REPAIR_STATE

            # Record system and component histories
            self.history = np.vstack([self.history, [self.current_time, self.state]])
            for comp in self.components:
                comp.history = np.vstack([comp.history, [self.current_time, comp.state]])

    def repair(self, dt: float = 1.0):
        """Repair system: advance time, update histories, restore working states."""
        comps = self.components_to_repair()
        if not comps:
            return 0.0

        # Sample repair times
        repair_times = [(c, c.sample_repair_time()) for c in comps]

        # Advance system and component histories during repair
        while any(rt > 0 for _, rt in repair_times):
            self.current_time += dt
            self.state = REPAIR_STATE

            for i, (comp, remaining) in enumerate(repair_times):
                if remaining > 0:
                    comp.state = REPAIR_STATE
                    repair_times[i] = (comp, max(0.0, remaining - dt))
            
            # Record system and component histories
            self.history = np.vstack([self.history, [self.current_time, self.state]])
            for comp in self.components:
                comp.history = np.vstack([comp.history, [self.current_time, comp.state]])

        # Finalize repaired components
        for comp, _ in repair_times:
            comp.state = 1
            comp.time_to_failure = comp.sample_failure_time()
            comp.history = np.vstack([comp.history, [self.current_time, comp.state]])

        # Update system state at end of repair
        self.state = self.structure_function()
        self.history = np.vstack([self.history, [self.current_time, self.state]])

        return max(rt for _, rt in repair_times)


    # -------------------------------------------------------------------------
    # SIMULATION LOOP
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
        self.dt = dt
        while self.current_time < t_end:
            self.step(dt)
            if repairable and self.state == 0:
                self.repair(dt)
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