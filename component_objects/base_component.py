# from dataclasses import dataclass, field
# from abc import ABC, abstractmethod
# from objects import BasicObject
# import numpy as np


# @dataclass
# class BaseComponent(BasicObject, ABC):
#     name: str
#     MTTF: float
#     MTTR: float
#     states: dict[int, str] = field(default_factory=lambda: {1: "working", 0: "failed"})

#     state: int = field(default=1, init=False)
#     time_to_failure: float = field(init=False)
#     dt = 1.0                                    # time step for simulation/ repair
#     current_time: float = field(default=0.0, init=False)
#     history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
#     repairable: bool = field(default=False, init=False)

#     # ----------------------------------------------------------------------
#     # ABSTRACT METHODS: ALL SUBCLASSES MUST HAVE THESE IMPLEMENTED
#     # ----------------------------------------------------------------------
#     @abstractmethod
#     def R_t(self, t) -> float:
#         "Suvivor Function: returns the reliability of the component at a desired time t"
#         raise NotImplementedError
    
#     @abstractmethod    
#     def z_t(self,t):
#         """Failure Rate Function: returns the failure rate of the component at a desired time t"""
#         pass
    
#     @abstractmethod
#     def f_t(self, t):
#         """Probability Density Function (pdf): The distribution of failure times based on the model type"""
#         pass
    
#     @abstractmethod
#     def sample_failure_time(self) -> float:
#         """ Returns a single failure time for a component using numpy random """
#         raise NotImplementedError
    
#     @abstractmethod
#     def __repr__(self):
#         raise NotImplementedError

#     # ----------------------------------------------------------------------
#     # COMMON METHODS: ALL SUBCLASSES INHERIT THESE FUNCTIONS
#     # ----------------------------------------------------------------------
    
#     def __post_init__(self):
#         if self.MTTF <= 0:
#             raise ValueError(f"{self.name}: MTTF must be positive.")

#         self.state = 1
#         self.current_time = 0.0
        
#         # Initialize history as an array with the initial state
#         self.history = np.array([[0.0, self.state]], dtype=float)

#     # ----------------------------------------------------------------------
#     def step(self, dt: float = 1.0):
#         """Advance time, update state, and append to history."""
#         self.current_time += dt
#         last_working_time = self.history[self.history[:, 1] == 1][-1, 0]
#         elapsed_time = self.current_time - last_working_time
#         if self.time_to_failure >= elapsed_time:
#             self.state = 0

#         # Append row [time, state] to history
#         new_row = np.array([[self.current_time, self.state]])
#         self.history = np.vstack([self.history, new_row])

#     # ----------------------------------------------------------------------
#     def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
#         """Run full simulation."""
#         t = 0.0
#         self.dt = dt
#         while len(self.history) < t_end / dt + 1:
#             self.step(dt)
            
#             if repairable and self.state == 0:
#                 self.repairable = True
#                 self.repair()
#             t += dt

#     # ----------------------------------------------------------------------
#     def repair(self, cv: float = 0.25, min_time: float = 5.0):
#         """ Sample repair time from lognormal distribution and update state history. 
#         (Repair State = -1), """
        
#         # sample repair time from lognormal distribution
#         def __sample_repair_time(self, cv, min_time) -> float:
#             sigma = np.sqrt(np.log(1.0 + cv**2))
#             mu = np.log(self.MTTR) - 0.5 * sigma**2

#             return max(
#                 np.random.lognormal(mu, sigma),
#                 min_time
#             )
        
#         repair_time = __sample_repair_time(self, cv, min_time)
#         self.current_time += repair_time
        
#         time_array = np.arange(
#             self.history[-1, 0] + self.dt,
#             self.current_time + self.dt,
#             self.dt
#         )
#         self.history = np.vstack([self.history, np.column_stack([time_array, np.ones_like(time_array)*-1])])
#         self.state = 1
#         self.current_time = self.history[-1, 0]
#         self.history = np.vstack([self.history, np.array([[self.current_time, self.state]])])
#         self.time_to_failure = self.sample_failure_time()

#     def plot_history(self, ax=None):
#         """Plot the history of the component's state over time."""
#         ax = super().plot_history(ax)
        
#         if self.repairable: 
#             ax.set_ylim(-1.1, 1.1)
            
#     # def reset_failure_time(self):
#     #     self.state = 1
#     #     self.current_time = 0.0
#     #     self.time_to_failure = self.sample_failure_time()

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from objects import BasicObject


@dataclass
class BaseComponent(BasicObject, ABC):
    name: str
    MTTF: float
    MTTR: float
    states: dict[int, str] = field(default_factory=lambda: {
        1: "working",
        0: "failed",
        -1: "repair"
    })

    state: int = field(default=1, init=False)
    current_time: float = field(default=0.0, init=False)
    life_time: float = field(default=0.0, init=False)   # ⬅️ life clock
    time_to_failure: float = field(init=False)
    dt: float = 1.0
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    repairable: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # ABSTRACT METHODS
    # ------------------------------------------------------------------
    @abstractmethod
    def sample_failure_time(self) -> float:
        pass

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.history = np.array([[0.0, 1]], dtype=float)

    # ------------------------------------------------------------------
    def step(self, dt: float = 1.0):
        self.current_time += dt

        if self.state == 1:  # working
            self.life_time += dt

            if self.life_time >= self.time_to_failure:
                self.state = 0

        self.history = np.vstack([
            self.history,
            [self.current_time, self.state]
        ])

    # ------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
        self.dt = dt
        self.repairable = repairable

        while self.current_time < t_end:
            self.step(dt)

            if repairable and self.state == 0:
                self.repair()

    # ------------------------------------------------------------------
    def repair(self, cv: float = 0.25, min_time: float = 5.0):
        sigma = np.sqrt(np.log(1.0 + cv**2))
        mu = np.log(self.MTTR) - 0.5 * sigma**2
        repair_time = max(np.random.lognormal(mu, sigma), min_time)

        t_repair = 0.0
        while t_repair < repair_time:
            self.current_time += self.dt
            t_repair += self.dt
            self.history = np.vstack([
                self.history,
                [self.current_time, -1]
            ])

        # Reset for new life
        self.state = 1
        self.life_time = 0.0
        self.time_to_failure = self.sample_failure_time()

        self.history = np.vstack([
            self.history,
            [self.current_time, 1]
        ])

    def plot_history(self, ax=None):
        ax = super().plot_history(ax)

        if self.repairable:
            ax.set_ylim(-1.1, 1.1)
            ax.grid()

        return ax