from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject

import numpy as np
import matplotlib.pyplot as plt

# GLOBAL CONSTANTS FOR COMPONENT STATES
WORKING_STATE = 1
FAILED_STATE = 0
REPAIR_STATE = -1

# =============================================================================
# BASE COMPONENT CLASS
# =============================================================================
@dataclass
class BaseComponent(BasicObject, ABC):
    # -------------------------------------------------------------------------
    # DATA ATTRIBUTES
    name: str
    MTTF: float
    MTTR: float
    states: dict[int, str] = field(default_factory=lambda: {1: "working", 0: "failed"})

    state: int = field(default=1, init=False)
    time_to_failure: float = field(init=False)
    dt = 1.0                                    # time step for simulation/ repair
    current_time: float = field(default=0.0, init=False)
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), init=False)
    repairable: bool = field(default=False, init=False)
    repair_end_time: float | None = field(default=None, init=False)

    # -------------------------------------------------------------------------
    # INITIALIZATION
    def __post_init__(self):
        if self.MTTF <= 0:
            raise ValueError(f"{self.name}: MTTF must be positive.")

        self.state = 1
        self.current_time = 0.0
        
        # Initialize history as an array with the initial state
        self.history = np.array([[0.0, self.state]], dtype=float)
    
    # ----------------------------------------------------------------------
    # ABSTRACT METHODS: aLL subclasses must implement these functions
    @abstractmethod
    def R_t(self, t) -> float:
        "Suvivor Function: returns the reliability of the component at a desired time t"
        raise NotImplementedError
    
    @abstractmethod    
    def z_t(self,t):
        """Failure Rate Function: returns the failure rate of the component at a desired time t"""
        pass
    
    @abstractmethod
    def f_t(self, t):
        """Probability Density Function (pdf): The distribution of failure times based on the model type"""
        pass
    
    @abstractmethod
    def sample_failure_time(self) -> float:
        """ Returns a single failure time for a component using numpy random """
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    
    # ----------------------------------------------------------------------
    # OPERATION STEP
    def step(self, dt: float = 1.0):
        self.current_time += dt

        if self.state == REPAIR_STATE:
            if self.current_time >= self.repair_end_time:
                self.state = WORKING_STATE
                self.repair_end_time = None
                self.time_to_failure = np.ceil(
                    self.sample_failure_time() / dt
                ) * dt

        elif self.state == WORKING_STATE:
            if self.current_time >= self.time_to_failure:
                self.state = FAILED_STATE

        self.history = np.vstack([
            self.history,
            [self.current_time, self.state]
        ])
        
    # def step(self, dt: float = 1.0):
    #     """Advance time, update state, and append to history."""
    #     # Advance time
    #     self.current_time += dt
        
    #     # Update state based on time to failure
    #     if  REPAIR_STATE in self.history[:,1]:
    #         if self.current_time - self.last_repair_time() >= self.time_to_failure:
    #             self.state = FAILED_STATE  # Component has failed
    #     else :
    #         if self.current_time >= self.time_to_failure:
    #             self.state = FAILED_STATE  # Component has failed
       
    #     # Append row [time, state] to history
    #     new_row = np.array([[self.current_time, self.state]])
    #     self.history = np.vstack([self.history, new_row])

    # -------------------------------------------------------------------------
    # REPAIR LOGIC
    def last_repair_time(self) -> float:
        """ Return the time of the last repair from history """
        repair_times = self.history[self.history[:,1] == REPAIR_STATE][:,0]
        if repair_times.size == 0:
            return 0.0
        else:
            return repair_times[-1]
    
    # sample repair time from lognormal distribution
    def sample_repair_time(self, cv, min_time) -> float:
        sigma = np.sqrt(np.log(1.0 + cv**2))
        mu = np.log(self.MTTR) - 0.5 * sigma**2

        return max(
            np.random.lognormal(mu, sigma),
            min_time
        )

    def start_repair(self, cv: float = 0.25, min_time: float = 1.0):
        """Initialize repair but do NOT advance time."""
        repair_time = self.sample_repair_time(cv, min_time)
        repair_time = np.ceil(repair_time / self.dt) * self.dt

        self.repair_end_time = self.current_time + repair_time
        self.state = REPAIR_STATE
                
    # def repair(self, t_end, cv: float = 0.25, min_time: float = 1.0):
    #     """ Sample repair time from lognormal distribution and update state history. 
    #     (Repair State = -1), """
                            
    #     repair_time = self.sample_repair_time(cv, min_time)
    #     repair_time = np.ceil(repair_time / self.dt) * self.dt # round up to nearest dt
    #     repair_end_time = self.current_time + repair_time
        
    #     while self.current_time != repair_end_time and self.current_time < t_end- self.dt:
    #         self.current_time += self.dt
            
    #         # Append repair state (-1) to history
    #         new_row = np.array([[self.current_time, REPAIR_STATE]])
    #         self.history = np.vstack([self.history, new_row])

    #     if self.current_time >= t_end:
    #         time_left_on_repair = repair_end_time - self.current_time
    #         return  # Exit if simulation time has ended
                            
    #     # After repair, set state to working and sample new failure time
    #     self.state = WORKING_STATE
    #     self.history[-1] = [self.current_time, self.state]  # update last entry to working state
    #     self.time_to_failure =  np.ceil(self.sample_failure_time() / self.dt) * self.dt # round up to nearest dt        
        
    # ----------------------------------------------------------------------
    # SIMULATION LOOP
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
            """Run full simulation."""

            self.dt = dt
            current_time = self.current_time
            while self.current_time < current_time+t_end:
                self.step(dt)
                
                if repairable and self.state == FAILED_STATE:
                    self.repairable = True
                    self.repair(t_end)

    # -------------------------------------------------------------------------
    # PLOTTING
    def plot_history(self, ax=None):
        """Plot the history of the component's state over time."""
        ax = super().plot_history(ax)
        

        # if (REPAIR_STATE in self.history[:,1]): 
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Repairing', 'Failed', 'Working'])
            
        # else:
        #     ax.set_ylim(-0.1, 1.1)
        
        plt.show()