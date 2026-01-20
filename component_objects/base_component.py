from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
import numpy as np

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
        """Advance time, update state, and append to history."""
        # Advance time
        self.current_time += dt
        
        # Update state based on time to failure
        if  REPAIR_STATE in self.history:
            if self.current_time - self.last_repair_time() >= self.time_to_failure:
                self.state = 0  # Component has failed
        else :
            if self.current_time >= self.time_to_failure:
                self.state = 0  # Component has failed
       
        # Append row [time, state] to history
        new_row = np.array([[self.current_time, self.state]])
        self.history = np.vstack([self.history, new_row])

    # -------------------------------------------------------------------------
    # REPAIR LOGIC
    def last_repair_time(self) -> float:
        """ Return the time of the last repair from history """
        repair_times = self.history[self.history[:,1] == REPAIR_STATE][:,0]
        if repair_times.size == 0:
            return 0.0
        else:
            return repair_times[-1]
        
    def repair(self, cv: float = 0.25, min_time: float = 1.0):
        """ Sample repair time from lognormal distribution and update state history. 
        (Repair State = -1), """
        
        # sample repair time from lognormal distribution
        def __sample_repair_time(self, cv, min_time) -> float:
            sigma = np.sqrt(np.log(1.0 + cv**2))
            mu = np.log(self.MTTR) - 0.5 * sigma**2

            return max(
                np.random.lognormal(mu, sigma),
                min_time
            )
                            
        repair_time = __sample_repair_time(self, cv, min_time)
        repair_time = np.ceil(repair_time / self.dt) * self.dt # round up to nearest dt
        repair_end_time = self.current_time + repair_time
        print("TTF: ", self.time_to_failure)
        print("Repair Duration: ", repair_time)
        
        while self.current_time != repair_end_time:
            self.current_time += self.dt
            
            # Append repair state (-1) to history
            new_row = np.array([[self.current_time, -1]])
            self.history = np.vstack([self.history, new_row])

        # After repair, set state to working and sample new failure time
        self.state = 1
        self.time_to_failure =  np.ceil(self.sample_failure_time() / self.dt) * self.dt # round up to nearest dt
        print("New TTF after repair: ", self.time_to_failure, "\n")
        
        # Append working state (1) to history
        self.current_time = repair_end_time
        new_row = np.array([[self.current_time, self.state]])
        self.history = np.vstack([self.history, new_row])
        
    # ----------------------------------------------------------------------
    # SIMULATION LOOP
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
            """Run full simulation."""
            t = 0.0
            self.dt = dt
            while self.current_time < t_end:
                self.step(dt)
                
                if repairable and self.state == 0:
                    self.repairable = True
                    self.repair()
                t += dt

    # -------------------------------------------------------------------------
    # PLOTTING
    def plot_history(self, ax=None):
        """Plot the history of the component's state over time."""
        ax = super().plot_history(ax)
        
        if self.repairable: 
            ax.set_ylim(-1.1, 1.1)