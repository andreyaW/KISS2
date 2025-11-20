"""
base_component.py

Defines an abstract base class for ship system components.
Provides a clear API for different reliability/failure models.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject

import numpy as np

@dataclass
class BaseComponent(BasicObject, ABC):
    """
    Abstract base class for components in long-term reliability simulation.
    """
    name: str
    MTTF: float
    MTTR: float
    states: dict[int, str] = field(default_factory=lambda: {1: "working", 0: "failed"})
    state: int = field(default=1, init=False)               # forces initial state to "working"
    time_to_failure: float = field(init=False)              # sampled time to failure
    history: np.ndarray[tuple[float, int]] = field(default_factory=list, init=False)  # (time, state) records

    def __post_init__(self):
        """Automatically initialize failure time when the component is created."""
        if self.MTTF <= 0:
            raise ValueError(f"{self.name}: MTTF must be positive.")
        self.reset_failure_time()

    # ABSTRACT METHODS = methods all subclasses MUST implement
    @abstractmethod
    def sample_failure_time(self) -> float:
        """
        Sample and return a random time to failure according to the component's failure model.
        """
        return NotImplementedError

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the component.
        """
        return NotImplementedError

    # COMMON METHODS = methods shared by all subclasses (inherited as-is)
    def reset_failure_time(self):
        """Reset component to working state and sample a new failure time."""
        self.time_to_failure = self.sample_failure_time()
        self.current_time = 0.0
        self.state = 1  # set state to "working"
        # logger.debug(f"{self.name}: failure time reset to {self.time_to_failure:.2f}")

    def step(self, dt: float = 1.0):
        """Advance the simulation by dt time units."""
        if self.state == 0:  # "failed"
            return
        self.current_time += dt
        if self.current_time >= self.time_to_failure:
            self.state = 0  # "failed"

# -------------------------------
    def repair(self):
        """Repair the component and reset failure time.
        DEFAULT: repair time is sampled from lognormal distribution."""
        
        pass # IMPLEMENT REPAIR LOGIC HERE