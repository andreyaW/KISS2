from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from objects import BasicObject
import numpy as np


@dataclass
class BaseComponent(BasicObject, ABC):
    name: str
    MTTF: float
    MTTR: float
    states: dict[int, str] = field(default_factory=lambda: {1: "working", 0: "failed"})

    state: int = field(default=1, init=False)
    time_to_failure: float = field(init=False)
    current_time: float = field(default=0.0, init=False)
    history: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.MTTF <= 0:
            raise ValueError(f"{self.name}: MTTF must be positive.")

        self.reset_failure_time()

        # Initialize history as an array with the initial state
        self.history = np.array([[0.0, self.state]], dtype=float)

    # ----------------------------------------------------------------------
    @abstractmethod
    def sample_failure_time(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def reset_failure_time(self):
        self.state = 1
        self.current_time = 0.0
        self.time_to_failure = self.sample_failure_time()

    # ----------------------------------------------------------------------
    def step(self, dt: float = 1.0):
        """Advance time, update state, and append to history."""
        self.current_time += dt
        if self.current_time >= self.time_to_failure:
            self.state = 0

        # Append row [time, state] to history
        new_row = np.array([[self.current_time, self.state]])
        self.history = np.vstack([self.history, new_row])

    # ----------------------------------------------------------------------
    def simulate(self, t_end: float, dt: float = 1.0):
        """Run full simulation."""
        t = 0.0
        while t < t_end:
            self.step(dt)
            t += dt

    # ----------------------------------------------------------------------
    def repair(self):
        """Optional: implement repair logic later."""
        pass
