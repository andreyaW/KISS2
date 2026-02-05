from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from component_objects.base_component import BaseComponent
from sensor_objects.diagnostic_sensors.diagnostic_sensor import DiagnosticSensor
from sensor_objects.prognostic_sensors.prognostic_sensor import PrognosticSensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class BaseSensedComponent(ABC):
    comp: BaseComponent
    diagnostic_sensors : list[DiagnosticSensor]
    prognostic_sensor: PrognosticSensor
    sensed_state: int = field(init=False)
    RUL: float = field(init=False)

    # intialize an empty history arrays with correct size
    history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)
    sensed_history: np.ndarray = field(default_factory=lambda: np.empty((0,2)), init=False)

    # empty pandas array which will be filled with all true and sensed states and RUL predictions
    all_states: pd.DataFrame = field(init=False)

    # @abstractmethod
    # def fuse_sensor_readings(self, t: float) -> int:
    #     """Return fused sensor reading from sensors at timestep t."""
    #     pass

    # @abstractmethod
    # def __post_init__(self):
    #     """ After defining the comp and it sensors, initialize the all states dataframe"""

    def __repr__(self):
        block_print_statement =f"""
        SensedComp: {self.comp.name}, 
        diagnostic sensor type  = {type(self.diagnostic_sensors[0])}, 
        # of Diagnostic Senosors = {len(self.diagnostic_sensors)}
        prognostic sensor type  = {type(self.prognostic_sensor)}
        """
        return block_print_statement


    
    # -------------------------------------------------------------------------
    # COMMONLY NEEDED METHODS
    # -------------------------------------------------------------------------
    def simulated_sensing_accuracy(self):
        return sum(self.sensed_history[:,1] == self.comp.history[1:,1])/len(self.sensed_history[:,1])
    
    def plotHistory(self):
        pass

    def plotMRLHistory(self):
        pass

