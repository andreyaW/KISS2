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

    def __repr__(self):
        block_print_statement =f"""
        SensedComp: {self.comp.name}, 
        diagnostic sensor type  = {type(self.diagnostic_sensors[0])}, 
        # of Diagnostic Senosors = {len(self.diagnostic_sensors)}
        prognostic sensor type  = {type(self.prognostic_sensor)}
        """
        return block_print_statement

    # @abstractmethod
    # def fuse_sensor_readings(self, t: float) -> int:
    #     """Return fused sensor reading from sensors at timestep t."""
    #     pass
    # 
    # -------------------------------------------------------------------------
    # OPERATION STEP
    def step(self, dt: float = 1.0, repairable: bool = False):
        """
        Advance component and sensors by one timestep
        """
        # 1. advance component true state
        self.comp.step(dt)

        t = self.comp.current_time
        true_state = int(self.comp.state)

        # 2. generate diagnostic sensor readings
        diag_readings = []
        for sensor in self.diagnostic_sensors:
            sensor.step(t, true_state)  # <-- use BaseSensor.step()
            diag_readings.append(sensor.sensed_history[-1, 1])
        # diag_readings = [
        #     sensor.diagnose(true_state, t)
        #     for sensor in self.diagnostic_sensors
        # ]

        # simple fusion rule: majority vote
        self.sensed_state = int(
            np.round(np.mean(diag_readings))
        )

        # 3. prognostic sensor RUL estimate
        # self.RUL = self.prognostic_sensor.sensorLogic(t, diag_readings)

        # 4. log histories
        self.history = np.vstack([
            self.history,
            [t, true_state]
        ])

        self.sensed_history = np.vstack([
            self.sensed_history,
            [t, self.sensed_state]
        ])

        # self.rul_history = np.vstack([
        #     self.rul_history,
        #     [t, self.RUL]
        # ])    
    
    def simulate(self, t_end: float, dt: float = 1.0, repairable: bool = False):
            """
            Run full sensed simulation using step()
            """
            t0 = self.comp.current_time
            while self.comp.current_time < t0 + t_end:
                self.step(dt, repairable)

            self._build_all_histories()

            
    # -------------------------------------------------------------------------
    def _build_all_histories(self):
        # base dataframe from component + fused sensing
        df = pd.DataFrame({
            "time": self.history[:, 0],
            "true_state": self.history[:, 1],
            "sensed_state": self.sensed_history[:, 1],
        })

        # merge each diagnostic sensor history by time
        for i, sensor in enumerate(self.diagnostic_sensors):
            sensor_df = pd.DataFrame(
                sensor.history,
                columns=["time", f"Sensor#{i+1}"]
            )

            df = df.merge(sensor_df, on="time", how="left")

        self.all_histories = df

    # -------------------------------------------------------------------------
    # USEFUL METHODS
    def simulated_sensing_accuracy(self):
        return sum(self.sensed_history[:,1] == self.comp.history[1:,1])/len(self.sensed_history[:,1])
    
    def plot_history(self, ax=None, plot_diagnostics: bool = False):
        if ax is None:
            ax = plt.gca()    
        if self.all_histories is None:
            raise ValueError("Run simulate() before plotting")
        df = self.all_histories
        
        # plot component truth history 
        ax.plot( df["time"], df["true_state"], label= "truth states")

        # plot diagnostics (majority readings)
        ax.plot(df["time"], df["sensed_state"], label= "majority sensed state")

        if plot_diagnostics:
            for i, diagnostic_sensor in enumerate(self.diagnostic_sensors):
                ax.plot(
                    df["time"],
                    df[f"Sensor#{i+1}"], 
                    "s", 
                    markersize= 12,
                    label=f"Diagnostic Sensor {i+1} Readings",
                    alpha=0.6)

        # plot prognostic sensor reading at time t
        
        
        # format figure
        ax.legend(loc="upper left", bbox_to_anchor=(1.05,1), fancybox=True, shadow=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.legend()
        
        # ax.set_title(f"{self.name} State History")
          
    def createMRLmovie(self):
        pass
                
    def plotMRLHistory(self):
        pass

