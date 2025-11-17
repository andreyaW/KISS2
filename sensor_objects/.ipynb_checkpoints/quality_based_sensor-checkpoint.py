from sensor_objects.base_sensor import BaseSensor
from dataclasses import dataclass

import numpy as np
from copy import deepcopy

@dataclass
class QualityBasedSensor(BaseSensor):
    quality: str | float = "Good" # Descriptor or numeric accuracy value (e.g., 'Good', 'Moderate', 'Bad', or 0.9).
    
    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.observation_probs = self.set_observation_probs()

    def set_observation_probs(self, comp_model= 'two_state') -> np.ndarray:
        """
        Set the observation probability matrix based on sensor quality.
        Each row corresponds to a true state; columns are sensed states.
        """
        if isinstance(self.quality, (float, int)):
            prob_correct = float(self.quality)
        else:
            q = self.quality.lower()
            if q == "good":
                prob_correct = 0.98
            elif q == "moderate":
                prob_correct = 0.75
            elif q == "bad":
                prob_correct = 0.5
            else:
                raise ValueError(f"Unknown sensor quality: {self.quality}")

        if comp_model == 'three state':
            prob_incorrect = (1 - prob_correct) / 2
            obs = np.array([
                [prob_correct, prob_incorrect, prob_incorrect],
                [prob_incorrect, prob_correct, prob_incorrect],
                [prob_incorrect, prob_incorrect, prob_correct],
            ])

        else: 
            prob_incorrect = (1 - prob_correct)
            obs = np.array([
                [prob_correct, prob_incorrect],
                [prob_incorrect, prob_correct],
                [prob_incorrect, prob_incorrect],
            ])
        return obs


    def sensorLogic(self, true_states, true_times):
        """ Quality based sensor generates sensor readings based its observation probability matrix
            
            Returns
            ----------
            sensed_history : ndarray
                The sensor readings of the component state
            sensor_state_history: ndarray
                The true health of the sensor (0 if broken/wrong reading, 1 if working/correct reading)
        """
        
        # initiailize sensor readings as a default (absurd) value
        default_init_val = 30
        sensed_history = np.full((len(true_states),2), default_init_val)
        sensor_state_history = np.empty_like(sensed_history)

        # # assuming initial reading is correct 
        # self.sensed_history = np.array([[true_times[0], true_states[0]]])
        # self.sensed_history = self.sensed_history.reshape(1,2)
        # self.history = np.array([[true_times[0], 1]])
        # self.history.reshape(1,2)

        # use obs matrix to determine the sensor readings
        sensed_states = deepcopy(true_states)
        for state in range(self.observation_probs.shape[0]):
            pass            
            mask = (true_states == state)
            if mask.any():
                sensed_states[mask] = np.random.choice(
                    [0, 1],
                    size=mask.sum(),
                    p=self.observation_probs[state]
                )

        # store sensed states to an array similar to the component history 
        sensed_history[:,0] = true_times
        sensed_history[:,1] = sensed_states

        # Sensor working history: 1 if match, 0 if not
        sensor_working = (sensed_history[:,1] == true_states)
        sensor_state_history[:,0] = true_times
        sensor_state_history[:,1] = sensor_working

        return sensed_history, sensor_state_history

    
        