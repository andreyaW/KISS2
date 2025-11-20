from sensor_objects.base_sensor import BaseSensor
from dataclasses import dataclass
import numpy as np

@dataclass
class QualityBasedSensor(BaseSensor):
    quality: str | float = "Good"

    def __post_init__(self):
        self.observation_probs = self.set_observation_probs()

    def set_observation_probs(self, comp_model='two_state') -> np.ndarray:
        
        # determine the probability the sensor gets a correct reading
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

            prob_incorrect = (1 - prob_correct)
            
            obs = np.array([
                [prob_correct, prob_incorrect],
                [prob_incorrect, prob_correct],
            ])
        return obs

    def sensorLogic(self, true_state: int, t: float) -> int:
        """
        Generate a single sensor reading at a given timestep.
        Returns the sensed state (int).
        """
        # extract probability vector for this true state
        probs = self.observation_probs[true_state]

        # sample a sensed state
        sensed_state = np.random.choice(len(probs), p=probs)

        return sensed_state
