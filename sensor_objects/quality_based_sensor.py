from sensor_objects.base_sensor import BaseSensor
from dataclasses import dataclass

import numpy as np

@dataclass
class QualityBasedSensor(BaseSensor):
    quality : str | float # Descriptor or numeric accuracy value (e.g., 'Good', 'Moderate', 'Bad', or 0.9).
    observation_probs : np.ndarray # Probability matrix defining observation likelihoods given true states.
    
    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.observation_probs = self.set_observation_probs()