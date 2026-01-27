from sensor_objects.base_sensor import BaseSensor
from component_objects.base_component import BaseComponent
from dataclasses import dataclass
import numpy as np

@dataclass
class DiagnosisSensor(BaseSensor):
    observation_correct_prob: float = 0.95   # Probability of correct observation (sensed working when working)
    prob_miss_failure: float = 0.2                  # Probability of miss (not detecting failure)
        
    def observation_matrix(self, t):
        """ the probability of observing each state given the true state 
        sensor observation matrix:
        
                               (comp failed/ true state =0)      (comp working/ true state = 1)
        (sensor read failed)   [[ prob_hit ,                        prob_false_positive (false alarm) ]; 
        (sensor read working)   [ prob_false_negative (miss)        prob_correct_rejection ]]
        """
        
        # calculate time-dependent probabilities
        prob_correct_rejection = self.observation_correct_prob
        prob_correct_rejection = prob_correct_rejection * self.attached_object.R_t(t) / (prob_correct_rejection + self.prob_miss_failure)   
        # print(f"prob_correct_rejection: {prob_correct_rejection}")
        
        prob_false_positive = 1 - prob_correct_rejection
        
        prob_miss_failure = self.prob_miss_failure * (1-self.attached_object.R_t(t)) / (self.prob_miss_failure + self.observation_correct_prob)
        
        obs_matrix = np.array([
            [1 -prob_miss_failure, prob_false_positive],
            [prob_miss_failure, prob_correct_rejection]])
        
        return obs_matrix
    
    
    def sensorLogic(self, true_state: int, t: float) -> int:
        """
        Generate a single sensor reading at a given timestep.
        Returns the sensed state (int).
        """
        # extract probability vector for this true state
        probs = self.observation_matrix(t)
        probs = probs[:,true_state]

        # sample a sensed state
        sensed_state = np.random.choice(len(probs), p=probs)

        return sensed_state