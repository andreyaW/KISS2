from sensor_objects.base_sensor import BaseSensor
from component_objects.base_component import BaseComponent
from dataclasses import dataclass
import numpy as np

@dataclass
class PrognosisSensor(BaseSensor):
    
    prob_diagnosis_at_t_fail: float = .90   # Probability of diagnosing at time of failure
    
    pass

    def bounded_exponential(
        t: float,
        t0: float,
        tf: float,
        min_value: float,
        max_value: float
    ) -> float:
        """
        Exponential growth from min_value to max_value
        starting at t0 and ending at tf.
        """

        if t <= t0:
            return min_value

        if t >= tf:
            return max_value

        # Normalize time to [0, 1]
        tau = (t - t0) / (tf - t0)

        # Exponential curve normalized to hit endpoints exactly
        # exp(0) = 1 → min_value
        # exp(1) normalized → max_value
        growth = (np.exp(tau) - 1) / (np.e - 1)

        return min_value + growth * (max_value - min_value)


    def prognosis_probability(self,
        t: float,
        t0: float,
        tf: float,
        min_value: float,
        max_value: float
        ) -> float:
        """
        Probability of diagnosing failure at time t_diag given failure at t_fail 
        Exponential growth from min_value to max_value
        starting at t0 and ending at tf.
        """

        if t <= t0:
            return min_value

        if t >= tf:
            return max_value

        # Normalize time to [0, 1]
        tau = (t - t0) / (tf - t0)

        # Exponential curve normalized to hit endpoints exactly
        # exp(0) = 1 → min_value
        # exp(1) normalized → max_value
        growth = (np.exp(tau) - 1) / (np.e - 1)

        return min_value + growth * (max_value - min_value)
    
        # """ Probability of diagnosing failure at time t_diag given failure at t_fail """
        # # Example: exponential increase in probability as we approach failure time
        # return self.bounded_exponential(
        #     t=t_diag,
        #     t0=t_fail - 10,  # start increasing 10 time units before failure
        #     tf=t_fail,       # reach max probability at failure time
        #     min_value=self.prob_diagnosis_at_t_fail,    # minimum probability
        #     max_value=self.prob_diagnosis_at_t_fail     # maximum probability
        # )
    
    def predictFailure(self, t):
        """ Generates a prediction of failure time for the component based on the sensors prognosis probability """
        
        # component true failure time and state
        t_fail = self.attached_object.time_to_failure
        true_state = 1 if t_fail>t else 0
        false_state = 1 if t_fail<t else 1
        
        # Calculate probability of diagnosing failure at current time
        prob_correct = self.prognosis_probability(t, 
                                                  t_fail-10,
                                                  t_fail, 
                                                  0.1, 
                                                  self.prob_diagnosis_at_t_fail)
            
        diagnosed_state = true_state if np.random.rand() < prob_correct else false_state
        return diagnosed_state
        
    def sensorLogic(self, t):
        """
        Generate a prognosis reading at a given timestep.
        Returns the diagnosed state (int).
        """
    
        return self.predictFailure(t)        
        