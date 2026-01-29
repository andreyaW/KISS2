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
    
    
# ******* the logic below is how the diagnosis sensor should function (outputs a state not a prediction of failure time)     
    
# from sensor_objects.base_sensor import BaseSensor
# from component_objects.base_component import BaseComponent
# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class PrognosisSensor(BaseSensor):
    
#     prob_diagnosis_at_t_fail: float = .90   # Probability of diagnosing at time of failure
    
#     pass

#     def bounded_exponential(
#         t: float,
#         t0: float,
#         tf: float,
#         min_value: float,
#         max_value: float
#     ) -> float:
#         """
#         Exponential growth from min_value to max_value
#         starting at t0 and ending at tf.
#         """

#         if t <= t0:
#             return min_value

#         if t >= tf:
#             return max_value

#         # Normalize time to [0, 1]
#         tau = (t - t0) / (tf - t0)

#         # Exponential curve normalized to hit endpoints exactly
#         # exp(0) = 1 → min_value
#         # exp(1) normalized → max_value
#         growth = (np.exp(tau) - 1) / (np.e - 1)

#         return min_value + growth * (max_value - min_value)


#     def prognosis_probability(self,
#         t: float,
#         t0: float,
#         tf: float,
#         min_value: float,
#         max_value: float
#         ) -> float:
#         """
#         Probability of diagnosing failure at time t_diag given failure at t_fail 
#         Exponential growth from min_value to max_value
#         starting at t0 and ending at tf.
#         """

#         if t <= t0:
#             return min_value

#         if t >= tf:
#             return max_value

#         # Normalize time to [0, 1]
#         tau = (t - t0) / (tf - t0)

#         # Exponential curve normalized to hit endpoints exactly
#         # exp(0) = 1 → min_value
#         # exp(1) normalized → max_value
#         growth = (np.exp(tau) - 1) / (np.e - 1)

#         return min_value + growth * (max_value - min_value)
    
#         # """ Probability of diagnosing failure at time t_diag given failure at t_fail """
#         # # Example: exponential increase in probability as we approach failure time
#         # return self.bounded_exponential(
#         #     t=t_diag,
#         #     t0=t_fail - 10,  # start increasing 10 time units before failure
#         #     tf=t_fail,       # reach max probability at failure time
#         #     min_value=self.prob_diagnosis_at_t_fail,    # minimum probability
#         #     max_value=self.prob_diagnosis_at_t_fail     # maximum probability
#         # )
    
#     def predictFailure(self, t):
#         """ Generates a prediction of failure time for the component based on the sensors prognosis probability """
        
#         # component true failure time and state
#         t_fail = self.attached_object.time_to_failure
#         true_state = 1 if t_fail>t else 0
#         false_state = 1 if t_fail<t else 1
        
#         # Calculate probability of diagnosing failure at current time
#         prob_correct = self.prognosis_probability(t, 
#                                                   t_fail-10,
#                                                   t_fail, 
#                                                   0.1, 
#                                                   self.prob_diagnosis_at_t_fail)
            
#         diagnosed_state = true_state if np.random.rand() < prob_correct else false_state
#         return diagnosed_state
        
#     def sensorLogic(self, t):
#         """
#         Generate a prognosis reading at a given timestep.
#         Returns the diagnosed state (int).
#         """
    
#         return self.predictFailure(t)        
        