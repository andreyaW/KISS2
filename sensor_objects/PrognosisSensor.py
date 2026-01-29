from sensor_objects.base_sensor import BaseSensor
from component_objects.base_component import BaseComponent
from dataclasses import dataclass
import numpy as np

# @dataclass
# class PrognosisSensor(BaseSensor):
    
#     max_prob_predict_t_fail: float = .90   # Max probability of correctly predicting the correct time of failure
    
#     def exponential_prognosis_probability(
#         self, 
#         t: float,                                          # current time step
#         tf: float = None ,                                 # time of maximum probability of correct t_fail prediction
#         t0: float = None,                                  # time to start minimum prob. of correct t_fail prediction
#         min_value: float = None,                           # minimum probability of correct t_fail prediction 
#         ) -> float:
        
#         """
#         Exponential growth from min_value to max_value
#         starting at t0 and ending at tf.
#         """
#         # setting necessary variables
#         if tf == None: tf = self.attached_object.time_to_failure + self.attached_object.dt*10 # attains max prediction probability 10 steps after true failure time
#         if t0 == None: t0= 0.0
#         if min_value == None:min_value = 0.0 
#         max_value: float = self.max_prob_predict_t_fail,     # maximum value of correct t_fail prediction
            
#         # if t is before the exponential ramp up starts, return intial prognosis probability
#         if t <= t0:
#             return min_value

#         # if t after exponential ramp up has axed out, return max prognosis probability
#         if t >= tf:
#             return max_value

#         # Normalize time to [0, 1]
#         tau = (t - t0) / (tf - t0)

#         # Exponential curve normalized to hit endpoints exactly
#         # exp(0) = 1 → min_value
#         # exp(1) normalized → max_value
#         growth = (np.exp(tau) - 1) / (np.e - 1)

#         return min_value + growth * (max_value - min_value)

#     def predictFailure(self, t):
#         """
#         Generates a prediction of failure time for the component
#         based on the sensor's prognosis probability. (Gaussian Zero Mean Error)
#         """

#         # True failure time
#         t_fail = self.attached_object.time_to_failure

#         # Probability of correct prediction at time t (0–1)
#         prob_correct = self.exponential_prognosis_probability(t)

#         # --- Error model ---
#         # Maximum uncertainty (in time units)
#         max_error = t_fail * 0.5   # 50% of lifetime (tune this)

#         # Std dev shrinks as probability increases
#         sigma = max_error * (1.0 - prob_correct)

#         # Sample zero-mean error
#         prediction_error = np.random.normal(loc=0.0, scale=sigma)

#         # Predicted failure time
#         predicted_failure_time = t_fail + prediction_error

#         # Optional safety: failure time can't be negative
#         predicted_failure_time = max(predicted_failure_time, t)

#         return predicted_failure_time
        
        
@dataclass
class PrognosisSensor(BaseSensor):

    prob_predict_t_fail: float = 0.95  # max confidence

    prior_sigma_frac: float = 0.4      # uncertainty as fraction of MTTF

    def exponential_prognosis_probability(
        self,
        t: float,
        tf: float,
        t0: float = 0.0
    ) -> float:
        """
        Smooth exponential ramp from near 0 → max probability
        """
        if t <= t0:
            return 1e-3

        if t >= tf:
            return self.prob_predict_t_fail

        tau = (t - t0) / (tf - t0)
        growth = (np.exp(tau) - 1) / (np.e - 1)

        return growth * self.prob_predict_t_fail

    def predictFailure(self, t: float) -> float:
        """
        "Bayesian" prognosis of component failure time with delayed post-failure convergence.

        Behavior:
        - Before failure: estimate moves from MTTF toward true failure time.
        - After failure: estimate continues refining toward true failure time (no instantaneous collapse to truth).
        """

        comp = self.attached_object

        # True component failure time (latent)
        tf_true = comp.time_to_failure

        # PRIOR (population-level belief)
        mu_prior = comp.MTTF
        sigma_prior = self.prior_sigma_frac * mu_prior

        # ------------------------------------------------------------------
        # SENSOR CONFIDENCE
        # Before failure: confidence ramps up as tf approaches
        # After failure: confidence continues increasing toward 1.0
        # ------------------------------------------------------------------
        prob_correct = self.exponential_prognosis_probability(
            t=t,
            tf=tf_true
        )

        # After failure, force confidence to keep improving but prevent blow up
        if t >= tf_true:
            prob_correct = min(1.0, prob_correct + 0.1 * (t - tf_true) / comp.dt)
        prob_correct = np.clip(prob_correct, 1e-3, 0.999)

        # ------------------------------------------------------------------
        # MEASUREMENT UNCERTAINTY
        # After failure, uncertainty shrinks quickly but not instantly
        # ------------------------------------------------------------------
        sigma_meas = sigma_prior * (1 - prob_correct) / prob_correct

        # ------------------------------------------------------------------
        # SENSOR MEASUREMENT
        # Sensor observes evidence about *when* the failure occurred,
        # not that it occurred (that is already known post-failure).
        # in the future this can come from diagnostic sensors instead of random
        # ------------------------------------------------------------------
        z = tf_true + np.random.normal(0.0, sigma_meas)

        # ------------------------------------------------------------------
        # BAYESIAN UPDATE
        # Posterior mean is updated based on prior belief and new evidence
        # ------------------------------------------------------------------
        mu_post = (
            sigma_meas**2 * mu_prior +
            sigma_prior**2 * z
        ) / (
            sigma_meas**2 + sigma_prior**2
        )

        return mu_post

    def sensorLogic(self, t):
        """
        Generate a prognosis reading at a given timestep.
        Returns the diagnosed state (int).
        """
    
        return self.predictFailure(t)        
        