from sensor_objects.prognostic_sensors import PrognosticSensor
from component_objects.base_component import BaseComponent
from dataclasses import dataclass
import numpy as np
        
@dataclass
class ExponentialPrognosticSensor(PrognosticSensor):

    prob_predict_t_fail: float = 0.01  # max confidence
    prior_sigma_frac: float = 0.01      # uncertainty as fraction of MTTF

    def exponential_prognostic_probability(
        self,
        t: float,
        tf: float,
        t0: float = 0.0
    ) -> float:
        """
        Smooth exponential ramp from near 0 → max probability
            Before failure: confidence ramps up as tf approaches
            After failure: confidence continues increasing toward 1.0
        """
        # initial failure prediction remains at comp MTTF
        if t <= t0:
            return 1e-3

        # sensor becomes more confident as time approaches true comp TTF
        elif t0 < t:
            tau = (t - t0) / (tf - t0)
            growth = (np.exp(tau) - 1) / (np.e - 1)
            prob_correct = growth * self.prob_predict_t_fail
            
            # After failure, force confidence to keep improving but prevent blow up
            if t >= tf:
                prob_correct = min(1.0, prob_correct + 0.1 * (t - tf) / self.attached_object.dt)
            prob_correct = np.clip(prob_correct, 1e-3, 0.999)
            
            return prob_correct


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

        # PRIOR
        mu_prior = comp.MTTF
        sigma_prior = self.prior_sigma_frac * mu_prior

        # SENSOR CONFIDENCE
        prob_correct = self.exponential_prognostic_probability(
            t=t,
            tf=tf_true
        )
        
        # MEASUREMENT UNCERTAINTY (After failure, uncertainty shrinks quickly but not instantly)
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