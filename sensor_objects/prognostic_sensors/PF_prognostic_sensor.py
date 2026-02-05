from sensor_objects.prognostic_sensors.prognostic_sensor import PrognosticSensor
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PF_PrognosticSensor(PrognosticSensor):
    """
    Particle-filter-based failure time prognostics sensor
    with a single controllable sensor skill parameter.
    """

    def __init__(
        self,
        sensors,
        comp,
        n_particles=150,
        prior_sigma_frac=0.4
    ):
        # self.comp = comp
        self.true_ttf = comp.time_to_failure
        self.sensors = sensors
        self.n = n_particles

        sigma = prior_sigma_frac * comp.MTTF
        self.particles = np.clip(
            np.random.normal(comp.MTTF, sigma, self.n),
            0.1,
            None
        )
        self.weights = np.ones(self.n) / self.n

    def step(self, t):
        true_state = 1 if t < self.true_ttf else 0
        z = [s.generate_evidence(true_state) for s in self.sensors]

        for i in range(self.n):
            predicted_state = 1 if t < self.particles[i] else 0
            for obs, s in zip(z, self.sensors):
                self.weights[i] *= (
                    s.sensor_reliability if obs == predicted_state
                    else (1 - s.sensor_reliability)
                )

        self.weights += 1e-12
        self.weights /= np.sum(self.weights)

        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < 0.5 * self.n:
            idx = np.random.choice(self.n, self.n, p=self.weights)
            self.particles = self.particles[idx]
            self.weights[:] = 1.0 / self.n

    def predictFailure(self, t_array):
        for t in t_array:
            self.step(t)

        return np.sum(self.weights * self.particles)