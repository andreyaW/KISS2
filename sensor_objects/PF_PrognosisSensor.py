from sensor_objects.base_sensor import BaseSensor
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PF_PrognosisSensor(BaseSensor):
    """
    Particle-filter-based failure time prognosis sensor
    with a single controllable sensor skill parameter.
    """

    n_particles: int = 500

    # SINGLE CONTROL KNOB
    sensor_skill: float = 0.5   # ∈ (0, 1]

    prior_sigma_frac: float = 0.75
    resample_threshold: float = 0.5

    particles: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    initialized: bool = field(init=False, default=False)

    # ------------------------------------------------------------------
    def _initialize_particles(self):
        comp = self.attached_object

        mu = comp.MTTF
        sigma = self.prior_sigma_frac * mu

        self.particles = np.random.lognormal(
            mean=np.log(mu) - 0.5 * np.log(1 + (sigma / mu)**2),
            sigma=np.sqrt(np.log(1 + (sigma / mu)**2)),
            size=self.n_particles
        )

        self.weights = np.ones(self.n_particles) / self.n_particles
        self.initialized = True

    # ------------------------------------------------------------------
    def _measurement_likelihood(self, z, tf_particle, sigma):
        return np.exp(-0.5 * ((z - tf_particle) / sigma)**2)

    # ------------------------------------------------------------------
    def _resample(self):
        idx = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / self.n_particles)

    # ------------------------------------------------------------------
    def _measurement_sigma(self, t, tf_true):
        """
        Map sensor_skill → measurement noise

        sensor_skill → 0  ⇒ very large noise
        sensor_skill → 1  ⇒ very small noise
        """

        base_sigma = self.prior_sigma_frac * self.attached_object.MTTF

        # Optional: mild time dependence
        time_factor = 1.0
        if t >= tf_true:
            time_factor = 0.5   # post-failure sharpening

        eps = 1e-6
        skill = np.clip(self.sensor_skill, eps, 1.0)

        return base_sigma * time_factor * (1.0 / skill)

    # ------------------------------------------------------------------
    def predictFailure(self, t: float) -> float:
        comp = self.attached_object

        if not self.initialized:
            self._initialize_particles()

        tf_true = comp.time_to_failure

        # --------------------------------------------------------------
        # Measurement noise controlled ONLY by sensor_skill
        # --------------------------------------------------------------
        sigma_meas = self._measurement_sigma(t, tf_true)

        # Sensor measurement
        z = tf_true + np.random.normal(0.0, sigma_meas)

        # Importance weighting
        for i in range(self.n_particles):
            self.weights[i] *= self._measurement_likelihood(
                z=z,
                tf_particle=self.particles[i],
                sigma=sigma_meas
            )

        self.weights += 1e-12
        self.weights /= np.sum(self.weights)

        # Resample if needed
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.resample_threshold * self.n_particles:
            self._resample()

        # Posterior estimate
        return np.sum(self.weights * self.particles)

    # ------------------------------------------------------------------
    def sensorLogic(self, t):
        return self.predictFailure(t)
