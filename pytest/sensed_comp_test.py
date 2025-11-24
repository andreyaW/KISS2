import pytest
from math import comb
import numpy as np

from sensor_objects.quality_based_sensor import QualityBasedSensor
from component_objects.component_exponential import ExponentialComponent
from sensed_objects.kooN_vote_sensed_comp import KooNVoteSensedComp


# -----------------------------------------------------------------------------
# Helper: Closed-form K-of-N correctness probability
# -----------------------------------------------------------------------------
def expected_kooN_fusion_correctness(N: int, K: int, p: float) -> float:
    """Compute probability that K-of-N fusion is correct when sensors are iid."""
    if K <= 0:
        return 1.0
    if K > N:
        return 0.0

    total = 0.0
    for i in range(K, N + 1):
        total += comb(N, i) * (p**i) * ((1 - p)**(N - i))

    return total


# -----------------------------------------------------------------------------
# Define fixtures (variables/functions available to each test)
# -----------------------------------------------------------------------------
@pytest.fixture
def num_comps():
    """Number of Monte-Carlo runs to simulate."""
    return 1000


@pytest.fixture
def T():
    """Total simulation duration (hours)."""
    return 100


@pytest.fixture
def dt():
    """Time step size (hours)."""
    return 1


@pytest.fixture
def kooN_sensedComps(num_comps):
    """
    Factory fixture:
    Creates `num_comps` independent KooN sensed components, each with its own
    exponential component and attached sensors.

    This drastically speeds the test by performing initialization once per test
    instead of inside the simulation loop.
    """
    def _factory(k, quality):
        N = 3  # number of sensors
        ensemble = []

        for _ in range(num_comps):
            comp = ExponentialComponent("test_comp", MTTF=10, MTTR=2)
            sensors = [
                QualityBasedSensor(f"s{j}", comp, quality=quality)
                for j in range(N)
            ]
            ensemble.append(KooNVoteSensedComp(comp, sensors, k=k))

        return ensemble

    return _factory


# -----------------------------------------------------------------------------
# Define parameters and parameter sets
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("quality, k", [
                        ("Good", 1), 
                        ("Good", 2),
                        ("Good", 3),
                        
                        ("Moderate", 1),
                        ("Moderate", 2),
                        ("Moderate", 3),
                       
                        ("Bad", 1),
                        ("Bad", 2),
                        ("Bad", 3),                         
])

# -----------------------------------------------------------------------------
# TEST 1: Monte-Carlo test of K-of-N sensed component correctness
# -----------------------------------------------------------------------------
def test_kooN_sensed_comp(k, quality,
                          T, dt, num_comps, kooN_sensedComps):
    """
    Monte-Carlo test:
    Simulate `num_comps` independent components with 3 sensors each.
    Fuse them using K-of-N logic.
    Compare the average simulated correctness to the analytic expectation.

    This test uses pre-generated components (via fixture) for speed.
    """

    # Retrieve prebuilt objects
    sensed_comps = kooN_sensedComps(k, quality)
    N = 3  # fixed sensor count

    accuracies = np.zeros(num_comps)
    
    # Simulate all ensemble elements
    for i,sensed_comp in enumerate(sensed_comps):
        sensed_comp.simulate(T, dt)
        accuracies[i]= sensed_comp.simulated_sensing_accuracy()
    simulated_mean = np.mean(accuracies)
    
    
    # Setting expectations based on probability of a single sensor being correct
    p = sensed_comps[0].sensors[0].observation_probs[1, 1]
    error_tolerance = 1-p
    expected_accuracy = expected_kooN_fusion_correctness(N, k, p)

    # Compute the error between expectation and simulated result
    rel_err = abs(simulated_mean - expected_accuracy) / expected_accuracy
    
    print(
        f"\nQUALITY={quality}, K={k}\n"
        f"Simulated={simulated_mean:.4f}, Expected={expected_accuracy:.4f}, "
        f"RelErr={rel_err:.4f}"
    )

    assert rel_err < error_tolerance


# def test_majority_vote_comp():
    
#     """ determine that a majority vote component has the same result as a kooN comp where k= int(N/2)+1"""