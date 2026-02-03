from sensor_objects.diagnostic_sensors.quality_based_sensor import QualityBasedSensor
from component_objects.component_exponential import ExponentialComponent

import pandas as pd
import pytest
import numpy as np

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture()
def comp():
    return ExponentialComponent(name="test_comp", MTTF=50, MTTR=10)

@pytest.fixture()
def sensors(comp, quality, num_sensors):
    # Create N sensors with the input quality
    return [
        QualityBasedSensor(
            name=f"s{i}",
            attached_object=comp,
            quality=quality
        )
        for i in range(num_sensors)
    ]

@pytest.fixture()
def sim_hours():
    return 100

@pytest.fixture()
def dt():
    return 1

@pytest.fixture()
def num_sensors():
    return 1000

# --------------------------------------------------------------------------
# Parameter sets
# --------------------------------------------------------------------------
@pytest.mark.parametrize("run", range(3))
@pytest.mark.parametrize("tolerance, quality", 
                        [
                         (0.4, "Moderate")
                        ])

# --------------------------------------------------------------------------
# Test function
# --------------------------------------------------------------------------
def test_sensor_accuracy(run, quality, tolerance, num_sensors, sensors, comp, sim_hours, dt):
    """
    Test that N sensors with a given quality produce readings whose
    mean accuracy matches the expected probability of correct reading
    within a statistically justified tolerance.
    """

    # Simulate the component once
    comp.simulate(sim_hours, dt)
    
    # store comp history to referencable variables
    true_states = comp.history[1:, 1]
    time_steps = comp.history[1:, 0]
    T = len(time_steps) - 1           # number of timesteps
    
    # create a data frame to hold comp true states and the readings for each sensor
    cols = ["Time_Steps", "Comp_States"] + [sensor.name+"_Readings" for sensor in sensors] 
    simulation_data = pd.DataFrame(columns = cols)
    simulation_data["Time_Steps"] = time_steps
    simulation_data["Comp_States"] = true_states

    # Simulate all sensors generating readings
    for sensor in sensors:
        sensor.simulate()

        # Add sensor readings to dataframe
        simulation_data[sensor.name+"_Readings"] = sensor.sensed_history[:,1]

    # Vectorized computation of accuracies
    # Build a 2D array: rows = sensors, cols = timesteps
    sensed_matrix = np.array([sensor.sensed_history[:, 1] for sensor in sensors])
    
    # Compare against true states (broadcasting)
    matches = (sensed_matrix == true_states)
    accuracies = matches.mean(axis=1)  # mean accuracy per sensor
    mean_accuracy = accuracies.mean()  # mean across all sensors
    
    simulation_data.to_excel('test_sensor_accuracy_output.xlsx', index=False) # index=False prevents writing the DataFrame index as a column

    # Expected accuracy (sensor probability of correct observation)
    expected_accuracy = sensors[0].observation_probs[1, 1]
    
    # Acceptable range: expected accuracy + tolerance
    lower_bound = expected_accuracy - tolerance * expected_accuracy
    upper_bound = expected_accuracy + tolerance * expected_accuracy

    print(f"Quality: {quality}, Expected: {expected_accuracy:.2f}, "
          f"Mean: {mean_accuracy:.2f}, Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    assert (lower_bound <= mean_accuracy <= upper_bound)