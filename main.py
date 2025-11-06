from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent

import logging
import matplotlib.pyplot as plt
import numpy as np

def main():

    # set logging level here
    logging.basicConfig(level=logging.WARNING)

    # simulate 1000 exponential components and plot histogram of failure times
    failure_times = []
    for i in range(1000):
        pump = ExponentialComponent(name=f"Pump A-{i}", MTTF=100, MTTR=10)
        pump.simulate(t_end=500, dt=1)  # 500 time units, 1 per step
        failure_times.append(pump.history[-1, 0])  # record the time of last state change

    # plt.figure(figsize=(10, 5))
    # plt.hist(failure_times, bins=30, alpha=0.7, color='blue', edgecolor='black')
    # plt.xlabel("Time to Failure")
    # plt.ylabel("Number of Failures")
    # plt.title("Histogram of Failure Times for 1000 Exponential Components")
    # plt.grid()
    # plt.show()

    # simulate a weibull component
    pump2 = WeibullComponent(name="Pump B", MTTF=100, MTTR=10, shape=1.5)
    pump2.simulate(t_end=500, dt=1)  # 500 time units, 1 per step
    print(pump2)

    # plot the histories of both components in a single plot
    plt.figure(figsize=(10, 5))
    plt.plot(pump.history[:,0], pump.history[:,1], label="Pump A (Exponential)")
    plt.plot(pump2.history[:,0], pump2.history[:,1], label="Pump B (Weibull)")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("Component State Over Time")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()