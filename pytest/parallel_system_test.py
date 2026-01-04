import pytest
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent
from system_objects.parallel_system import ParallelSystem

# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def component_factory():
    """
    Factory that creates a list of Exponential or Weibull components.
    """
    def __factory__(dist_type="exp", params=None):
        comps = []

        if dist_type == "exp":
            for i, mttf in enumerate(params):
                comps.append(
                    ExponentialComponent(
                        name=f"Comp{i+1}",
                        MTTF=mttf,
                        MTTR=1
                    )
                )

        elif dist_type == "weib":
            for i, (mttf, k) in enumerate(params):
                comps.append(
                    WeibullComponent(
                        name=f"Comp{i+1}",
                        MTTF=mttf,
                        shape=k,
                        MTTR=1
                    )
                )

        return comps

    return __factory__


@pytest.fixture
def system_factory(component_factory):
    """
    Factory that produces N fresh SeriesSystem objects.
    """
    def __factory__(dist_type, params, n=1000):
        systems = []
        for i in range(n):
            comps = component_factory(dist_type, params)
            systems.append(ParallelSystem(f"Sys_{i}", comps))
        return systems
    return __factory__


@pytest.fixture
def expected_parallel():
    """
    Returns analytic R_s(t), f_s(t), z_s(t), and MTTF_s
    for a series system of exponential components.
    """
    def __factory__(dist_type, params):
        t = sp.Symbol("t", real=True, positive=True)

        # ----- exponential series -----
        if dist_type == "exp":
            lambdas = np.array([1/p for p in params])
            z_s = lambdas.sum()

            R_s_t = sp.exp(-z_s * t)
            f_s_t = z_s * sp.exp(-z_s * t)
            z_s_t = z_s
            MTTF_s = 1.0 / z_s

            return R_s_t, f_s_t, z_s_t, float(MTTF_s)
        
        elif dist_type =="weibull":
            
            pass
            raise NotImplementedError("Add Weibull analytic solution later.")

    return __factory__


# =============================================================================
# PARAMETERS
# =============================================================================
@pytest.mark.parametrize(
    "dist_type, kwargs",
    [
        ("exp", [10, 10, 10]),
        ("exp", [100, 100, 100]),
        ("exp", [50, 50, 50]),
        
    ]
)
# =============================================================================
# TEST
# =============================================================================
def test_series_system_exponential(dist_type, kwargs, system_factory, expected_series):
    """
    Monte-Carlo validation of system R(t), f(t), hazard z(t), MTTF.
    Also plots analytic vs simulated curves.
    """

    # -------- build systems --------
    N = 10000
    systems = system_factory(dist_type, kwargs, n=N)

    # -------- grab simulated system TTFs --------
    TTFs = np.array([sys.TTF() for sys in systems])
    
    # -------- analytic targets --------
    R_s_sym, f_s_sym, z_s_scalar, MTTF_analytical = expected_series(dist_type, kwargs)
    t_sym = list(R_s_sym.free_symbols)[0]

    time_points = np.linspace(0, np.max(TTFs), int(np.ceil(np.max(TTFs)/0.1)) + 1)

    R_analytic = np.array([float(R_s_sym.subs(t_sym, t)) for t in time_points])
    f_analytic = np.array([float(f_s_sym.subs(t_sym, t)) for t in time_points])
    z_analytic = np.full_like(time_points, z_s_scalar)

    # -------- Monte-Carlo R(t) --------
    R_sim = np.array([np.mean(TTFs > t) for t in time_points])

    # -------- Monte-Carlo MTTF --------
    MTTF_sim = TTFs.mean()

    # -------- Monte-Carlo f(t) (from KDE) --------
    kde = gaussian_kde(TTFs, bw_method=0.1)
    f_sim2 = kde(time_points)
    
    # -------- z(t) = f/R (from KDE)--------    
    n_samples = len(TTFs)          # number of observed TTFs
    rel_eps = max(1e-6, 1.0 / n_samples)   # or 0.01 if you need more aggressive floor

    z_sim2 = f_sim2 / np.maximum(R_sim, rel_eps)
    
    """
    # -------- Monte-Carlo f(t) (from histogram method) --------
    # compute bin edges from centers (works for uniform or nonuniform spacing)
    dt_left = time_points[1:] - time_points[:-1]
    # internal half-steps
    left_edges = time_points[:-1] - dt_left/2
    right_edges = time_points[1:] + dt_left/2
    # simpler robust approach:
    # Create edges as midpoints between consecutive centers, and extend ends
    midpoints = 0.5 * (time_points[:-1] + time_points[1:])
    edges = np.empty(len(time_points) + 1)
    edges[1:-1] = midpoints
    edges[0]      = time_points[0] - (midpoints[0] - time_points[0])
    edges[-1]     = time_points[-1] + (time_points[-1] - midpoints[-1])

    hist_counts, bin_edges = np.histogram(TTFs, bins=edges, density=True)
    
    # hist_counts now has length len(time_points) and corresponds to these centers
    f_sim = hist_counts
    t_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])  # should equal time_points

    # -------- Monte-Carlo z(t) = f/R (from histogram method)--------
    eps = 1e-10 #(prevent divide by zero error)
    z_sim = f_sim / np.maximum(R_sim, eps)
    
    # check if time_points and t_centers are the same
    assert np.allclose(t_centers, time_points, rtol=1e-12, atol=1e-16), \
        "t_centers do not match time_points!"
        
    """

    # -------------------------------------------------------------------------
    # OPTIONAL PLOTTING
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12,10))

    plt.subplot(3,1,1)
    plt.plot(time_points, R_analytic,"k", label="Analytic R(t)")
    plt.plot(time_points, R_sim, "o", markersize=3, label="Simulated R(t)")
    plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
    plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
    plt.legend()
    plt.title("Reliability R(t)")

    plt.subplot(3,1,2)
    plt.plot(time_points, f_analytic, "k", label="Analytic f(t)")
    # plt.plot(time_points, f_sim, "bo", markersize=3, label="Simulated f(t) (histogram method)")
    plt.plot(time_points, f_sim2, "rs", markersize=3, label="Simulated f(t) (KDE)")
    plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
    plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
    plt.legend()
    plt.title("PDF f(t)")

    plt.subplot(3,1,3)
    plt.plot(time_points, z_analytic,"k", label="Analytic hazard z(t)")
    # plt.plot(time_points, z_sim, "bo", markersize=3, label="Simulated hazard z(t) (histogram method)")
    plt.plot(time_points, z_sim2, "rs", markersize=3, label="Simulated hazard z(t) (KDE)")
    plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
    plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
    plt.legend()
    plt.title("Hazard z(t)")
    plt.tight_layout()
    plt.show()

    print("\nAnalytic MTTF:", MTTF_analytical)
    print("Simulated MTTF:", MTTF_sim)
    
    # check the MTTF is reasonably close to expectation 1% difference
    assert(np.isclose(MTTF_sim, MTTF_analytical, rtol=MTTF_analytical*0.01))

# def test_series_system_weibull(dist_type, kwargs, system_factory, expected_series):
#     pass
