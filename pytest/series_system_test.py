import pytest
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from component_objects.component_exponential import ExponentialComponent
from component_objects.component_weibull import WeibullComponent
from system_objects.series_system import SeriesSystem

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
            systems.append(SeriesSystem(f"Sys_{i}", comps))
        return systems
    return __factory__


@pytest.fixture
def expected_series():
    """
    Returns analytic R_s(t), f_s(t), z_s(t), and MTTF_s
    for a series system.
    """
    def __factory__(dist_type, params):
        t = sp.Symbol("t", real=True, positive=True)

        # ======================================================
        # EXPONENTIAL SERIES ANAlYTIC SOLUTION
        # ======================================================
        if dist_type == "exp":
            lambdas = np.array([1 / MTTF for MTTF in params])
            z_s = lambdas.sum()

            R_s_t = sp.exp(-z_s * t)
            f_s_t = z_s * sp.exp(-z_s * t)
            z_s_t = z_s
            MTTF_s = 1.0 / z_s

            return R_s_t, f_s_t, z_s_t, float(MTTF_s)

        # ======================================================
        # WEIBULL SERIES ANALYTIC SOLUTION
        # (COMMON SHAPE PARAMETER)
        # ======================================================
        elif dist_type == "weib":
            mttfs = [p[0] for p in params]
            shapes = [p[1] for p in params]

            # require identical shape
            if len(set(shapes)) != 1:
                raise NotImplementedError(
                    "Analytic Weibull series solution requires identical shape parameters"
                )

            k = shapes[0]

            # compute scales θ_i from MTTF_i
            thetas = [
                mttf / sp.gamma(1 + 1 / k)
                for mttf in mttfs
            ]

            # A = sum(theta_i^{-k})
            A = sum(theta ** (-k) for theta in thetas)

            # analytic expressions
            R_s_t = sp.exp(-A * t ** k)
            f_s_t = sp.diff(1 - R_s_t, t)
            z_s_t = sp.simplify(f_s_t / R_s_t)

            # analytic MTTF
            MTTF_s = float(sp.gamma(1 + 1 / k) / (A ** (1 / k)))

            return R_s_t, f_s_t, z_s_t, MTTF_s

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    return __factory__


# # =============================================================================
# # WEIBULL SERIES SYSTEM TEST 1
# # all components having the same shape and scale parameters
# #   scale_param = θ = η 
# #   shape param = α = k
# #   MTTF = scale_param * gamma(1+1/shape_param)
# # =============================================================================
# # PARAMETERS
# # =============================================================================
# @pytest.mark.parametrize(
#     "dist_type, kwargs",
#     [
#         ("weib", [(7701.4263, 2.25), (7701.4263, 2.25), (7701.4263, 2.25), (7701.4263, 2.25), (7701.4263, 2.25)]), # Weibull parameters : (MTTF, shape k)
#     ]
# )

# def test_series_system_weibull_identical(dist_type, kwargs, system_factory, expected_series):
#     """
#     Check that the resultant Rs(t), z_s(t), and fs(t) matches analytic solution
#     (analytic solution only available if all components have the same shape parameter)
    
#     """
#     N = 12000
#     systems = system_factory("weib", kwargs, n=N)

#     # grab simulated system lifetimes
#     TTFs = np.array([sys.TTF() for sys in systems])
#     assert np.all(TTFs > 0), "Non-positive system lifetimes detected"

#     # generate a time array for Rs(t) to be plotted
#     t_max = np.percentile(TTFs, 95)
#     time_points = np.linspace(0.1, t_max, 250)

#     # create a figure
#     plt.figure()

#     # simulated Reliability R(t)
#     R_sim = np.array([np.mean(TTFs > t) for t in time_points])
#     plt.plot(time_points, R_sim, "k--", label= "simulated")
    
#     # simulated MTTF
#     simulated_MTTF = np.mean(TTFs)
#     plt.axvline(x=simulated_MTTF, color='black', linestyle='-', linewidth=1, label=r"MTTF$_{simulated}$ " + f"={simulated_MTTF:.3f}")
   
#     # determine and plot analytic solutions   
#     R_s_sym, f_s_sym, z_s_sym, analytic_MTTF = expected_series(dist_type, kwargs)
#     t_sym = list(R_s_sym.free_symbols)[0]

#     analytic_Rs = np.array([float(R_s_sym.subs(t_sym, t)) for t in time_points])
    
#     plt.axvline( x=analytic_MTTF, color='green', linestyle=':', linewidth=2,label=rf"MTTF$_{{analytical}}$ = {analytic_MTTF:.3f}")
#     plt.plot(time_points, analytic_Rs, "g.", label = "analytical")
#     plt.legend()
#     plt.show()
    

# =============================================================================
# WEIBULL SERIES SYSTEM TEST 2
# Series system of three Weibull components:
#   - Comp1: k < 1  (decreasing failure rate)
#   - Comp2: k = 1  (constant failure rate / exponential)
#   - Comp3: k > 1  (increasing failure rate)
# =============================================================================
# PARAMETERS
# =============================================================================
@pytest.mark.parametrize(
    "dist_type, kwargs",
    [
        ("weib", [(1000, 0.6),      # Decreasing Failure Rate
                  (1000, 1.0),      # Constant Failure Rate (exponential)
                  (1000, 2.20)]),   # Increasing Failure Rate
    ]
)

def test_series_system_weibull_mixed_hazards(
    system_factory, dist_type, kwargs
):
    """
    The system hazard z_s(t) = f_s(t) / R_s(t) should exhibit
    a bathtub-like shape for mixed Weibull components
    (DFR + CFR + IFR).
    """

    # -----------------------------
    # Monte Carlo simulation
    # -----------------------------
    N = 12000
    systems = system_factory(dist_type, kwargs, n=N)

    TTFs = np.array([sys.TTF() for sys in systems])
    assert np.all(TTFs > 0), "Non-positive system lifetimes detected"

    # -----------------------------
    # Time grid
    # -----------------------------
    t_max = np.percentile(TTFs, 95)
    time_points = np.linspace(0.1, t_max, 250)

    # -----------------------------
    # Reliability R_s(t)
    # -----------------------------
    R_sim = np.array([np.mean(TTFs > t) for t in time_points])

    # -----------------------------
    # PDF f_s(t) via KDE
    # -----------------------------
    kde = gaussian_kde(TTFs, bw_method=0.15)
    f_sim = kde(time_points)

    # -----------------------------
    # Hazard z_s(t) = f / R
    # -----------------------------
    eps = max(1e-6, 1.0 / len(TTFs))  # avoid division by zero
    z_sim = f_sim / np.maximum(R_sim, eps)

    # # -----------------------------
    # # Assertions (physics-based)
    # # -----------------------------
    # # 1) Hazard must be positive
    # assert np.all(z_sim > 0), "System hazard must be positive"

    # # 2) Hazard must not be constant
    # assert np.std(z_sim) > 1e-3, "System hazard appears constant"

    # # 3) Long-term trend must increase (IFR dominates)
    # slope = np.polyfit(time_points, z_sim, deg=1)[0]
    # assert slope > 0, "System hazard does not show increasing trend"

    # -----------------------------
    # Plot: system hazard
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(time_points, z_sim, "r", label="Simulated system hazard z_s(t)")
    plt.xlabel("Time")
    plt.ylabel("Hazard")
    plt.title("Series System Hazard (DFR + CFR + IFR → Bathtub)")
    plt.legend()
    plt.tight_layout()
    plt.show()



# # =============================================================================
# # EXPONENTIAL SERIES SYSTEM TEST
# # =============================================================================
# # PARAMETERS
# # =============================================================================
# @pytest.mark.parametrize(
#     "dist_type, kwargs",
#     [
#         ("exp", [10, 10, 10]),   # MTTF
#         ("exp", [100, 100, 100]),
#         ("exp", [5, 5, 5])
#     ]
# )

# def test_series_system_exponential(dist_type, kwargs, system_factory, expected_series):
#     """
#     Monte-Carlo validation of system R(t), f(t), hazard z(t), MTTF.
#     Also plots analytic vs simulated curves.
#     """

#     # -------- build systems --------
#     N = 10000
#     systems = system_factory(dist_type, kwargs, n=N)

#     # -------- grab simulated system TTFs --------
#     TTFs = np.array([sys.TTF() for sys in systems])
    
#     # -------- analytic targets --------
#     R_s_sym, f_s_sym, z_s_scalar, MTTF_analytical = expected_series(dist_type, kwargs)
#     t_sym = list(R_s_sym.free_symbols)[0]

#     time_points = np.linspace(0, np.max(TTFs), int(np.ceil(np.max(TTFs)/0.1)) + 1)

#     R_analytic = np.array([float(R_s_sym.subs(t_sym, t)) for t in time_points])
#     f_analytic = np.array([float(f_s_sym.subs(t_sym, t)) for t in time_points])
#     z_analytic = np.full_like(time_points, z_s_scalar)

#     # -------- Monte-Carlo R(t) --------
#     R_sim = np.array([np.mean(TTFs > t) for t in time_points])

#     # -------- Monte-Carlo MTTF --------
#     MTTF_sim = TTFs.mean()

#     # -------- Monte-Carlo f(t) (from KDE) --------
#     kde = gaussian_kde(TTFs, bw_method=0.1)
#     f_sim2 = kde(time_points)
    
#     # -------- z(t) = f/R (from KDE)--------    
#     n_samples = len(TTFs)          # number of observed TTFs
#     rel_eps = max(1e-6, 1.0 / n_samples)   # or 0.01 if you need more aggressive floor

#     z_sim2 = f_sim2 / np.maximum(R_sim, rel_eps)
    
#     """
#     # -------- Monte-Carlo f(t) (from histogram method) --------
#     # compute bin edges from centers (works for uniform or nonuniform spacing)
#     dt_left = time_points[1:] - time_points[:-1]
#     # internal half-steps
#     left_edges = time_points[:-1] - dt_left/2
#     right_edges = time_points[1:] + dt_left/2
#     # simpler robust approach:
#     # Create edges as midpoints between consecutive centers, and extend ends
#     midpoints = 0.5 * (time_points[:-1] + time_points[1:])
#     edges = np.empty(len(time_points) + 1)
#     edges[1:-1] = midpoints
#     edges[0]      = time_points[0] - (midpoints[0] - time_points[0])
#     edges[-1]     = time_points[-1] + (time_points[-1] - midpoints[-1])

#     hist_counts, bin_edges = np.histogram(TTFs, bins=edges, density=True)
    
#     # hist_counts now has length len(time_points) and corresponds to these centers
#     f_sim = hist_counts
#     t_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])  # should equal time_points

#     # -------- Monte-Carlo z(t) = f/R (from histogram method)--------
#     eps = 1e-10 #(prevent divide by zero error)
#     z_sim = f_sim / np.maximum(R_sim, eps)
    
#     # check if time_points and t_centers are the same
#     assert np.allclose(t_centers, time_points, rtol=1e-12, atol=1e-16), \
#         "t_centers do not match time_points!"
        
#     """

#     # OPTIONAL PLOTTING
#     plt.figure(figsize=(12,10))

#     plt.subplot(3,1,1)
#     plt.plot(time_points, R_analytic,"k", label="Analytic R(t)")
#     plt.plot(time_points, R_sim, "o", markersize=3, label="Simulated R(t)")
#     plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
#     plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
#     plt.legend()
#     plt.title("Reliability R(t)")

#     plt.subplot(3,1,2)
#     plt.plot(time_points, f_analytic, "k", label="Analytic f(t)")
#     # plt.plot(time_points, f_sim, "bo", markersize=3, label="Simulated f(t) (histogram method)")
#     plt.plot(time_points, f_sim2, "rs", markersize=3, label="Simulated f(t) (KDE)")
#     plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
#     plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
#     plt.legend()
#     plt.title("PDF f(t)")

#     plt.subplot(3,1,3)
#     plt.plot(time_points, z_analytic,"k", label="Analytic hazard z(t)")
#     # plt.plot(time_points, z_sim, "bo", markersize=3, label="Simulated hazard z(t) (histogram method)")
#     plt.plot(time_points, z_sim2, "rs", markersize=3, label="Simulated hazard z(t) (KDE)")
#     plt.axvline(x=MTTF_sim, color='black', linestyle='--', linewidth=2, label=r"MTTF$_{simulated}$ " + f"={MTTF_sim:.3f}")
#     plt.axvline(x=MTTF_analytical, color='green', linestyle='--', linewidth=2, label=r"MTTF$_{analytical}$ " + f"={MTTF_analytical:.3f}")
#     plt.legend()
#     plt.title("Hazard z(t)")
#     plt.tight_layout()
#     plt.show()

#     print("\nAnalytic MTTF:", MTTF_analytical)
#     print("Simulated MTTF:", MTTF_sim)
    
#     # check the MTTF is reasonably close to expectation 1% difference
#     assert(np.isclose(MTTF_sim, MTTF_analytical, rtol=MTTF_analytical*0.01))
