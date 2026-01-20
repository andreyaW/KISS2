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
    Returns analytic R_s(t), f_s(t), z_s(t)
    for a parallel system.

    MTTF intentionally omitted to avoid symbolic integration blow-ups.
    """
    def __factory__(dist_type, params):
        t = sp.Symbol("t", real=True, positive=True)

        # EXPONENTIAL COMPS PARALLEL SOLUTION
        if dist_type == "exp":
            lambdas = [1 / MTTF for MTTF in params]

            product_term = sp.Integer(1)
            for lam in lambdas:
                product_term *= (1 - sp.exp(-lam * t))

            R_s_t = sp.simplify(1 - product_term)
            f_s_t = sp.diff(1 - R_s_t, t)
            z_s_t = sp.simplify(f_s_t / R_s_t)

            return R_s_t, f_s_t, z_s_t

        # WEIBULL COMPS PARALLEL SOLUTION
        elif dist_type == "weib":
            mttfs = [p[0] for p in params]
            shapes = [p[1] for p in params]

            if len(set(shapes)) != 1:
                raise NotImplementedError(
                    "Parallel Weibull analytic form requires identical shape"
                )

            k = shapes[0]

            thetas = [
                mttf / sp.gamma(1 + 1 / k)
                for mttf in mttfs
            ]

            product_term = sp.Integer(1)
            for theta in thetas:
                product_term *= (1 - sp.exp(-(t / theta) ** k))

            R_s_t = sp.simplify(1 - product_term)
            f_s_t = sp.diff(1 - R_s_t, t)
            z_s_t = sp.simplify(f_s_t / R_s_t)

            return R_s_t, f_s_t, z_s_t

        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

    return __factory__


# =============================================================================
# TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TEST 1 : Check that Parallel System Reliability Matches Analytic Expression
# ---------------------------------------------------------------------------- 
# PARAMETERS:
@pytest.mark.parametrize(
    "dist_type, params, time_scale",
    [
        # -------------------------
        # Exponential systems
        # -------------------------
        ("exp", [10.0, 20.0], 10),
        ("exp", [5.0, 10.0, 20.0], 8),
        ("exp", [8.0, 15.0, 30.0, 60.0], 6),

        # -------------------------
        # Weibull systems (common shape)
        # -------------------------
        ("weib", [(10.0, 1.5), (20.0, 1.5)], 10),
        ("weib", [(5.0, 2.0), (10.0, 2.0), (20.0, 2.0)], 8),
        ("weib", [(8.0, 1.2), (15.0, 1.2), (30.0, 1.2), (60.0, 1.2)], 6),
    ]
)

# TEST FUNCTION:
def test_parallel_system_general(
    dist_type,
    params,
    time_scale,
    component_factory,
    expected_parallel
):
    """
    Verify analytic reliability of an N-component parallel system
    for exponential or Weibull components.
    """

    t = sp.Symbol("t", real=True, positive=True)

    # Build components
    components = component_factory(dist_type, params)
    system = ParallelSystem("ParallelSys", components)

    # Analytic target
    R_expected, _, _ = expected_parallel(dist_type, params)

    # System result
    R_system = sp.simplify(system.R_s())

    # -------------------------------------------------
    # Symbolic equivalence
    assert sp.simplify(R_system - R_expected) < 1e-1

    # -------------------------------------------------
    # Numeric sanity check
    # -------------------------------------------------
    f_expected = sp.lambdify(t, R_expected, "numpy")
    f_system = sp.lambdify(t, R_system, "numpy")

    # Choose reasonable time grid
    if dist_type == "exp":
        t_max = time_scale * min(params)
    else:
        t_max = time_scale * min(mttf for mttf, _ in params)

    time_grid = np.linspace(0, t_max, 50)

    assert np.allclose(
        f_expected(time_grid),
        f_system(time_grid),
        rtol=1e-12,
        atol=1e-14
    )