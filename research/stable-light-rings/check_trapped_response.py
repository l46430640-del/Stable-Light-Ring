"""Bounded checks of axial quasimodes and their stationary monopole response.

The signs below are floating-point evidence for the projected response only.
All eigenvalue and response solves have an explicit 4000-node ceiling.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar

from boson_star import boundary, equations
from mean_response import MeanResponse, background_jacobian, tail_variation
from trapped_axial import Background, TrappedAxial


def check_linearization(bg):
    r = np.array([.2, 2., 14., 16., 18.])
    y = bg.sol(r)
    direction = np.array([[.3], [-.1], [.05], [.2]]) * np.ones_like(y)
    analytic = np.einsum("ijn,jn->in", background_jacobian(bg, r), direction)
    errors = []
    for step in (2e-5, 1e-5):
        finite = (equations(r, y + step * direction, bg.central, bg.scale)
                  - equations(r, y - step * direction, bg.central, bg.scale)) / (2 * step)
        errors.append(float(np.max(np.abs(finite - analytic))))
    assert errors[-1] < 1e-8 and errors[-1] < errors[0] / 3
    right = bg.sol(bg.radius)
    perturbation = np.array([.2, -.1, .04, .3])
    step = 1e-5
    finite = (boundary(y[:, 0], right + step * perturbation, bg.radius)[-1]
              - boundary(y[:, 0], right - step * perturbation, bg.radius)[-1]) / (2 * step)
    error = float(abs(finite - tail_variation(bg, perturbation)))
    assert error < 1e-9
    # Independent reduction of the two first-order axial equations.
    g = bg.fields(r)
    ell = 7
    reduced = g["alpha"]**2 * ((ell * (ell + 1) - 2 + 2 * g["f"]) / r**2
                               - g["f"] * g["nup"] / r - g["fp"] / (2 * r))
    assert np.max(np.abs(reduced - bg.axial_potential(r, ell))) < 1e-12
    return {"ODE_directional_errors": errors, "tail_directional_error": error}


def finite_action_difference(bg, response, step, target_action=None,
                             light_ring_bounds=None, turning_bounds=None):
    light_ring_bounds = light_ring_bounds or ((15.7, 16.2), (16.4, 16.7))
    turning_bounds = turning_bounds or (15., 17.2)
    nodes, weights = np.polynomial.legendre.leggauss(192)
    theta = (nodes + 1) * np.pi / 4
    def integrate(fun, lo, hi):
        r = lo + (hi - lo) * np.sin(theta)**2
        return np.dot(weights * np.pi / 4 * (hi - lo) * np.sin(2 * theta), fun(r))

    def evaluate(parameter):
        def geometry(r):
            g = bg.fields(r)
            x, v = response.metric(r)
            U = g["U"] * np.exp(2 * parameter * v)
            f = g["f"] - 2 * parameter * x / r
            return U, 1 / (r * np.sqrt(U * f))

        rs = minimize_scalar(lambda r: geometry(r)[0], bounds=light_ring_bounds[0], method="bounded").x
        ru = minimize_scalar(lambda r: -geometry(r)[0], bounds=light_ring_bounds[1], method="bounded").x
        Us, Uu = geometry(rs)[0], geometry(ru)[0]
        def actions(E):
            fun = lambda r: geometry(r)[0] - E
            a = brentq(fun, turning_bounds[0], rs, xtol=1e-13)
            b = brentq(fun, rs, ru, xtol=1e-13)
            c = brentq(fun, ru, turning_bounds[1], xtol=1e-13)
            J = integrate(lambda r: geometry(r)[1] * np.sqrt(np.maximum(E - geometry(r)[0], 0)), a, b)
            S = integrate(lambda r: geometry(r)[1] * np.sqrt(np.maximum(geometry(r)[0] - E, 0)), b, c)
            return J, S
        target = target_action if target_action is not None else np.pi / (2 * (response.mode.ell + .5))
        fraction = brentq(lambda t: actions(Us + t * (Uu - Us))[0] - target, .001, .999, xtol=1e-12)
        E = Us + fraction * (Uu - Us)
        return np.array([E, actions(E)[1]])
    return (evaluate(step) - evaluate(-step)) / (2 * step)


def checked_pair(bg, ell, tolerance, interval=(15, 16.75)):
    mode = TrappedAxial(bg, ell, tolerance=tolerance, interval=interval)
    response = MeanResponse(bg, mode, tolerance=tolerance)
    md, rd = mode.diagnostics(), response.diagnostics()
    assert md["mode_nodes_in_well"] == 0
    assert 0 < md["fraction_of_well_depth"] < 1
    assert md["cutoff_operator_defect"] < 1e-6
    assert rd["mass_energy_error"] < 2e-7
    assert abs(rd["charge_integral"]) < 2e-8
    assert rd["max_mass_equation_residual"] < 6e-6
    assert rd["max_lapse_equation_residual"] < 3e-8
    assert rd["max_KG_equation_residual"] < 1e-7
    assert rd["max_angular_Einstein_residual"] < 2e-6
    # These checks record the outcome for this background, not a universal sign.
    assert rd["minimum_enclosed_mass_response_on_trapping_interval"] > 2
    assert rd["minimum_raw_lapse_derivative_on_trapping_interval_sampled"] > .03
    assert .19 < rd["action"]["delta_barrier_action"] < .22
    report = {"background_radius": bg.radius, "tolerance": tolerance,
              "mode_interval": list(interval), "mode": md, "response": rd}
    print(f"ell={ell}, tol={tolerance:.0e}, R={bg.radius:g}, "
          f"dS={rd['action']['delta_barrier_action']:.10f}, "
          f"mass_error={rd['mass_energy_error']:.2g}, "
          f"nodes={md['nodes']}/{rd['nodes']}")
    return mode, response, report


def verification():
    bg = Background()
    linearization = check_linearization(bg)
    reports = []
    fine_pairs = {}
    for ell, tolerances in ((50000, (1e-7, 1e-8)), (100000, (1e-7,))):
        for tolerance in tolerances:
            mode, response, report = checked_pair(bg, ell, tolerance)
            reports.append(report)
        fine_pairs[ell] = (mode, response)
        if len(tolerances) > 1:
            coarse, fine = reports[-2], reports[-1]
            assert abs(coarse["response"]["action"]["delta_barrier_action"]
                       - fine["response"]["action"]["delta_barrier_action"]) < 2e-7
            assert fine["mode"]["full_cut_polynomial_operator_defect"] < coarse["mode"]["full_cut_polynomial_operator_defect"] / 2

    differences = []
    _, response = fine_pairs[50000]
    action = response.diagnostics()["action"]
    expected = np.array([2 * action["energy"] * action["vstar"], action["delta_barrier_action"]])
    for step in (2e-5, 1e-5):
        finite = finite_action_difference(bg, response, step)
        error = float(abs(finite[1] - expected[1]))
        differences.append({"step": step, "frequency_response": float(finite[0]),
                            "action_response": float(finite[1]), "action_error": error})
        assert error < 3e-6
        assert abs(finite[0] - expected[0]) < 1e-9
        print(f"moving turning points: step={step:.0e}, dS={finite[1]:.10f}, error={error:.2g}")
    assert differences[-1]["action_error"] < differences[0]["action_error"] / 2
    path = Path(__file__).parent / "data/s08a06/background.npz"
    result = {"status": "passed_numerical_checks", "units": "mu=G=c=hbar=1",
              "scope": "Finite-interval axial modes with cutoff; stationary quadratic monopole, unit added energy. No rigorous error bounds, outgoing resonance, complete second-order metric, or nonlinear counterexample.",
              "background_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "cutoff": {"rising_interval": [15.1, 15.3], "falling_interval": [16.5, 16.7],
                         "polynomial": "10*t**3-15*t**4+6*t**5"},
              "normalization": "c_ell=1; 4*pi*integral(r**2*alpha/sqrt(f)*rho_ax dr)=1",
              "resource_limit": {"max_nodes_per_solve": 4000,
                                 "not_completed": "ell=100000 at tolerance 1e-8 reached the mesh ceiling; no finer solve or domain sweep was performed. User requested retaining the limit."},
              "software": {"numpy": np.__version__, "scipy": scipy.__version__},
              "linearization": linearization, "convergence_runs": reports,
              "moving_turning_point_checks": differences}
    print("PASS: axial operator, ell=50000 tolerance convergence, fixed charge, energy, original mean equations, moving-action check")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verification()
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
