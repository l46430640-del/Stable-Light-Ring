"""Check and optionally retain the S08A06 numerical background.

All solves use at most 4000 mesh nodes. This is numerical evidence, not an
interval existence proof, a radial-stability calculation, or a counterexample.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import PPoly

from boson_star import (SINGULAR, diagnostics, equations, integral_diagnostics,
                       potential, residual_diagnostics, solve_background,
                       trapping_diagnostics)


def check_variable_change():
    r = np.array([0.3, 2.0, 13.0])
    y = np.array([[0.99, 0.9, 0.7], [-0.003, -0.04, -0.08],
                  [0.001, 0.05, 0.2], np.log([0.55, 0.5, 0.3])])
    dy = equations(r, y, 0.06, 0.08) + SINGULAR @ y / r
    A, Ap = 0.06 * y[0], 0.06 * y[1]
    m, f, Omega2 = r * y[2], 1 - 2 * y[2], np.exp(2 * y[3])
    V, Vs = potential(A, 0.08)
    rho, pr = (Omega2 * A**2 + f * Ap**2 + V) / 2, (Omega2 * A**2 + f * Ap**2 - V) / 2
    mp = 4 * np.pi * r**2 * rho
    nup = (m + 4 * np.pi * r**3 * pr) / (r**2 * f)
    fp = 2 * m / r**2 - 2 * mp / r
    App = -(2 / r + nup + fp / (2 * f)) * Ap + (Vs - Omega2) * A / f
    assert np.max(np.abs(y[2] + r * dy[2] - mp)) < 1e-14
    assert np.max(np.abs(-dy[3] - nup)) < 1e-14
    assert np.max(np.abs(0.06 * dy[1] - App)) < 1e-14


def checked_solve(central=0.06, radius=40, tolerance=1e-9, previous=None):
    solution = solve_background(central=central, radius=radius, tolerance=tolerance,
                                max_nodes=4000, previous=previous)
    if not solution.success:
        raise RuntimeError(f"Bounded solve failed: {solution.message}; nodes={solution.x.size}")
    report = diagnostics(solution, central, 0.08, radius)
    assert report["minimum_f"] > 0 and report["minimum_A"] > 0
    assert 0 < report["omega"] < 1
    assert abs(solution.sol(0)[0] - 1) < 1e-12
    assert np.max(np.abs(solution.sol(0)[1:3])) < 1e-12
    report["requested_tolerance"] = tolerance
    return solution, report


def verification():
    check_variable_change()
    reports = []
    baseline = None
    baseline_report = None
    for radius, tolerance in ((40, 1e-6), (40, 1e-7), (40, 1e-8), (40, 1e-9), (30, 1e-9), (50, 1e-9)):
        solution, report = checked_solve(radius=radius, tolerance=tolerance)
        report["integrals"] = integral_diagnostics(solution, 0.06, 0.08, report["omega"])
        report["trapping"] = trapping_diagnostics(solution, report)
        assert len(report["light_rings"]) == 2
        reports.append(report)
        print(f"R={radius:2g}, tol={tolerance:.0e}, nodes={report['nodes']}, "
              f"M={report['mass']:.11f}, N={report['charge']:.11f}, omega={report['omega']:.11f}")
        if radius == 40 and tolerance == 1e-9:
            baseline, baseline_report = solution, report
    tight = [report for report in reports if report["requested_tolerance"] == 1e-9]
    spreads = {key: float(np.ptp([report[key] for report in tight]))
               for key in ("mass", "charge", "omega", "r99")}
    assert spreads["mass"] < 2e-8 and spreads["charge"] < 1e-7
    assert spreads["omega"] < 1e-9 and spreads["r99"] < 1e-7
    for report in tight:
        assert abs(report["integrals"]["mass_integral"] - report["mass"]) < 2e-8
        assert abs(report["integrals"]["komar_mass"] - report["mass"]) < 2e-8
        assert report["trapping"]["light_ring_derivative_crosscheck"] < 2e-7
    assert abs(baseline_report["mass"] - 5.65) < 0.005
    assert abs(baseline_report["charge"] - 26.1) < 0.05
    assert abs(baseline_report["r99"] - 17.9) < 0.05

    residuals = residual_diagnostics(baseline, 0.06, 0.08)
    for key, threshold in (("A_prime_error", 1e-10), ("mass_equation_error", 2e-8),
                           ("lapse_equation_error", 2e-9), ("KG_equation_error", 1e-10),
                           ("angular_Einstein_error", 3e-9), ("center_A_second_error", 1e-10),
                           ("center_u_second_error", 2e-10), ("center_log_frequency_second_error", 2e-10)):
        assert residuals[key] < threshold, (key, residuals[key])

    first_law = []
    for step in (2e-5, 1e-5):
        _, minus = checked_solve(central=0.06 - step, previous=baseline)
        _, plus = checked_solve(central=0.06 + step, previous=baseline)
        mass_derivative = (plus["mass"] - minus["mass"]) / (2 * step)
        charge_derivative = (plus["charge"] - minus["charge"]) / (2 * step)
        ratio = mass_derivative / charge_derivative
        error = abs(ratio - baseline_report["omega"])
        first_law.append({"central_amplitude_step": step, "dM_dA0": mass_derivative,
                          "dN_dA0": charge_derivative, "dM_dN": ratio,
                          "absolute_error_against_omega": error})
        print(f"first law: dA0={step:.0e}, dM/dN={ratio:.11f}, error={error:.3g}")
    assert first_law[-1]["absolute_error_against_omega"] < 1e-5
    assert first_law[-1]["absolute_error_against_omega"] < first_law[0]["absolute_error_against_omega"] / 3
    result = {"status": "passed_numerical_checks", "units": "mu=G=c=hbar=1",
              "scope": "Float64 consistency checks; no rigorous error bounds, radial spectrum, or physical feedback result.",
              "software": {"numpy": np.__version__, "scipy": scipy.__version__},
              "reference": "https://arxiv.org/abs/2504.17775v3",
              "primary_background": baseline_report, "convergence_runs": reports,
              "outer_domain_spreads": spreads, "independent_residuals": residuals,
              "first_law_checks": first_law}
    print("trapping:", json.dumps(baseline_report["trapping"]))
    print("PASS: regularity, finite-domain convergence, mass/charge/Komar integrals, light rings, first law")
    return baseline, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    solution, result = verification()
    if args.output_directory is not None:
        directory = args.output_directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        profile = directory / "background.npz"
        np.savez_compressed(profile, r=solution.x, y=solution.y, coefficients=solution.sol.c)
        with np.load(profile, allow_pickle=False) as data:
            restored = PPoly.construct_fast(data["coefficients"], data["r"],
                                           extrapolate=False, axis=1)
            assert np.array_equal(restored(solution.x), solution.sol(solution.x))
        result["profile"] = {"filename": profile.name,
                             "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                             "state_order": ["A/A0", "(A/A0)'", "m/r", "log(omega/alpha)"],
                             "valid_radius_interval": [float(solution.x[0]), float(solution.x[-1])],
                             "spline": "scipy.interpolate.PPoly.construct_fast(coefficients, r, extrapolate=False, axis=1)"}
        (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved numerical background to {directory}")


if __name__ == "__main__":
    main()
