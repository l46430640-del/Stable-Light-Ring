"""Check finite-width orbit sources without resolving high-frequency nodes."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import sympy as sp

from check_trapped_response import finite_action_difference
from finite_width_axial import FiniteWidthResponse, OrbitSource
from trapped_axial import Background


def check_algebra():
    r, E, C = sp.symbols("r E C", positive=True)
    alpha = sp.Function("alpha")(r)
    U = alpha**2 / r**2
    rho = C / (r**2 * alpha**2 * sp.sqrt(E - U))
    pr, pt = (1 - U / E) * rho, U * rho / (2 * E)
    conservation = sp.diff(pr, r) + (rho + pr) * sp.diff(alpha, r) / alpha + 2 * (pr - pt) / r
    assert sp.simplify(conservation) == 0
    assert sp.simplify(-rho + pr + 2 * pt) == 0
    h, psi, psip, potential, potentialp, remainder, remainderp = sp.symbols("h psi psip V Vp W Wp")
    psipp = (potential + h**2 * remainder - E) * psi / h**2
    stress_derivative = (2 * h**2 * psip * psipp + 2 * (E - potential - h**2 * remainder) * psi * psip
                         - (potentialp + h**2 * remainderp) * psi**2)
    assert sp.expand(stress_derivative + (potentialp + h**2 * remainderp) * psi**2) == 0


def source_checks(source, conservation_tolerance=1e-9):
    bg = source.background
    t = np.linspace(0, 1, 151)
    r, dr, forcing = source.forcing(t)
    assert np.all(np.isfinite(forcing))
    assert np.all(dr >= 0)
    # The mapped endpoint values must agree with their nearby limits.
    _, _, edge = source.forcing(np.array([0., 1e-5, 1 - 1e-5, 1.]))
    edge_error = float(max(np.max(np.abs(edge[:, 0] - edge[:, 1])),
                           np.max(np.abs(edge[:, 2] - edge[:, 3]))))
    assert edge_error < 1e-7
    g = bg.fields(r[1:-1])
    _, _, gap, _ = source.measure(t[1:-1])
    rho = 1 / (4 * np.pi * source.normalization * r[1:-1]**2 * g["alpha"]**2 * np.sqrt(gap))
    pressure = gap / source.energy
    direct = np.array([np.zeros_like(rho), 4 * np.pi * r[1:-1] * (1 - pressure) * rho * bg.sol(r[1:-1])[1] / g["f"],
                       4 * np.pi * r[1:-1] * rho, -4 * np.pi * r[1:-1] * pressure * rho / g["f"]])
    forcing_error = float(np.max(np.abs(forcing[:, 1:-1] - direct * dr[1:-1])))
    assert forcing_error < 1e-12
    root_error = float(max(abs(bg.fields(root)["U"] - source.energy) for root in (source.a, source.b, source.c)))
    assert root_error < 1e-17
    conservation = (1 + pressure) * (g["nup"] + bg.sol(r[1:-1], 1)[3])
    conservation_error = float(np.max(np.abs(conservation)))
    assert conservation_error < conservation_tolerance
    return {"mapped_endpoint_error": edge_error, "forcing_reconstruction_error": forcing_error,
            "turning_point_residual": root_error, "conservation_residual_over_density": conservation_error}


def check_report(report):
    assert report["radial_points"] <= 4000
    assert report["max_rms_residual"] < report["tolerance"]
    assert report["max_angle_ODE_residual"] < 7e-5
    assert report["boundary_matching_residual"] < 1e-12
    assert abs(report["added_mass"] - 1) < 1e-9
    assert abs(report["charge_integral"]) < 6e-8
    assert report["normalization_quadrature_change"] < 5e-8
    assert report["pressure_identity_error"] < 1e-11
    assert report["action_quadrature_change"] < 5e-8
    assert 0 < report["radial_pressure_fraction"] < .000181
    assert report["delta_barrier_action"] > 0


def verification():
    if not __debug__:
        raise RuntimeError("Run without -O: numerical checks require assertions.")
    check_algebra()
    bg = Background()
    for bad in (0, 1, -.1, 1.1):
        try:
            OrbitSource(bg, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid source energy was accepted.")
    reports, consistency = [], []
    for fraction in (.01, .1, .3, .5, .7, .9, .99):
        source = OrbitSource(bg, fraction)
        consistency.append({"energy_fraction": fraction, **source_checks(source)})
        response = FiniteWidthResponse(source)
        report = response.diagnostics()
        check_report(report)
        reports.append(report)
        print(f"eta={fraction:g}: dS={report['delta_barrier_action']:.10f}, "
              f"pressure={report['radial_pressure_fraction']:.6g}, points={report['radial_points']}")
    refinements, differences = [], []
    for index in (3, 6):
        source = OrbitSource(bg, reports[index]["energy_fraction"])
        response = FiniteWidthResponse(source, tolerance=2e-7)
        report = response.diagnostics()
        check_report(report)
        report["action_tolerance_change"] = abs(report["delta_barrier_action"] - reports[index]["delta_barrier_action"])
        assert report["action_tolerance_change"] < 1e-7
        refinements.append(report)
        print(f"refined eta={source.fraction:g}: points={report['radial_points']}, "
              f"dS change={report['action_tolerance_change']:.3g}")
        if index == 3:
            expected = response.action_response()
            for step in (2e-5, 1e-5):
                finite = finite_action_difference(bg, response, step, target_action=source.action)
                error = float(abs(finite[1] - expected[0]))
                assert error < 4e-6
                assert abs(finite[0] - 2 * source.energy * expected[3]) < 1e-9
                differences.append({"step": step, "action_response": float(finite[1]), "action_error": error})
                print(f"moving turning points: step={step:g}, dS error={error:.3g}")
            assert differences[-1]["action_error"] < differences[0]["action_error"] / 2
            # Exercise the original ground-action argument without a new wave solve.
            adapter = SimpleNamespace(metric=response.metric,
                                      mode=SimpleNamespace(ell=np.pi / (2 * source.action) - .5))
            original = finite_action_difference(bg, adapter, step)
            assert np.max(np.abs(original - finite)) < 1e-10
    background = Path(__file__).parent / "data/s08a06/background.npz"
    return {"status": "passed_numerical_checks",
            "scope": "Finite-width semiclassical axial monopole source on the represented background, approximate regular/decaying endpoints and fixed charge. Sampled floating-point signs, not an interval theorem for all energies or the exact star; no nonlinear stability conclusion.",
            "background_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
            "max_total_radial_points_per_solve": 4000,
            "relative_well_depth": float((bg.Uu - bg.Us) / bg.Us),
            "software": {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__},
            "source_consistency": consistency, "energy_scan": reports, "tolerance_refinements": refinements,
            "moving_turning_point_checks": differences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verification()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: source conservation, turning-point coordinates, finite-width response and fixed-action derivative")
