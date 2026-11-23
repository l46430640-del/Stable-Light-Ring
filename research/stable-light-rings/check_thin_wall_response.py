"""Finite-width radiation feedback on the S06A044 numerical background.

The sign is an output, never a test assumption. Solves retain the 4000-point
ceiling. A stopped computation is checkpointed and is not retried here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from check_finite_width_response import check_algebra, source_checks
from check_joint_screen import equilibrium
from check_trapped_response import finite_action_difference
from concentrated_axial import ConcentratedResponse
from finite_width_axial import FiniteWidthResponse, OrbitSource
from separatrix_response import interface_diagnostics, upper_well_data, upper_well_quadrature
from trapped_axial import Background


DATA = Path(__file__).parent/"data/solitonic-screen"
BACKGROUND = DATA/"s06a044-background.npz"
FRACTIONS = (.001, .01, .1, .3, .5, .7, .9, .99, .999, .9999)


def load_background():
    with np.load(BACKGROUND) as data:
        return Background(path=BACKGROUND, central=float(data["central"]),
                          scale=float(data["scale"]), ring_brackets=data["ring_brackets"])


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


def checked_response(bg, fraction, tolerance=3e-6):
    source = OrbitSource(bg, fraction)
    source_report = source_checks(source, conservation_tolerance=2e-8)
    response = FiniteWidthResponse(source, tolerance=tolerance)
    result = response.diagnostics()
    result["source_checks"] = source_report
    assert result["radial_points"] <= 4000
    assert result["max_rms_residual"] < tolerance
    assert result["max_relative_angle_ODE_residual"] < 5*tolerance
    assert result["boundary_matching_residual"] < 1e-12
    assert abs(result["added_mass"]-1) < 1e-9
    assert abs(result["charge_integral"]) < 3e-6
    assert result["normalization_quadrature_change"] < 2e-7
    assert result["pressure_identity_error"] < 1e-9
    assert result["action_quadrature_change"] < 2e-7
    assert 0 < result["radial_pressure_fraction"] < 1-bg.Us/bg.Uu
    r, weight, gap = source.raw_quadrature(240)
    x, v = response.metric(r)
    vu = response.metric(bg.ru)[1]
    H_E = float(np.dot(weight, vu-v+x/(r*bg.fields(r)["f"])*gap/source.energy))
    _, barrier_weights, barrier_gap = source.raw_quadrature(240, barrier=True)
    B_E = float(np.sum(barrier_weights))
    barrier = float(np.dot(barrier_weights, barrier_gap))
    result.update({"relative_energy_deficit": 1-fraction, "H_E": H_E, "barrier_weight": B_E,
                   "response_at_stable_ring": response.state(bg.rs)[:, 0].tolist(),
                   "mean_distance_inside_unstable_ring": float(weight@(bg.ru-r)/np.sum(weight)),
                   "unperturbed_barrier_action": barrier,
                   "Z_times_action": source.normalization*result["delta_barrier_action"],
                   "log_times_action": float(np.log(1/(1-fraction))*result["delta_barrier_action"]),
                   "renormalized_barrier_remainder": float(source.normalization*result["delta_barrier_action"]
                                                           -B_E*source.energy*H_E)})
    return response, result


def upper_endpoint(bg, tolerance):
    response = ConcentratedResponse(bg, tolerance=tolerance, light_ring="unstable")
    result = upper_well_data(bg, response)
    result.update(interface_diagnostics(response))
    result["tolerance"] = tolerance
    result["response_at_stable_ring"] = response.state(bg.rs)[:4].tolist()
    result["H_quadrature_change"] = abs(result["H"]-upper_well_data(bg, response, order=240)["H"])
    assert result["radial_points"] <= 4000
    assert result["max_rms_residual"] < tolerance
    assert result["max_angle_ODE_residual"] < 2e-4
    assert result["jump_residual"] < 1e-12
    assert result["lapse_slope_jump_error"] < 1e-10
    assert abs(result["added_mass"]-1) < 2e-8
    assert abs(result["charge_integral"]) < 3e-6
    assert result["H_quadrature_change"] < 1e-5
    return result


def finite_part_checks(bg, scan):
    limits = []
    for order in (160, 240):
        r, weights, gap, _, _ = upper_well_quadrature(bg, order)
        limits.append(np.array([weights@(r-bg.ru), weights@((r-bg.ru)**2), weights@gap/bg.Uu]))
    assert np.max(np.abs(limits[1]-limits[0])/(1+abs(limits[1]))) < 1e-8
    results = []
    for item in scan[-4:]:
        source = OrbitSource(bg, item["energy_fraction"])
        r, weights, gap = source.raw_quadrature(240)
        actual = np.array([weights@(r-bg.ru), weights@((r-bg.ru)**2), weights@gap/source.energy])
        errors = np.abs(actual-limits[1])/(1+abs(limits[1]))
        results.append({"energy_fraction": source.fraction, "unnormalized_moments": actual.tolist(),
                        "relative_errors": errors.tolist()})
    assert max(results[-1]["relative_errors"]) < .002
    return {"test_functions": ["r-ru", "(r-ru)^2", "radial pressure"],
            "limits": limits[1].tolist(), "quadrature_change": (limits[1]-limits[0]).tolist(),
            "samples": results}


def cubic_moment_checks():
    # U=1-x^2(1+x), q=1, au=-1, ru=0 has elementary limiting moments.
    nodes, weights = np.polynomial.legendre.leggauss(240)
    theta = (nodes+1)*np.pi/4
    limits = np.array([-2., 4/3, 4/15])
    results = []
    for deficit in (1e-3, 1e-5, 1e-7):
        roots = np.sort(np.roots([1., 1., 0., -deficit]))
        assert np.isrealobj(roots)
        a, b, c = roots
        r = a+(b-a)*np.sin(theta)**2
        measure = weights*np.pi/(2*np.sqrt(c-r))
        gap = (r-a)*(r-b)*(r-c)
        actual = np.array([measure@r, measure@(r*r), measure@gap/(1-deficit)])
        results.append({"deficit": deficit, "moments": actual.tolist(),
                        "error": float(np.max(abs(actual-limits)))})
    assert results[-1]["error"] < 1e-3
    assert all(b["error"] < .1*a["error"] for a, b in zip(results, results[1:]))
    return {"exact_limits": limits.tolist(), "samples": results}


def refinements(bg, record, output):
    checks = record["refinements"] = []
    for fraction in (.5, .9999):
        record["current_check"] = f"response tolerance at {fraction}"
        save(record, output)
        response, item = checked_response(bg, fraction, tolerance=1e-6)
        original = next(x for x in record["energy_scan"] if x["energy_fraction"] == fraction)
        item["action_change"] = abs(item["delta_barrier_action"]-original["delta_barrier_action"])
        assert item["action_change"] < 2e-6
        checks.append(item)
        save(record, output)
        print(f"refined eta={fraction:g}: change={item['action_change']:.4g}, points={item['radial_points']}", flush=True)
        if fraction == .5:
            differences = record["moving_turning_points"] = []
            width = bg.ru-bg.rs
            bounds = ((bg.rs-width/3, bg.rs+width/3), (bg.ru-width/3, bg.ru+width/3))
            for step in (2e-4, 1e-4):
                finite = finite_action_difference(bg, response, step, target_action=response.source.action,
                                                  light_ring_bounds=bounds, turning_bounds=(25., 35.))
                exact = response.action_response()
                error = abs(float(finite[1])-exact[0])
                frequency_error = abs(float(finite[0])-2*response.source.energy*exact[3])
                differences.append({"step": step, "action_error": error, "frequency_error": frequency_error})
                assert error < 5e-5 and frequency_error < 1e-8
            assert differences[-1]["action_error"] < .7*differences[0]["action_error"]
            save(record, output)
    record["current_check"] = "unstable-ring response tolerance"
    save(record, output)
    record["upper_endpoint_refined"] = upper_endpoint(bg, 1e-6)
    save(record, output)
    assert abs(record["upper_endpoint_refined"]["H"]-record["upper_endpoint"]["H"]) < 1e-3
    record["current_check"] = "second unstable-ring response refinement"
    save(record, output)
    record["upper_endpoint_fine"] = upper_endpoint(bg, 3e-7)
    save(record, output)
    assert abs(record["upper_endpoint_fine"]["H"]-record["upper_endpoint_refined"]["H"]) < 3e-5
    save(record, output)
    backgrounds = record["background_checks"] = []
    for radius in (bg.radius, 60.):
        record["current_check"] = f"background tolerance/outer radius {radius}"
        save(record, output)
        solution, _, new_bg, count = equilibrium(bg, .06, .044, radius, tolerance=1e-8)
        assert count == 2
        item = {"radius": radius, "background_nodes": int(solution.x.size),
                "background_tolerance": 1e-8, "samples": []}
        backgrounds.append(item)
        for fraction in (.5, .9999):
            _, result = checked_response(new_bg, fraction, tolerance=1e-6)
            old = next(x for x in checks if x["energy_fraction"] == fraction)
            result["action_change_from_saved_background"] = abs(result["delta_barrier_action"]-old["delta_barrier_action"])
            assert result["action_change_from_saved_background"] < 2e-6
            item["samples"].append(result)
            save(record, output)
        item["upper_endpoint"] = upper_endpoint(new_bg, 1e-6)
        assert abs(item["upper_endpoint"]["H"]-record["upper_endpoint_refined"]["H"]) < 2e-4
        save(record, output)


def verification(output):
    if not __debug__:
        raise RuntimeError("Run without -O: numerical checks require assertions.")
    start = perf_counter()
    check_algebra()
    bg = load_background()
    record = {"status": "running", "scope": "Floating-point fixed-charge monopole response on a finite-domain numerical star. No uniform energy sign certificate or radial stability theorem.",
              "background_sha256": hashlib.sha256(BACKGROUND.read_bytes()).hexdigest(),
              "software": {"numpy": np.__version__, "scipy": scipy.__version__},
              "central_amplitude": bg.central, "sigma": bg.scale,
              "relative_well_depth": float(bg.Uu/bg.Us-1), "max_radial_points": 4000,
              "cubic_moment_checks": cubic_moment_checks(),
              "energy_scan": []}
    try:
        for fraction in FRACTIONS:
            record["current_check"] = f"energy fraction {fraction}"
            save(record, output)
            _, result = checked_response(bg, fraction)
            record["energy_scan"].append(result)
            save(record, output)
            print(f"eta={fraction:g}: action={result['delta_barrier_action']:.10g}, "
                  f"mass={result['mass_action_term']:.6g}, lapse={result['lapse_action_term']:.6g}, "
                  f"points={result['radial_points']}", flush=True)
        record["current_check"] = "independent unstable-ring coefficient"
        save(record, output)
        record["upper_endpoint"] = upper_endpoint(bg, 3e-6)
        print(f"upper coefficient: {record['upper_endpoint']['log_times_action_limit']:.10g}", flush=True)
        record["finite_part_moments"] = finite_part_checks(bg, record["energy_scan"])
        refinements(bg, record, output)
        for item in record["energy_scan"]:
            item["scaled_H_correction"] = item["normalization_Z"]*(item["H_E"]-record["upper_endpoint_fine"]["H"])
            item["scaled_response_correction_at_stable_ring"] = (item["normalization_Z"]
                *(np.array(item["response_at_stable_ring"])
                  -np.array(record["upper_endpoint_fine"]["response_at_stable_ring"]))).tolist()
        record["status"] = "completed_finite_width_checks"
        record["elapsed_seconds"] = perf_counter()-start
        save(record, output)
    except Exception as error:
        record["status"] = "stopped"
        record["error"] = repr(error)
        record["elapsed_seconds"] = perf_counter()-start
        save(record, output)
        raise
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA/"finite-width-response.json")
    args = parser.parse_args()
    report = verification(args.output)
    print(json.dumps({"status": report["status"], "samples": len(report["energy_scan"]),
                      "elapsed_seconds": report["elapsed_seconds"]}))
