"""Bounded intermediate-degree responses for the original near-top source.

The previously stopped fine quadrupole is never called. Each converged
degree is saved before proceeding. Compact quadrature and radial BVP
budgets are tracked separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Do not disable verification assertions with python -O.")

import numpy as np

from orientation_response import complete_orientation_minimum
from finite_width_axial import OrbitSource
from mean_response import MeanResponse
from polar_optical_tail import leading_action, orbit_quadrature, resummed_kernel, sampled_minimum
from polar_static import OrbitBandSource, PolarResponse, angular_parameters
from trapped_axial import Background, gauss_mesh

ROOT = Path(__file__).parent
FRACTIONS = (.1, .5, .9, .99, .999)
TOLERANCE = 2e-6
MAX_NODES = 4000


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


def solve(source, degree, tolerance, report, path, **kwargs):
    if degree < 2 or tolerance < TOLERANCE:
        raise ValueError("This calculation uses only coarse derivative-shifted responses.")
    try:
        return PolarResponse(source, degree=degree, tolerance=tolerance,
                             max_nodes=MAX_NODES, source_derivative_shift=True, **kwargs)
    except (RuntimeError, ValueError) as error:
        if isinstance(error, ValueError) and "quadrature points" not in str(error):
            raise
        report["stopped_attempt"] = {"degree": degree, "tolerance": tolerance,
            "maximum_nodes": MAX_NODES, "message": str(error), "boundary_changes": kwargs}
        save(report, path)
        raise SystemExit("Stopped response calculation; review the recorded resource limit.") from error


def split_action(response, fraction, order):
    probe = OrbitSource(response.background, fraction, order=order)
    r, weights, gap = orbit_quadrature(probe, response.source, order)
    v, k = response.optical(r)
    vstar = float(weights@(v-k*gap/probe.energy)/sum(weights))
    r, weights, gap = orbit_quadrature(probe, response.source, order, barrier=True)
    v, k = response.optical(r)
    distance = float(weights@(k*gap))
    frequency = float(weights@(probe.energy*(v-vstar)))
    return {"fraction": fraction, "degree": response.degree, "D_degree": distance+frequency,
            "distance_term": distance, "frequency_term": frequency, "vstar": vstar}


def split_monopole(response, source, fraction, order):
    probe = OrbitSource(response.background, fraction, order=order)
    def optical(r):
        x, v = response.metric(r)
        return v, x/(r*response.background.fields(r)["f"])
    r, weights, gap = orbit_quadrature(probe, source, order)
    v, k = optical(r)
    vstar = float(weights@(v-k*gap/probe.energy)/sum(weights))
    r, weights, gap = orbit_quadrature(probe, source, order, barrier=True)
    v, k = optical(r)
    return float(weights@(k*gap+probe.energy*(v-vstar)))


def angle_normalized_source(bg, band, order=320):
    source = OrbitBandSource(bg, band)
    angle = 2/np.pi*np.arcsin(np.sqrt(np.clip((source.knots-source.a)/(source.b-source.a), 0, 1)))
    t, q = gauss_mesh(angle, order)
    if len(t) > MAX_NODES:
        raise ValueError("The source normalization exceeds 4000 radial points.")
    r, dr, _, _ = source.orbit.mapped(t)
    g = bg.fields(r)
    source.factor /= q@(dr*4*np.pi*r*r*g["alpha"]/np.sqrt(g["f"])*source.fields(r)[0])
    return source


def validate_mode(row):
    residuals = row["residuals"]
    assert residuals["radial_points"] <= MAX_NODES
    assert max(residuals["independent_equation_residuals"].values()) < 5e-5
    assert max(row["quadrature_changes"]) < 2e-9
    assert [a["fraction"] for a in row["actions"]] == list(FRACTIONS)
    assert np.all(np.isfinite([a["D_degree"] for a in row["actions"]]))


def analyze(report, source, maximum_degree):
    for row in report["new_modes"]:
        validate_mode(row)
    rows = {row["degree"]: row for row in report["new_modes"]}
    assert len(rows) == len(report["new_modes"])
    results = []
    for index, fraction in enumerate(FRACTIONS):
        low = report["low_degree"]["actions"][index]
        assert low["fraction"] == fraction
        harmonics = {2: low["D2"]}
        harmonics.update({ell: rows[ell]["actions"][index]["D_degree"]
                          for ell in range(4, maximum_degree+1, 2)})
        finite = []
        for N in sorted(set([2, maximum_degree]+[n for n in (12, 32, 64, 128, 192) if n <= maximum_degree])):
            retained = {ell: value for ell, value in harmonics.items() if ell <= N}
            optimum = complete_orientation_minimum(low["D0"], retained, N, max(32, N+1))
            alternate = complete_orientation_minimum(low["D0"], retained, N, max(32, N+1)+32)
            difference = abs(optimum["action"]-alternate["action"])
            # Endpoint-concentrated densities amplify floating Gauss-node errors.
            assert difference < 2e-10
            if finite:
                assert optimum["action"] <= finite[-1]["action"]+2e-12
            finite.append({"degree": N, "quadrature_change": difference, **optimum})
        local = leading_action(source, fraction, 240)
        local_check = leading_action(source, fraction, 320)
        kernel = resummed_kernel(low["D0"], harmonics, local)
        results.append({"fraction": fraction, "D0": low["D0"], "D2": low["D2"],
            "leading_action": local, "leading_quadrature_change": abs(local-local_check),
            "negative_degrees": [ell for ell, value in harmonics.items() if value < 0],
            "finite_positive_minima": finite,
            "resummed_kernel_minimum": sampled_minimum(kernel),
            "resummed_kernel_at_pole": float(kernel(1.)),
            "resummed_kernel_scope": "Local high-degree completion only; no certified remainder bound."})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-degree", type=int, choices=range(4, 257, 2), default=12)
    parser.add_argument("--band", nargs=2, type=float, default=(.98, .99))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-quadrature", action="store_true",
                        help="Resume only the recorded 4000-point compact quadrature stop using 16000 points.")
    args = parser.parse_args()
    if not 0 < args.band[0] < args.band[1] < 1:
        parser.error("Use an energy band strictly inside the trapping well.")
    original = tuple(args.band) == (.98, .99)
    if args.output is None:
        suffix = "" if original else f"-{args.band[0]:.8g}-{args.band[1]:.8g}"
        args.output = ROOT/f"data/solitonic-screen/near-top-orientation{suffix}.json"
    start = perf_counter()
    input_path = ROOT/"data/solitonic-screen/near-separatrix-observations.json"
    observations = json.loads(input_path.read_text())
    background_path = ROOT/"data/s08a06/background.npz"
    metadata = {"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "background_sha256": hashlib.sha256(background_path.read_bytes()).hexdigest(),
                "energy_band_fractions": list(args.band), "source_normalization_order": 240 if original else 320,
                "tolerance": TOLERANCE, "maximum_nodes": MAX_NODES}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report.get("stopped_attempt"):
            stopped = report["stopped_attempt"]
            if not (args.resume_quadrature and stopped["message"] ==
                    "The compact source primitive exceeds 4000 quadrature points."):
                raise SystemExit("A stopped calculation is recorded. No automatic retry is permitted.")
            report.setdefault("previous_stops", []).append(stopped)
            report["stopped_attempt"] = None
            report["resource_decision"] = "User authorized autonomous resource choices on 2026-10-30; compact quadrature limit raised to 16000, BVP nodes retained at 4000."
        if any(report.get(key) != value for key, value in metadata.items()):
            raise ValueError("The saved run does not match these inputs and parameters.")
    else:
        report = {"scope": "Exploratory coarse radial responses and full finite positive orientations; no interval certificate.",
                  **metadata,
                  "new_modes": [], "elapsed_seconds": 0., "stopped_attempt": None}
    bg = Background(background_path)
    source = OrbitBandSource(bg, (.98, .99), order=240) if original else angle_normalized_source(bg, args.band)
    if "formulation" not in report:
        report["unshifted_comparison"] = report["new_modes"]
        report["new_modes"] = []
        report["formulation"] = "Derivative-shifted states with separate compact source primitives."
    report["source_factor"] = source.factor
    report["source_support"] = [source.a, source.b]
    report["maximum_compact_quadrature_points"] = 16000
    if not original:
        report.pop("imported_monopole_and_quadrupole", None)
    save(report, args.output)
    if "low_degree" not in report:
        if original:
            report["low_degree"] = observations["low_degree"]
            report["low_degree_origin"] = "Imported earlier coarse unshifted responses."
        else:
            try:
                mono = MeanResponse(bg, source, tolerance=TOLERANCE, max_nodes=MAX_NODES)
            except RuntimeError as error:
                report["stopped_attempt"] = {"degree": 0, "tolerance": TOLERANCE,
                    "maximum_nodes": MAX_NODES, "message": str(error)}
                save(report, args.output)
                raise SystemExit("Stopped monopole solve; review the recorded resource limit.") from error
            polar = solve(source, 2, TOLERANCE, report, args.output)
            actions = [{"fraction": f, "D0": split_monopole(mono, source, f, 80),
                        "D2": split_action(polar, f, 80)["D_degree"]} for f in FRACTIONS]
            report["low_degree"] = {"actions": actions, "monopole_nodes": len(mono.solution.x),
                "monopole_max_rms_residual": float(max(mono.solution.rms_residuals)),
                "monopole_charge_endpoint": float(mono.solution.sol(bg.radius)[4]),
                "polar_residuals": polar.residuals(), "tolerance": TOLERANCE,
                "compact_quadrature_points": polar._source_primitive.maximum_points_used,
                "quadrature_changes": [[abs(a["D0"]-split_monopole(mono, source, f, 120)),
                                        abs(a["D2"]-split_action(polar, f, 120)["D_degree"])]
                                       for f, a in zip(FRACTIONS, actions)]}
            report["low_degree_origin"] = "New fixed-charge monopole and derivative-shifted quadrupole."
        save(report, args.output)
    done = {row["degree"] for row in report["new_modes"]}
    for row in report["new_modes"]:
        validate_mode(row)
    for ell in range(4, args.maximum_degree+1, 2):
        if ell in done:
            continue
        response = solve(source, ell, TOLERANCE, report, args.output)
        residuals = response.residuals()
        actions = [split_action(response, fraction, 80) for fraction in FRACTIONS]
        changes = [abs(row["D_degree"]-split_action(response, fraction, 120)["D_degree"])
                   for fraction, row in zip(FRACTIONS, actions)]
        L, k, _ = angular_parameters(ell)
        row = {"degree": ell, "actions": actions, "residuals": residuals,
               "quadrature_changes": changes,
               "compact_quadrature_points": response._source_primitive.maximum_points_used,
               "scaled_actions": [(L-2)*a["D_degree"]/k**2 for a in actions]}
        report["new_modes"].append(row)
        report["new_modes"].sort(key=lambda value: value["degree"])
        save(report, args.output)
        # Save convergence evidence before checking independently reconstructed fields.
        validate_mode(row)
        if ell in (4, 16, 64, 128, 192, 256):
            coarser = solve(source, ell, 2e-5, report, args.output)
            row["coarser_tolerance"] = 2e-5
            row["coarser_nodes"] = len(coarser.solution.x)
            row["tolerance_changes"] = [abs(a["D_degree"]-split_action(coarser, fraction, 80)["D_degree"])
                                        for fraction, a in zip(FRACTIONS, actions)]
            save(report, args.output)
        if ell == 128 and original:
            known = observations["high_degree"]
            assert known["fraction"] == FRACTIONS[1]
            row["previous_finer_action_change"] = abs(actions[1]["D_degree"]-known["D_degree"])
            save(report, args.output)
        print(json.dumps({"degree": ell, "nodes": len(response.solution.x),
                          "midpoint_action": actions[1]["D_degree"]}), flush=True)
    report["probes"] = analyze(report, source, args.maximum_degree)
    report["timing_scope"] = "Completed invocation timings; interrupted postprocessing may leave converged modes without an accumulated duration."
    report["analyzed_maximum_degree"] = args.maximum_degree
    report["elapsed_seconds"] += perf_counter()-start
    save(report, args.output)
    print(json.dumps({"maximum_degree": args.maximum_degree,
                      "minima": [p["finite_positive_minima"][-1]["action"] for p in report["probes"]],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
