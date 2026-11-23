"""Verify the derivative shift and a fixed positive source with negative numerical feedback."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

if not __debug__:
    raise SystemExit("Do not disable verification assertions with python -O.")

import numpy as np
import sympy as sp
from mpmath import mp
from numpy.polynomial import Legendre
from scipy.interpolate import CubicHermiteSpline
from scipy.special import gammaln

from check_near_top_orientation import angle_normalized_source, split_action, split_monopole, validate_mode
from mean_response import MeanResponse
from polar_static import (CompactSourcePrimitive, OrbitBandSource, PolarResponse, angular_parameters,
                          coefficients, polar_flux_forcing, polar_forcing, polar_matrix, vacuum_metric_slope)
from trapped_axial import Background

ROOT = Path(__file__).parent
FRACTIONS = (.1, .5, .9, .99, .999)
INTERIOR_FRACTIONS = (.9982, .9985, .9988)


def background_hash():
    return hashlib.sha256((ROOT/"data/s08a06/background.npz").read_bytes()).hexdigest()


def validate_series(run, fractions):
    assert [r["degree"] for r in run["modes"]] == list(range(2, 257, 2))
    assert len(run["monopoles"]) == len(fractions) and np.all(np.isfinite(run["monopoles"]))
    assert run["monopole_nodes"] <= 4000
    for row in run["modes"]:
        assert row["nodes"] <= 4000 and row["compact_quadrature_points"] <= 16000
        assert len(row["actions"]) == len(fractions) and np.all(np.isfinite(row["actions"]))


def input_checks(bg):
    source = angle_normalized_source(bg, (.998, .999))
    cases = [{"order": value} for value in (3, 17, 8.5, True)]
    cases += [{"max_points": value} for value in (3999, 16001, 4000.5, True)]
    for kwargs in cases:
        try:
            CompactSourcePrimitive(source, 4, source.knots, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid primitive parameters accepted.")
    for value in (None, 1, "yes"):
        try:
            PolarResponse(source, source_derivative_shift=value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid shift flag accepted.")
    primitive = CompactSourcePrimitive(source, np.int64(4), source.knots, np.int64(8))
    try:
        primitive.integrand(np.array([source.a]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid primitive derivative accepted.")
    return len(cases)+4


def endpoint_density(m):
    coefficients = np.zeros(2*m+1)
    coefficients[::2] = 4*np.arange(m+1)+1
    polynomial = Legendre(coefficients)
    norm = (m+1)*(2*m+1)
    return polynomial*polynomial/norm


def gaunt_endpoint_coefficients(m):
    values = np.zeros(4*m+1)
    for i in range(0, 2*m+1, 2):
        for j in range(i, 2*m+1, 2):
            ell = np.arange(abs(i-j), i+j+1, 2)
            s = (i+j+ell)//2
            parts = np.array([s-i, s-j, s-ell])
            log_weight = 2*gammaln(s+1)-2*np.sum(gammaln(parts+1), axis=0)
            log_weight += np.sum(gammaln(2*parts+1), axis=0)-gammaln(2*s+2)
            multiplicity = 1 if i == j else 2
            values[ell] += multiplicity*(2*i+1)*(2*j+1)*(2*ell+1)*np.exp(log_weight)
    return values/((m+1)*(2*m+1))


def algebra():
    r, n, F, f, d, A, Ap, V, Om, D, Dp = sp.symbols("r n F f d A Ap V Om D Dp", nonzero=True)
    npp, App = F/f-d*n, (V-Om)*A/f-d*Ap
    original_v = (2*F/f-2*n*(1/r+n))*D+n*Dp
    original_u = Ap*(Dp-2*(1/r+n)*D)+2*A*(V-Om)*D/f
    shifted_v = original_v-d*n*D-npp*D-n*Dp
    shifted_u = original_u-d*Ap*D-App*D-Ap*Dp
    assert sp.simplify(shifted_v-(F/f-2*n*(1/r+n))*D) == 0
    assert sp.simplify(shifted_u-(A*(V-Om)/f-2*(1/r+n)*Ap)*D) == 0
    j, m = sp.symbols("j m", integer=True, nonnegative=True)
    assert sp.simplify(sp.summation(4*j+1, (j, 0, m))-(m+1)*(2*m+1)) == 0
    small = endpoint_density(3)
    z = sp.symbols("z")
    exact = sum((4*j+1)*sp.legendre(2*j, z) for j in range(4))**2/28
    for ell in range(0, 13, 2):
        value = sp.integrate(sp.expand(exact*sp.legendre(ell, z)), (z, -1, 1))*(2*ell+1)/2
        assert abs(float(value)-small.coef[ell]) < 2e-14
    return {"shift_identities": 2, "exact_endpoint_normalization": True,
            "rational_small_density_coefficients": 7}


def field_checks(bg):
    comparisons, primitive_changes, jet_errors = [], [], []
    maximum_points = 0
    for band in ((.98, .99), (.998, .999)):
        source = angle_normalized_source(bg, band)
        r = np.r_[np.linspace(.01, source.a-.01, 17), np.linspace(source.a, source.b, 401),
                  np.linspace(source.b+.01, bg.radius, 17)]
        for ell in (2, 4, 128, 256):
            g, damping, F, npp, App, *_ = coefficients(bg, r, ell)
            gain = angular_parameters(ell)[2]
            _, _, D, Dp = source.fields(r)
            D, Dp = gain*D, gain*Dp
            shift = np.zeros((5, len(r)))
            shift[1], shift[3] = g["nup"]*D, g["Ap"]*D/bg.central
            derivative = np.zeros_like(shift)
            derivative[1], derivative[3] = npp*D+g["nup"]*Dp, (App*D+g["Ap"]*Dp)/bg.central
            direct = polar_forcing(source, r, ell)+np.einsum("nij,jn->in", polar_matrix(bg, r, ell), shift)-derivative
            flux = polar_flux_forcing(source, r, ell)
            difference = float(np.max(abs(direct-flux)/(1+abs(direct)+abs(flux))))
            assert difference < 2e-12
            comparisons.append(difference)
        orbit_r = source.orbit.mapped(np.linspace(0, 1, 301))[0]
        knots = np.unique(np.r_[np.linspace(.002, bg.radius, 151), np.linspace(source.a, source.b, 201),
                               orbit_r, source.knots])
        first = CompactSourcePrimitive(source, 4, knots, 8)
        second = CompactSourcePrimitive(source, 4, knots, 12)
        values = first(r)
        change = float(np.max(abs(values-second(r))))
        assert change < 2e-9
        primitive_changes.append(change)
        maximum_points = max(maximum_points, first.maximum_points_used, second.maximum_points_used)
        # The zero collocation state tests reconstruction, not the field equations.
        state = np.zeros((5, len(knots)))
        spline = CubicHermiteSpline(knots, state, state, axis=1)
        fake = SimpleNamespace(x=knots, sol=spline, success=True, rms_residuals=np.zeros(len(knots)-1))
        for shifted in (False, True):
            with patch("polar_static.solve_bvp", return_value=fake):
                response = PolarResponse(source, degree=4, source_derivative_shift=shifted)
            points = np.linspace(source.a+.01, source.b-.01, 31)
            h = 1e-4
            center = response.physical_jet(points)
            jets = [response.physical_jet(points+k*h) for k in (-2, -1, 1, 2)]
            for value, derivative in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)):
                difference = (jets[0][value]-8*jets[1][value]+8*jets[2][value]-jets[3][value])/(12*h)
                error = float(np.max(abs(difference-center[derivative])/(1+abs(center[derivative]))))
                assert error < 2e-6, (band, shifted, value, error)
                jet_errors.append(error)
    with mp.workdps(100):
        M, R, ell = mp.mpf(5), mp.mpf(40), 256
        x = R/M-1
        ratio = mp.diff(lambda t: mp.legenq(ell, 2, t, type=3), x)/(M*mp.legenq(ell, 2, x, type=3))
        vacuum_difference = abs(vacuum_metric_slope(float(M), float(R), ell)-float(mp.re(ratio)))
    assert vacuum_difference < 2e-13
    return {"forcing_identity_maximum": max(comparisons), "primitive_order_change": max(primitive_changes),
            "manufactured_jet_scaled_difference": max(jet_errors), "manufactured_jet_checks": len(jet_errors),
            "maximum_primitive_points": maximum_points, "degree256_vacuum_difference": vacuum_difference}


def density_checks(record):
    results = []
    nodes, weights = np.polynomial.legendre.leggauss(384)
    for m in (48, 64):
        density = endpoint_density(m)
        gaunt = gaunt_endpoint_coefficients(m)
        assert abs(density.coef[0]-1) < 2e-13
        assert min(density.coef[::2]) > 0 and np.all(density.coef[1::2] == 0)
        assert abs(sum(density.coef)-(m+1)*(2*m+1)) < 2e-9
        assert np.max(abs(density.coef-gaunt)) < 2e-9
        coefficients = np.array([record["low_degree"]["actions"][i]["D0"] for i in range(5)])
        values = []
        for i in range(5):
            D = np.zeros(4*m+1)
            D[0], D[2] = coefficients[i], record["low_degree"]["actions"][i]["D2"]
            for row in record["new_modes"]:
                if row["degree"] <= 4*m:
                    D[row["degree"]] = row["actions"][i]["D_degree"]
            value = float(density.coef@D)
            alternate = float(gaunt@D)
            kernel = Legendre(D*(2*np.arange(len(D))+1))
            direct = float(weights@(density(nodes)*kernel(nodes))/2)
            assert abs(value-alternate) < 2e-11 and abs(value-direct) < 2e-10
            values.append({"fraction": FRACTIONS[i], "action": value,
                           "gaunt_difference": abs(value-alternate), "quadrature_difference": abs(value-direct)})
        results.append({"m": m, "maximum_degree": 4*m, "exact_normalization": (m+1)*(2*m+1),
                        "legendre_coefficients": density.coef.tolist(), "actions": values})
    return results


def coarse_responses(bg, source, path):
    mono = MeanResponse(bg, source, tolerance=2e-5, max_nodes=4000)
    result = {"background_sha256": background_hash(), "source_band": [.998, .999],
              "fractions": list(FRACTIONS), "primitive_order": 16,
              "tolerance": 2e-5, "monopole_nodes": len(mono.solution.x),
              "monopole_max_rms_residual": float(max(mono.solution.rms_residuals)),
              "monopoles": [split_monopole(mono, source, f, 120) for f in FRACTIONS], "modes": []}
    def persist():
        path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    persist()
    try:
        for ell in range(2, 257, 2):
            response = PolarResponse(source, degree=ell, tolerance=2e-5, source_derivative_shift=True,
                                     source_integral_order=16)
            result["modes"].append({"degree": ell, "nodes": len(response.solution.x),
                "actions": [split_action(response, f, 120)["D_degree"] for f in FRACTIONS],
                "compact_quadrature_points": response._source_primitive.maximum_points_used})
            persist()
            if ell % 64 == 0:
                print(json.dumps({"coarse_degree": ell}), flush=True)
    except (RuntimeError, ValueError) as error:
        result["stopped_attempt"] = str(error)
        persist()
        raise
    return result


def interior_responses(bg, path):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("complete"):
            assert saved["background_sha256"] == background_hash()
            assert saved["source_band"] == [.998, .999] and saved["density_m"] == 64
            assert saved["fractions"] == list(INTERIOR_FRACTIONS)
            assert [r["tolerance"] for r in saved["runs"]] == [2e-6, 2e-5]
            assert [r["primitive_order"] for r in saved["runs"]] == [12, 16]
            h = endpoint_density(64).coef
            for run in saved["runs"]:
                validate_series(run, INTERIOR_FRACTIONS)
                recomputed = [run["monopoles"][i]+sum(h[r["degree"]]*r["actions"][i] for r in run["modes"])
                              for i in range(len(INTERIOR_FRACTIONS))]
                assert np.max(abs(np.array(recomputed)-run["density_actions"])) < 2e-13
            return saved
        raise SystemExit("An incomplete interior comparison is retained; inspect before resuming.")
    source = angle_normalized_source(bg, (.998, .999))
    h = endpoint_density(64).coef
    report = {"background_sha256": background_hash(),
              "fractions": list(INTERIOR_FRACTIONS), "source_band": [.998, .999],
              "density_m": 64, "runs": [], "complete": False}
    for tolerance, order in ((2e-6, 12), (2e-5, 16)):
        mono = MeanResponse(bg, source, tolerance=tolerance, max_nodes=4000)
        run = {"tolerance": tolerance, "primitive_order": order, "monopole_nodes": len(mono.solution.x),
               "monopoles": [split_monopole(mono, source, f, 120) for f in INTERIOR_FRACTIONS], "modes": []}
        report["runs"].append(run)
        path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        for ell in range(2, 257, 2):
            response = PolarResponse(source, degree=ell, tolerance=tolerance,
                                     source_derivative_shift=True, source_integral_order=order)
            values = [split_action(response, f, 120)["D_degree"] for f in INTERIOR_FRACTIONS]
            run["modes"].append({"degree": ell, "nodes": len(response.solution.x), "actions": values,
                                 "compact_quadrature_points": response._source_primitive.maximum_points_used})
            path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        run["density_actions"] = [run["monopoles"][i]+sum(h[r["degree"]]*r["actions"][i] for r in run["modes"])
                                  for i in range(len(INTERIOR_FRACTIONS))]
        print(json.dumps({"interior_tolerance": tolerance, "actions": run["density_actions"]}), flush=True)
    report["tolerance_changes"] = [abs(a-b) for a, b in zip(report["runs"][0]["density_actions"],
                                                           report["runs"][1]["density_actions"])]
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-check", action="store_true")
    parser.add_argument("--interior-check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"data/solitonic-screen/polar-flux.json")
    args = parser.parse_args()
    start = perf_counter()
    path = ROOT/"data/solitonic-screen/near-top-orientation-0.998-0.999.json"
    record = json.loads(path.read_text())
    assert record["analyzed_maximum_degree"] == 256 and record["stopped_attempt"] is None
    assert record["background_sha256"] == background_hash()
    assert record["energy_band_fractions"] == [.998, .999]
    assert [r["degree"] for r in record["new_modes"]] == list(range(4, 257, 2))
    for row in record["new_modes"]:
        validate_mode(row)
    bg = Background()
    report = {"scope": "Exact variable-change identities and numerical finite positive-source candidates; no rigorous feedback certificate.",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "algebra": algebra(),
        "rejected_inputs": input_checks(bg),
        "field_checks": field_checks(bg), "fixed_densities": density_checks(record)}
    coarse_path = ROOT/"data/solitonic-screen/near-top-coarse.json"
    if args.response_check:
        if coarse_path.exists():
            coarse = json.loads(coarse_path.read_text())
            if coarse.get("stopped_attempt") or len(coarse["modes"]) != 128:
                raise SystemExit("Incomplete coarse comparison retained; inspect before retrying.")
        else:
            coarse = coarse_responses(bg, angle_normalized_source(bg, (.998, .999)), coarse_path)
        assert coarse["background_sha256"] == background_hash()
        assert coarse["source_band"] == [.998, .999] and coarse["fractions"] == list(FRACTIONS)
        assert coarse["tolerance"] == 2e-5 and coarse["primitive_order"] == 16
        validate_series(coarse, FRACTIONS)
        comparisons = []
        for density in report["fixed_densities"]:
            h = np.array(density["legendre_coefficients"])
            for i, f in enumerate(FRACTIONS):
                coarse_value = coarse["monopoles"][i]+sum(h[row["degree"]]*row["actions"][i]
                    for row in coarse["modes"] if row["degree"] < len(h))
                comparisons.append({"degree": density["maximum_degree"], "fraction": f,
                    "fine_action": density["actions"][i]["action"], "coarse_action": coarse_value,
                    "change": abs(coarse_value-density["actions"][i]["action"])})
        report["complete_tolerance_comparison"] = comparisons
        report["coarse_input_sha256"] = hashlib.sha256(coarse_path.read_bytes()).hexdigest()
    if args.interior_check:
        interior_path = ROOT/"data/solitonic-screen/near-top-interior.json"
        interior = interior_responses(bg, interior_path)
        report["interior_input_sha256"] = hashlib.sha256(interior_path.read_bytes()).hexdigest()
        report["source_band_interior"] = {"fractions": interior["fractions"],
            "fine_actions": interior["runs"][0]["density_actions"],
            "coarse_actions": interior["runs"][1]["density_actions"], "changes": interior["tolerance_changes"]}
    report["elapsed_seconds"] = perf_counter()-start
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "field_checks": report["field_checks"],
        "explicit_degree256_actions": [row["action"] for row in report["fixed_densities"][-1]["actions"]],
        "maximum_tolerance_change": max((r["change"] for r in report.get("complete_tolerance_comparison", [])), default=None)}))


if __name__ == "__main__":
    main()
