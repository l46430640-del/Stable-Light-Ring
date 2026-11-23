"""Numerical residual budgets for the explicit positive near-top source.

The finite-domain integrals guide an eventual enclosure. They omit exact
background/source transfer and global endpoint errors, and are not a sign
certificate. Saved collocation polynomials are reused without solving again.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Run the residual checks without python -O.")

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from check_near_top_orientation import angle_normalized_source, split_action
from check_polar_flux import endpoint_density
from polar_optical_dual import OpticalDualBound
from polar_optical_tail import OpticalWeights
from polar_static import PolarResponse, angular_parameters
from polar_transverse import TransverseDeformation, transverse_coefficients
from trapped_axial import Background, gauss_mesh

ROOT = Path(__file__).parent
CACHE = ROOT/"data/solitonic-screen/polar-residual-fields"
MAX_RESIDUAL_POINTS = 20000


class RepresentedResponse:
    def __init__(self, source, degree, tolerance, build=False):
        self.source, self.background, self.degree = source, source.background, degree
        self.eigenvalue, self.funk, self.gain = angular_parameters(degree)
        self.tolerance = tolerance
        digest = hashlib.sha256((ROOT/"data/s08a06/background.npz").read_bytes()).hexdigest()
        path = CACHE/f"degree-{degree:03d}-tol-{tolerance:.0e}.npz"
        self.built = False
        if not path.exists():
            if not build:
                raise FileNotFoundError(f"No saved field for degree {degree}; use --build explicitly.")
            response = PolarResponse(source, degree=degree, tolerance=tolerance,
                                     source_derivative_shift=True, source_integral_order=12)
            CACHE.mkdir(parents=True, exist_ok=True)
            knots = response.solution.sol.x
            np.savez_compressed(path, knots=knots, state=response.solution.sol(knots),
                slope=response.solution.sol(knots, 1),
                degree=degree, tolerance=tolerance, background_sha256=np.array(digest),
                band=np.array([source.floor, source.ceiling]), source_factor=source.factor,
                max_rms_residual=max(response.solution.rms_residuals))
            self.built = True
        with np.load(path) as data:
            assert str(data["background_sha256"]) == digest
            assert int(data["degree"]) == degree and float(data["tolerance"]) == tolerance
            assert np.array_equal(data["band"], [source.floor, source.ceiling])
            assert float(data["source_factor"]) == source.factor
            knots, state, slope = [data[key].copy() for key in ("knots", "state", "slope")]
            assert len(knots) <= 4000 and np.all(np.diff(knots) > 0)
            assert state.shape == slope.shape == (5, len(knots))
            assert np.all(np.isfinite(state)) and np.all(np.isfinite(slope))
            self.max_rms = float(data["max_rms_residual"])
            assert self.max_rms <= tolerance
            self.sol = CubicHermiteSpline(knots, state, slope, axis=1)
        self.inner, self.outer = float(knots[0]), float(knots[-1])
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.path = path

    def optical(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r < self.inner) or np.any(r > self.outer):
            raise ValueError("Use radii inside the saved response interval.")
        y = self.sol(r)
        D = self.gain*self.source.fields(r)[2]
        return self.funk*(2*y[0]-y[4]), self.funk*(D-y[4])


def residuals(response, deformation, radius):
    r = np.atleast_1d(radius)
    bg = response.background
    c = transverse_coefficients(bg, r)
    T = deformation(r)
    state, prime = response.sol(r), response.sol(r, 1)
    factors = np.array([1., np.sqrt(4*np.pi)*bg.central])[:, None]
    y, flux = (state[index]*factors for index in ([0, 2], [1, 3]))
    yp, fluxp = (prime[index]*factors for index in ([0, 2], [1, 3]))
    rho, _, D, _ = response.source.fields(r)
    D = response.gain*D
    mu = response.eigenvalue-2
    Q = c["Q1"]+mu*c["W"][:, None, None]*np.eye(2)
    r0 = yp.T-flux.T-c["a"]*D[:, None]
    r1 = fluxp.T+c["d"][:, None]*flux.T-np.einsum("nij,nj->ni", Q, y.T)
    r1 -= np.einsum("nij,nj->ni", T, c["a"])*D[:, None]
    g = bg.fields(r)
    r1[:, 0] -= response.funk*8*np.pi*rho/g["f"]
    rw = prime[4]+2*np.sum(c["a"]*y.T, axis=1)-(1/r+g["nup"])*D
    return r0, r1-np.einsum("nij,nj->ni", T, r0), rw, c


def finite_budget(response, deformation, weights, order=6):
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 4 <= order <= 10:
        raise ValueError("Use a residual quadrature order from four through ten.")
    probe = weights.probe
    edges = np.unique(np.r_[response.sol.x, response.source.knots,
                            deformation.breakpoints, probe.a, probe.b, probe.c])
    assert edges[0] == response.inner and edges[-1] == response.outer
    r, q = gauss_mesh(edges, int(order))
    if len(r) > MAX_RESIDUAL_POINTS:
        raise RuntimeError(f"The finite residual quadrature exceeds {MAX_RESIDUAL_POINTS} points.")
    r0, r1, rw, c = residuals(response, deformation, r)
    mu = response.eigenvalue-2
    norm2 = q@(c["P"]*np.sum(r0*r0, axis=1)+c["P"]/c["W"]*np.sum(r1*r1, axis=1)/mu)
    reconstruction = float(q@(abs(weights.H(r)*rw)*(r < probe.c)))
    dual = OpticalDualBound(weights, mu, deformation=deformation).evaluate(24)
    value = abs(response.funk)*(dual["optical_dual_bound"]*np.sqrt(norm2)+reconstruction)
    return {"order": int(order), "points": len(r), "residual_norm": float(np.sqrt(norm2)),
            "optical_dual": dual["optical_dual_bound"], "angular_reconstruction": reconstruction,
            "finite_action_error_estimate": float(value),
            "maximum_r0": float(np.max(abs(r0))), "maximum_r1_minus_T_r0": float(np.max(abs(r1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum-degree", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--output", type=Path, default=ROOT/"data/solitonic-screen/polar-residual-budget.json")
    args = parser.parse_args()
    angular_parameters(args.maximum_degree)
    if args.tolerance not in (2e-6, 2e-7):
        parser.error("Use the saved coarse tolerance or one bounded refinement, 2e-6 or 2e-7.")
    history = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if (previous.get("maximum_degree") != args.maximum_degree
                or previous.get("tolerance") != args.tolerance):
            raise SystemExit("Use a different output path for a different degree range or tolerance.")
        history = previous.get("previous_stopped_attempts", [])
        if previous.get("stopped_attempt"):
            if not args.resume:
                raise SystemExit("A stopped calculation is retained; inspect it before using --resume.")
            history.append(previous["stopped_attempt"])
    started = perf_counter()
    bg = Background()
    source = angle_normalized_source(bg, (.998, .999))
    deformation = TransverseDeformation(bg)
    weights = OpticalWeights(bg, .9985, count=1601)
    density = endpoint_density(64).coef
    original = json.loads((ROOT/"data/solitonic-screen/near-top-interior.json").read_text())
    reference = {row["degree"]: row["actions"][1] for row in original["runs"][0]["modes"]}
    report = {"scope": "Numerical finite-domain residual budget only; endpoint contributions are separate, and interior/background/source/optical enclosures remain open.",
              "tolerance": args.tolerance, "probe_fraction": .9985, "source_band": [.998, .999],
              "maximum_degree": args.maximum_degree, "maximum_residual_points": MAX_RESIDUAL_POINTS,
              "previous_stopped_attempts": history,
              "complete": False, "rows": [], "stopped_attempt": None}
    def persist():
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    try:
        for degree in range(2, args.maximum_degree+1, 2):
            response = RepresentedResponse(source, degree, args.tolerance, args.build)
            budgets = [finite_budget(response, deformation, weights, order) for order in (6, 8)]
            action = split_action(response, .9985, 120)["D_degree"]
            report["rows"].append({"degree": degree, "nodes": len(response.sol.x),
                "field_sha256": response.sha256, "built": response.built, "action": action,
                "difference_from_reconstructed_action": abs(action-reference[degree]),
                "orientation_weight": float(density[degree]), "budgets": budgets,
                "inner_state": response.sol(response.inner).tolist(),
                "outer_state": response.sol(response.outer).tolist()})
            persist()
            if degree <= 8 or degree % 32 == 0:
                print(json.dumps({"degree": degree, "budget": budgets[-1]["finite_action_error_estimate"]}), flush=True)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        report["stopped_attempt"] = {"degree": degree, "message": str(error)}
        persist()
        raise
    report["complete"] = True
    report["weighted_finite_error_estimate"] = sum(row["orientation_weight"]*row["budgets"][-1]["finite_action_error_estimate"] for row in report["rows"])
    report["weighted_representation_change"] = sum(row["orientation_weight"]*row["difference_from_reconstructed_action"] for row in report["rows"])
    report["elapsed_seconds"] = perf_counter()-started
    persist()
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
