"""Enclose quadrupolar first-order residuals on the exact background.

The shared Hermite nodal data define the trial fields exactly. Background
and source uncertainty is retained throughout. Optical error constants and
the monopole are not certified by this calculation.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Run residual enclosures without python -O.")

import numpy as np
import sympy as sp
from mpmath import iv

from check_background_defect import field_equations
from check_barrier_certificate import Jet, exact, enclosure
from check_finite_static_inverse import upper
from check_polar_band_source import DOMAIN, log_potential_cell, source_partition
from check_polar_coercivity import check_inputs, physical_cell
from check_polar_transverse import hermite
from check_near_top_orientation import angle_normalized_source
from polar_band_moments import band_moments
from polar_residual_budget import RepresentedResponse, residuals
from polar_transverse import TransverseDeformation, transverse_coefficients
from trapped_axial import Background

ROOT = Path(__file__).parent
DATA = ROOT/"data/solitonic-screen"
MAX_CELLS = 6500
MAX_SECONDS = 120


def coefficients(r, y, transverse):
    A0, sigma, pi, c = [Jet(v, order=1) for v in (iv.mpf(".06"), iv.mpf(".08"), iv.pi, iv.sqrt(4*iv.pi))]
    A, Ap, f, zeta = A0*y[0], A0*y[1], 1-2*y[2], y[3]
    O2 = (2*zeta).exp()
    ratio = A*A/(sigma*sigma)
    V = A*A*(1-2*ratio)*(1-2*ratio)
    Vs, Vss = (1-2*ratio)*(1-6*ratio), (-8+24*ratio)/(sigma*sigma)
    rho, pressure = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    n = (y[2]/r+4*pi*r*pressure)/f
    fp = 2*y[2]/r-8*pi*r*rho
    d = 2/r+n+fp/(2*f)
    F = 4*pi*(2*O2*A*A-V)
    np_ = F/f-d*n
    App = (Vs-O2)*A/f-d*Ap
    av, ap = [n, c*Ap], [np_, c*App]
    norm = av[0]*av[0]+av[1]*av[1]
    assert norm.c[0].a > 0
    gamma = (av[0]*ap[0]+av[1]*ap[1])/norm-n+fp/(2*f)
    turn = (av[0]*ap[1]-av[1]*ap[0])/norm
    aa, bb, ab = av[0]*av[0], av[1]*av[1], av[0]*av[1]
    T11 = (gamma*aa-2*turn*ab+transverse*bb)/norm
    T12 = ((gamma-transverse)*ab+turn*(aa-bb))/norm
    T22 = (gamma*bb+2*turn*ab+transverse*aa)/norm
    T = [[T11, T12], [T12, T22]]
    W = 1/(r*r*f)
    cross = c*(4*n*Ap+2*A*(2*O2-Vs)/f)
    Q = [[6*W+4*n*n+8*pi*(V-4*O2*A*A)/f, cross],
         [cross, 6*W+(Vs+2*A*A*Vss-O2)/f+16*pi*Ap*Ap]]
    G = n-fp/(2*f)
    return dict(a=av, b=[ap[j]-G*av[j] for j in range(2)], d=d, Q=Q, T=T, n=n, f=f, W=W)


class ResidualEnclosure:
    def __init__(self, relative_path=None):
        self.bg = Background()
        self.hashes = check_inputs()
        path = DATA/"polar-band-source.json"
        self.source_certificate = json.loads(path.read_text())
        source_record = self.source_certificate
        assert source_record["status"] == "positive_band_source_and_normalization_enclosed"
        assert source_record["input_sha256"]["background.npz"] == self.hashes["background.npz"]
        assert source_record["input_sha256"]["global-background.json"] == self.hashes["global-background.json"]
        assert source_record["band"] == [".998", ".999"]
        self.hashes["polar-band-source.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.Ws = enclosure(*source_record["stable_reduced_potential"])
        self.depth = enclosure(*source_record["reduced_well_depth"])
        self.omega = enclosure(*source_record["frequency"])
        self.amplitude = exact(Fraction(source_record["reference_factor_exact_rational"]))
        source = angle_normalized_source(self.bg, (.998, .999))
        self.response = RepresentedResponse(source, 2, 2e-7)
        assert self.response.sha256 == source_record["input_sha256"]["reference_response.npz"]
        self.hashes["response.npz"] = self.response.sha256
        with np.load(self.response.path) as saved:
            self.knots, self.state, self.slope = [saved[name].copy() for name in ("knots", "state", "slope")]
        self.knots = [Fraction(float(v)) for v in self.knots]
        self.background_knots = [Fraction(float(v)) for v in self.bg.sol.x]
        deformation = TransverseDeformation(self.bg)
        self.deformation_knots = [Fraction(float(v)) for v in deformation.spline.x]
        path = DATA/"polar-transverse-deformation.npz"
        with np.load(path) as saved:
            self.values, self.slopes = saved["values"].copy(), saved["slopes"].copy()
        self.hashes["deformation.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.deformation = deformation
        self.partition = sorted(set(self.knots) | set(self.background_knots[1:])
                                | set(self.deformation_knots) | set(source_partition(self.bg, source, 128)))
        self.relative = None
        if relative_path is not None:
            from check_relative_optics import RelativePotential
            self.relative = RelativePotential.load(self.bg, relative_path)
            self.hashes["relative-optics.json"] = self.relative.record_sha256
            self.partition = sorted(set(self.partition) | set(self.relative.nodes))
        assert self.partition[0] == self.knots[0] and self.partition[-1] == 40
        if len(self.partition)-1 > MAX_CELLS:
            raise RuntimeError("The residual partition exceeds 6500 cells.")

    def transverse(self, r, middle):
        if middle < 15:
            return 11*r/(4*(1+r))
        if middle > 20:
            return Jet(0, order=1)
        i = bisect_right(self.deformation_knots, middle)-1
        return hermite(r, self.deformation_knots[i], self.deformation_knots[i+1],
                       self.values[i:i+2], self.slopes[i:i+2])[0]

    def source_fields(self, r, bg, c, left, right, point=False):
        middle = (left+right)/2
        radius = r.c[0]
        density, D = Jet(0, order=1), Jet(0, order=1)
        if DOMAIN[0] < middle < DOMAIN[1]:
            lo, hi = (middle, middle) if point else (left, right)
            logW, _ = log_potential_cell(self.bg, self.background_knots, lo, hi)
            W = iv.exp(logW)
            x = (W-self.Ws)/self.depth
            fraction_factor = W/self.depth
            if self.relative is not None:
                correlated_x, correlated_factor = self.relative.fraction(lo, hi)
                x = enclosure(max(x.a, correlated_x.a), min(x.b, correlated_x.b))
                fraction_factor = enclosure(max(fraction_factor.a, correlated_factor.a),
                                           min(fraction_factor.b, correlated_factor.b))
            J0, J1, J0p = band_moments(x, derivative=True)
            xp = 2*fraction_factor*(c["n"].c[0]-1/radius)
            j0, j1 = Jet(0, [J0, J0p*xp]), Jet(0, [J1, -J0*xp/2])
            pressure = (2*bg[3]).exp()*self.amplitude*self.depth*j1/(r*r*(4*iv.pi))
            density = j0*self.amplitude/(r*r*r*r*(4*iv.pi))+pressure
            D = j0*self.amplitude/(4*r*r)
        return density, D

    def evaluate(self, left, right, point=False):
        middle = (left+right)/2
        radius = exact(middle) if point else enclosure(left, right)
        piece = bisect_right(self.background_knots, middle)-1
        index = bisect_right(self.knots, middle)-1
        y0 = physical_cell(self.bg, piece, radius.a, radius.b)
        flow = field_equations(radius, y0, iv.mpf(".06"), iv.mpf(".08"), iv.pi)
        r = Jet(0, [radius, iv.mpf(1)])
        bg = [Jet(0, [v, F]) for v, F in zip(y0, flow)]
        c = coefficients(r, bg, self.transverse(r, middle))
        state, derivative = [], []
        for j in range(5):
            value, prime = hermite(r, self.knots[index], self.knots[index+1],
                                   self.state[j, index:index+2], self.slope[j, index:index+2])
            state.append(value)
            derivative.append(prime)
        scale = Jet(iv.sqrt(4*iv.pi)*iv.mpf(".06"), order=1)
        y, yp = [state[0], scale*state[2]], [derivative[0], scale*derivative[2]]
        flux, fluxp = [state[1], scale*state[3]], [derivative[1], scale*derivative[3]]
        density, D = self.source_fields(r, bg, c, left, right, point)
        r0 = [yp[j]-flux[j]-c["a"][j]*D for j in range(2)]
        q = [fluxp[j]+c["d"]*flux[j]
             -sum((c["Q"][j][k]*y[k]+c["T"][j][k]*(yp[k]-flux[k]) for k in range(2)), Jet(0, order=1))
             +density*(4*iv.pi)/c["f"]*int(j == 0) for j in range(2)]
        rw = derivative[4]+2*sum((c["a"][j]*y[j] for j in range(2)), Jet(0, order=1))-(1/r+c["n"])*D
        P = radius**2*self.omega*iv.exp(-y0[3])*iv.sqrt(c["f"].c[0])
        return r0+q+[rw], P, c["W"].c[0]

    def cell(self, left, right):
        middle, half = (left+right)/2, (right-left)/2
        center, _, _ = self.evaluate(left, right, point=True)
        ranged, P, W = self.evaluate(left, right)
        bounds = []
        for one, all_values in zip(center, ranged):
            mean = one.c[0]+enclosure(-half, half)*all_values.c[1]
            lo, hi = max(mean.a, all_values.c[0].a), min(mean.b, all_values.c[0].b)
            assert lo <= hi
            bounds.append(iv.mpf([lo, hi]))
        energy = exact(right-left)*P*(sum((abs(v)**2 for v in bounds[:2]), iv.mpf(0))
                                      +sum((abs(v)**2 for v in bounds[2:4]), iv.mpf(0))/(4*W))
        angular = exact(right-left)*abs(bounds[4])
        return upper(energy), upper(angular), [upper(abs(v)) for v in bounds]


def algebra_checks():
    yp, flux, fluxp, av, y, forcing = [sp.Matrix(sp.symbols(name+"0:2"))
                                     for name in ("yp", "flux", "fluxp", "a", "y", "R")]
    t0, t1, t2, q0, q1, q2, d, D, k = sp.symbols("t0 t1 t2 q0 q1 q2 d D k")
    T, Q = sp.Matrix([[t0, t1], [t1, t2]]), sp.Matrix([[q0, q1], [q1, q2]])
    r0 = yp-flux-av*D
    r1 = fluxp+d*flux-Q*y-T*av*D-k*forcing
    simplified = fluxp+d*flux-Q*y-T*(yp-flux)-k*forcing
    assert all(sp.expand(v) == 0 for v in r1-T*r0-simplified)
    return {"anisotropy_cancellation_in_regular_residual": True}


def implementation_checks(obj):
    radii = [.003, .1, 5., 14., 15.2, 15.6234, 15.8, 16., 16.515, 19.995, 20.1, 30., 39.9]
    maximum = 0.
    for radius in radii:
        r = Jet(exact(radius), order=1)
        bg = [Jet(exact(v), order=1) for v in obj.bg.sol(radius)]
        c = coefficients(r, bg, obj.transverse(r, Fraction(radius)))
        independent = transverse_coefficients(obj.bg, [radius])
        Q = independent["Q1"][0]+4*independent["W"][0]*np.eye(2)
        for key, values in (("Q", Q), ("T", obj.deformation([radius])[0])):
            actual = np.array([[float(v.c[0].mid) for v in row] for row in c[key]])
            difference = float(np.max(abs(actual-values)/(1+abs(values))))
            assert difference < 3e-11, (radius, key, difference, actual, values)
            maximum = max(maximum, difference)
        a, b, w, _ = residuals(obj.response, obj.deformation, [radius])
        reference = np.r_[a[0], b[0], w[0]]
        enclosed, _, _ = obj.evaluate(Fraction(radius), Fraction(radius), point=True)
        for value, box in zip(reference, enclosed):
            # Float evaluation is checked against the continuous enclosure,
            # with a separate allowance for the independent code's roundoff.
            assert box.c[0].a-iv.mpf("1e-12") <= value <= box.c[0].b+iv.mpf("1e-12")
    assert maximum < 3e-11
    return {"independent_radii": len(radii), "coefficient_scaled_difference": maximum,
            "independent_residual_components_checked": 5*len(radii)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, choices=(30, 50), default=30)
    parser.add_argument("--region", choices=("source", "complement", "full"), default="source")
    parser.add_argument("--max-seconds", type=int, choices=(120, 180), default=MAX_SECONDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--recheck-worst", type=Path)
    parser.add_argument("--relative-optics", type=Path)
    args = parser.parse_args()
    iv.dps = args.digits
    start = perf_counter()
    enclosure_object = ResidualEnclosure(args.relative_optics)
    checks = dict(algebra_checks(), **implementation_checks(enclosure_object))
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    if args.recheck_worst:
        previous = json.loads(args.recheck_worst.read_text())
        assert previous["status"] == "quadrupole_interval_residual_enclosed"
        assert previous["input_sha256"] == enclosure_object.hashes
        rows = []
        for row in previous["worst_cells"]:
            a, b = map(Fraction, row["interval"])
            energy, angular, components = enclosure_object.cell(a, b)
            assert energy <= row["energy_squared_upper"]
            assert angular <= row["angular_L1_upper"]
            assert all(v <= old for v, old in zip(components, row["residual_components_upper"]))
            rows.append(dict(row, energy_squared_upper=energy, angular_L1_upper=angular,
                             residual_components_upper=components))
        report = {"status": "saved_worst_cells_rechecked", "digits": args.digits,
                  "input_sha256": enclosure_object.hashes, "checks": checks, "cells": rows,
                  "scope": "Precision recheck of the ten recorded worst cells only, not a new full integral."}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("cells", "input_sha256")}, indent=2))
        return
    nodes = enclosure_object.partition
    if args.region == "source":
        nodes = [v for v in nodes if DOMAIN[0] <= v <= DOMAIN[1]]
    cells = list(zip(nodes[:-1], nodes[1:]))
    if args.region == "complement":
        cells = [(a, b) for a, b in cells if b <= DOMAIN[0] or a >= DOMAIN[1]]
    report = {"status": "running", "degree": 2, "digits": args.digits, "region": args.region,
              "input_sha256": enclosure_object.hashes, "partition_points": len(nodes),
              "checks": checks, "maximum_cells": MAX_CELLS, "maximum_seconds": args.max_seconds,
              "processed_cells": 0, "worst_cells": []}
    energy, angular = iv.mpf(0), iv.mpf(0)
    def persist():
        if args.output:
            args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    try:
        for a, b in cells:
            if perf_counter()-start > args.max_seconds:
                raise RuntimeError(f"The interval residual calculation reached its {args.max_seconds}-second budget.")
            E, A, components = enclosure_object.cell(a, b)
            energy += exact(E)
            angular += exact(A)
            report["processed_cells"] += 1
            row = {"interval": [str(a), str(b)], "energy_squared_upper": E,
                   "angular_L1_upper": A, "residual_components_upper": components}
            report["worst_cells"] = sorted(report["worst_cells"]+[row], key=lambda v: v["energy_squared_upper"], reverse=True)[:10]
            if report["processed_cells"] % 500 == 0:
                persist()
        report.update(status="quadrupole_interval_residual_enclosed",
                      energy_dual_residual_upper=upper(iv.sqrt(energy)),
                      angular_L1_residual_upper=upper(angular),
                      scope="Exact-background residual bound on the stated region; optical weights and monopole remain open.")
    except (RuntimeError, AssertionError, ValueError, NotImplementedError) as error:
        report.update(status="stopped", reason=str(error))
        persist()
        raise
    report["elapsed_seconds"] = perf_counter()-start
    persist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
