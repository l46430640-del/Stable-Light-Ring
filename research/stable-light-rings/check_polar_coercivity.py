"""Certify a global polar coercivity bound on the exact original star.

Use the saved nonlinear background enclosure. The small central interval
is handled by its regular Volterra variables; logarithmic scalar slopes
in the outer region are controlled by Riccati barriers, not division by
an exponentially small numerical scalar amplitude.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Do not disable interval verification with python -O.")

import numpy as np
import sympy as sp
from mpmath import iv

from check_asymptotic_exterior import lower
from check_background_defect import background_jets
from check_barrier_certificate import enclosure, exact
from check_finite_static_inverse import upper
from check_polar_factorization import certify_tail
from check_regular_center import Dual, integrands, leading, matter
from finite_static_inverse import WEIGHTS
from polar_factorization import exterior_negative_trace
from polar_static import coefficients
from trapped_axial import Background

ROOT = Path(__file__).parent
DATA = ROOT/"data/s08a06"
ERROR = "7.2e-9"


def check_inputs():
    names = ("background.npz", "global-background.json", "finite-static-reference.npz", "background-defect.npz")
    hashes = {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in names}
    report = json.loads((DATA/"global-background.json").read_text())
    assert report["status"] == "nonlinear_global_background_validated"
    assert report["background_sha256"] == hashes["background.npz"]
    assert report["reference_sha256"] == hashes["finite-static-reference.npz"]
    assert report["defect_enclosure_sha256"] == hashes["background-defect.npz"]
    assert exact(report["fixed_point"]["decimal_weight_error_bound"]) < iv.mpf(ERROR)
    assert exact(report["endpoint_graphs"]["central_parameter_displacement_bound"]) < iv.mpf("2.1e-8")
    assert exact(report["endpoint_graphs"]["exterior_amplitude_bound"]) < iv.mpf("1e-8")
    for key, low, high in (("exterior_mass_image", "5.64", "5.66"),
                           ("frequency_bound_on_the_graph", ".139", ".141")):
        interval = report["endpoint_graphs"][key]
        assert exact(interval[0]) > iv.mpf(low) and exact(interval[1]) < iv.mpf(high)
    assert exact(report["nonlinear_interior"]["normalized_scalar_lower_up_to_20"]) > 0
    assert exact(report["nonlinear_interior"]["scalar_ODE_Q_lower_from_20_to_40"]) > 0
    return hashes


def algebra_checks(bg):
    y, yp, ypp, q, qp, a, ap, R = [sp.Matrix(sp.symbols(name+"0:2"))
                                  for name in ("y", "yp", "ypp", "q", "qp", "a", "ap", "R")]
    d, D, Dp, k = sp.symbols("d D Dp k")
    S = sp.diag(*sp.symbols("S0:2"))
    Q0, Q1, Q2 = sp.symbols("Q0:3")
    Q = sp.Matrix([[Q0, Q1], [Q1, Q2]])
    r0 = yp-q-a*D
    r0p = ypp-qp-ap*D-a*Dp
    r1 = qp+d*q-Q*y-S*a*D-k*R
    j = a*Dp+(ap+d*a+S*a)*D+k*R
    error_forcing = ypp+d*yp-Q*y-j
    weak_forcing = (r1-S*r0)+r0p+d*r0+S*r0
    assert all(sp.expand(v) == 0 for v in error_forcing-weak_forcing)
    x, s = sp.symbols("x s", nonnegative=True)
    Lw, rw = x-x*x, x**3+2*x
    H = sp.integrate(Lw.subs(x, s), (s, x, 1))
    ew = -sp.integrate(rw.subs(x, s), (s, 0, x))
    assert sp.integrate(Lw*ew+H*rw, (x, 0, 1)) == 0
    r = np.array([.002, 1., 16., 17.3, 20., 40.])
    raw, damping, _, _, _, C, *_ = coefficients(bg, r)
    differences = []
    for index, radius in enumerate(r):
        value = fields(exact(float(radius)), [exact(float(v)) for v in bg.sol(radius)])
        for key, expected in (("d", damping[index]), ("C", C[index]), ("n", raw["nup"][index]),
                              ("f", raw["f"][index]), ("Q", (raw["Vs"][index]-raw["Omega"][index]**2)/raw["f"][index])):
            difference = abs(float(value[key].mid)-expected)/(1+abs(expected))
            assert difference < 2e-13
            differences.append(float(difference))
    return {"derivative_free_residual_identity": True, "angular_reconstruction_error_pairing": True,
            "field_formula_checks": len(differences),
            "maximum_scaled_formula_difference": max(differences)}


def central_signs(bg):
    A0, sigma = iv.mpf(".06"), iv.mpf(".08")
    r0 = exact(float(bg.sol.x[1]))
    nominal = [iv.mpf(1), exact(float(bg.sol(0)[3]))]
    parameters = [v+enclosure(-iv.mpf("2.1e-8"), iv.mpf("2.1e-8")) for v in nominal]
    baseline = leading(nominal, A0, sigma, iv.pi)
    radius = iv.mpf("1e-6")
    box = [v+enclosure(-radius, radius) for v in baseline]
    s = enclosure(0, r0*r0)
    factors = [iv.mpf(1)/2, iv.mpf(1)/3, iv.mpf(1)/3, iv.mpf(1)/2]
    image = integrands(s, box, parameters, A0, sigma, iv.pi)
    image_radius = max(upper(abs(f*v-b)) for f, v, b in zip(factors, image, baseline))
    derivatives = integrands(s, [Dual(box[j], np.eye(4, dtype=int)[j]) for j in range(4)],
                            parameters, A0, sigma, iv.pi)
    contraction = max(upper(f*sum(map(abs, v.d), iv.mpf(0))) for f, v in zip(factors, derivatives))
    assert exact(image_radius) < radius and exact(contraction) < 1
    p, q, mass, z = box
    a, zeta = parameters[0]+s*p, parameters[1]+s*z
    f, _, Vs, O2, _, pressure = matter(a, q, s, mass, zeta, A0, sigma)
    n_over_r = (mass+4*iv.pi*pressure)/f
    C = 4*s*n_over_r*A0*q+2*A0*a*(2*O2-Vs)/f
    assert lower(a) > 0 and upper(q) < 0 and lower(n_over_r) > 0 and lower(C) > 0
    return {"radius": float(r0.mid), "parameter_radius": "2.1e-8", "volterra_box_radius": "1e-6",
            "image_radius": image_radius, "contraction": contraction,
            "normalized_Aprime_over_r_upper": upper(q), "n_over_r_lower": lower(n_over_r),
            "C_lower": lower(C), "dipole_negative_bound": 0}


def physical_cell(bg, piece, left, right):
    _, y, _ = background_jets(bg, piece, enclosure(left, right), order=1)
    delta = enclosure(-iv.mpf(ERROR), iv.mpf(ERROR))
    return [v.c[0]+delta*exact(float(w)) for v, w in zip(y, WEIGHTS)]


def fields(r, y):
    a, b, u, zeta = y
    A, Ap = iv.mpf(".06")*a, iv.mpf(".06")*b
    f, O2 = 1-2*u, iv.exp(2*zeta)
    ratio = A*A/iv.mpf(".08")**2
    V = A*A*(1-2*ratio)**2
    Vs = (1-2*ratio)*(1-6*ratio)
    rho, pr = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    N = r*u+4*iv.pi*r**3*pr
    n = N/(r*r*f)
    fp = 2*u/r-8*iv.pi*r*rho
    d = 2/r+n+fp/(2*f)
    Q = (Vs-O2)/f
    C = 4*n*Ap+2*A*(2*O2-Vs)/f
    return {"A": A, "Ap": Ap, "f": f, "O2": O2, "Vs": Vs, "N": N, "n": n,
            "d": d, "Q": Q, "C": C}


def finite_region(bg):
    knots = list(map(lambda x: Fraction(float(x)), bg.sol.x))
    edges = sorted({knots[1], Fraction(20), knots[-1]} | {x for x in knots if knots[1] < x < knots[-1]})
    queue = list(zip(edges[:-1], edges[1:]))[::-1]
    checked, refinements = 0, 0
    worst, worst_cell = 0., None
    inner_n, inner_Ap = float("inf"), -float("inf")
    lower_barrier_margin, upper_barrier_margin, outer_bound = float("inf"), float("inf"), 0.
    while queue:
        left, right = queue.pop()
        piece = bisect_right(knots, (left+right)/2)-1
        r = enclosure(left, right)
        g = fields(r, physical_cell(bg, piece, left, right))
        assert lower(g["f"]) > 0 and lower(g["N"]) > 0
        if right <= 20:
            assert lower(g["A"]) > 0 and upper(g["Ap"]) < 0
            inner_n, inner_Ap = min(inner_n, lower(g["n"])), max(inner_Ap, upper(g["Ap"]))
            if lower(g["C"]) >= 0:
                bound = 0.
            else:
                trace = r*r*g["f"]*g["C"]*(4*iv.pi*g["Ap"]/g["n"]+g["n"]/g["Ap"])
                bound = max(0., upper(trace))
        else:
            # The backward Riccati flow leaves [-2,-.9] invariant.
            at_lower = g["Q"]+2*g["d"]-4
            at_upper = g["Q"]+iv.mpf(".9")*g["d"]-iv.mpf(".81")
            assert upper(at_lower) < 0 and lower(at_upper) > 0
            lower_barrier_margin = min(lower_barrier_margin, lower(-at_lower))
            upper_barrier_margin = min(upper_barrier_margin, lower(at_upper))
            t = enclosure(-2, -iv.mpf(".9"))
            d2, d4 = [4*iv.pi*r**j*g["A"]**2 for j in (2, 4)]
            trace = exterior_negative_trace(1/r, g["f"], g["N"], d2, d4, t, g["O2"], g["Vs"])
            bound = max(0., upper(trace))
            outer_bound = max(outer_bound, bound)
        if bound >= 52:
            if checked+len(queue)+3 > 4000 or refinements >= 500:
                raise RuntimeError("The finite coercivity enclosure exceeds its bounded subdivision budget.")
            middle = (left+right)/2
            queue.extend(((middle, right), (left, middle)))
            refinements += 1
            continue
        checked += 1
        if bound > worst:
            worst, worst_cell = bound, [float(left), float(right)]
    assert checked+1 <= 4000 and exact(worst) < 52 and exact(outer_bound) < 32
    return {"partition_points": checked+1, "refinements": refinements,
            "maximum_negative_bound": worst, "maximum_cell": worst_cell,
            "n_lower_to_20": inner_n, "Aprime_upper_to_20": inner_Ap,
            "outer_slope_interval": [-2, -.9], "outer_range": [20, 40],
            "lower_Riccati_barrier_margin": lower_barrier_margin,
            "upper_Riccati_barrier_margin": upper_barrier_margin,
            "outer_negative_bound": outer_bound}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 25 <= args.digits <= 80:
        parser.error("Use 25 through 80 interval digits.")
    iv.dps = args.digits
    start = perf_counter()
    hashes = check_inputs()
    bg = Background()
    algebra = algebra_checks(bg)
    center = central_signs(bg)
    interior = finite_region(bg)
    tail = certify_tail(args.digits)
    report = {"status": "global_polar_coercivity_verified",
              "scope": "Exact original star; all even l>=8. Low polar degrees and feedback errors remain unverified.",
              "input_sha256": hashes, "background_error": ERROR, "digits": args.digits, "algebra": algebra,
              "center": center, "finite_region": interior, "tail": tail,
              "verified_global_negative_bound": "52", "coercivity": "l*(l+1)-54",
              "first_even_degree": 8, "elapsed_seconds": perf_counter()-start}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
