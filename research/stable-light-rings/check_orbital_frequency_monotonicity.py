"""Certify decreasing radial null-orbit frequency on the source band.

Schaaf's period criterion is checked with the existing exact-star enclosure.
Radial derivatives come from the nonlinear ODE, not derivatives of its
piecewise-polynomial approximation. No new background or wave BVP is solved.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import json
from pathlib import Path
import time

import sympy as sp

from check_axial_constraint_lift import CHECKS, zero


ROOT = Path(__file__).parent


def symbolic_checks():
    r = sp.Symbol("r", positive=True)
    p, u = sp.Function("p")(r), sp.Function("U")(r)
    logarithm = sp.diff(p, r)/p
    g = sp.diff(u, r, 2)+logarithm*sp.diff(u, r)
    h = (sp.diff(u, r, 3)+3*logarithm*sp.diff(u, r, 2)
         +(2*logarithm**2+sp.diff(logarithm, r))*sp.diff(u, r))
    fourth = (sp.diff(u, r, 4)+6*logarithm*sp.diff(u, r, 3)
              +(11*logarithm**2+4*sp.diff(logarithm, r))*sp.diff(u, r, 2)
              +(6*logarithm**3+7*logarithm*sp.diff(logarithm, r)
                +sp.diff(logarithm, r, 2))*sp.diff(u, r))
    actual = p*sp.diff(u, r)
    for n, expected in ((2, p**2*g), (3, p**3*h), (4, p**4*fourth)):
        actual = p*sp.diff(actual, r)
        zero(f"tortoise derivative {n}", sp.simplify(actual-expected))
    energy = sp.Symbol("E", positive=True)
    z = sp.Function("Z")(energy)
    wr = sp.pi/(sp.sqrt(energy)*z)
    zero("radial frequency slope",
         sp.diff(wr, energy)/wr+1/(2*energy)+sp.diff(z, energy)/z)
    m = sp.Symbol("m", positive=True)
    omega = sp.Symbol("omega", positive=True)
    frequency = sp.Function("Omega")(energy)
    derivative = -(omega-m*sp.sqrt(energy))*sp.diff(frequency, energy)/frequency**2
    center = omega**2/m**2
    zero("resonance fold stationary point", derivative.subs(energy, center))
    zero("resonance fold curvature", sp.diff(derivative, energy).subs(energy, center)
         -m*sp.diff(frequency, energy).subs(energy, center)
         /(2*sp.sqrt(center)*frequency.subs(energy, center)**2))


def certify(digits):
    from mpmath import iv
    from check_asymptotic_exterior import lower
    from check_background_defect import field_equations
    from check_barrier_certificate import Jet, exact, enclosure
    from check_finite_static_inverse import upper
    from check_polar_coercivity import check_inputs, physical_cell, ERROR
    from trapped_axial import Background

    iv.dps = digits
    hashes = check_inputs()
    background = Background()
    global_report = json.loads((ROOT/"data/s08a06/global-background.json").read_text())
    knots = [Fraction(float(x)) for x in background.sol.x]
    left, right = Fraction("15.62"), Fraction("16.521")

    def partition(a, b):
        return sorted({a, b} | {x for x in knots if a < x < b})

    def derivative_data(a, b):
        radius = enclosure(a, b)
        piece = bisect_right(knots, (a+b)/2)-1
        initial = physical_cell(background, piece, a, b)
        radial = Jet(0, [radius, exact(1), exact(0), exact(0), exact(0)])
        state = [Jet(0, [v]+[exact(0)]*4) for v in initial]
        for k in range(4):
            rhs = field_equations(radial, state, iv.mpf(".06"), iv.mpf(".08"), iv.pi)
            for y, value in zip(state, rhs):
                y.c[k+1] = value.c[k]/(k+1)
        compact, zeta = state[2].c, state[3].c
        f = 1-2*compact[0]
        if lower(f) <= 0:
            raise ValueError("The exact-star cell is not horizonless.")
        l1 = -2*zeta[1]-2/radius
        l2 = -4*zeta[2]+2/radius**2
        l3 = -12*zeta[3]-4/radius**3
        l4 = -48*zeta[4]+12/radius**4
        lp = -zeta[1]-compact[1]/f
        lp1 = -2*zeta[2]-2*compact[2]/f-2*compact[1]**2/f**2
        lp2 = (-6*zeta[3]-6*compact[3]/f-12*compact[1]*compact[2]/f**2
               -8*compact[1]**3/f**3)
        u2 = l1*l1+l2
        u3 = l1**3+3*l1*l2+l3
        u4 = l1**4+6*l1*l1*l2+3*l2*l2+4*l1*l3+l4
        g = u2+lp*l1
        h = u3+3*lp*u2+(2*lp*lp+lp1)*l1
        fourth = (u4+6*lp*u3+(11*lp*lp+4*lp1)*u2
                  +(6*lp**3+7*lp*lp1+lp2)*l1)
        return {"g": g, "h": h, "test": 5*h*h-3*g*fourth, "l1": l1,
                "scaled_optical": iv.exp(-2*zeta[0])/radius**2, "f": f}

    edges = partition(left, right)
    queue = list(zip(edges[:-1], edges[1:]))[::-1]
    checked = refinements = mixed = 0
    positive_margin = float("inf")
    zero_margin = float("inf")
    increasing_branch_margin = float("inf")
    horizon_margin = float("inf")
    mixed_cells = []
    while queue:
        a, b = queue.pop()
        data = derivative_data(a, b)
        curvature = data["g"]
        needs_positive_test = upper(curvature) >= 0
        could_vanish = lower(curvature) <= 0 <= upper(curvature)
        good = not needs_positive_test or lower(data["test"]) > 0
        if could_vanish:
            good = good and upper(data["l1"]*data["h"]) < 0
        # A critical point is a minimum; the right concave branch has U'>0.
        if lower(curvature) <= 0:
            good = good and lower(data["l1"]) > 0
        if not good:
            if checked+len(queue)+3 > 4000 or refinements >= 2000:
                raise RuntimeError("Stopped within the 4000-point monotonicity budget.")
            middle = (a+b)/2
            queue.extend(((middle, b), (a, middle)))
            refinements += 1
            continue
        checked += 1
        horizon_margin = min(horizon_margin, lower(data["f"]))
        if needs_positive_test:
            positive_margin = min(positive_margin, lower(data["test"]))
        if could_vanish:
            mixed += 1
            zero_margin = min(zero_margin, lower(-data["l1"]*data["h"]))
            if mixed_cells and mixed_cells[-1][1] == float(a):
                mixed_cells[-1][1] = float(b)
            else:
                mixed_cells.append([float(a), float(b)])
        if lower(curvature) <= 0:
            increasing_branch_margin = min(increasing_branch_margin, lower(data["l1"]))

    if upper(derivative_data(left, left)["l1"]) >= 0:
        raise ValueError("The left endpoint does not precede the minimum.")
    if lower(derivative_data(right, right)["l1"]) <= 0:
        raise ValueError("The right endpoint is not in the increasing branch.")

    def stationary_value(ring):
        a, b = map(Fraction, ring["radius_interval"])
        center = (a+b)/2
        point = derivative_data(center, center)["scaled_optical"]
        parts = partition(a, b)
        slope = max(upper(abs(derivative_data(x, y)["l1"]))
                    for x, y in zip(parts[:-1], parts[1:]))
        exponent = exact((b-a)/2)*exact(slope)
        return point*iv.exp(enclosure(-exponent, exponent))

    stable, unstable = global_report["light_rings"]
    us, uu = stationary_value(stable), stationary_value(unstable)
    energy_upper = iv.mpf(".001")*us+iv.mpf(".999")*uu
    margins = [lower(derivative_data(r, r)["scaled_optical"]-energy_upper) for r in (left, right)]
    if min(margins) <= 0 or lower(uu-us) <= 0:
        raise ValueError("The verified radial interval does not contain the source-band orbits.")
    return {"status": "exact_star_orbital_frequency_strictly_decreasing",
            "digits": digits, "input_sha256": hashes, "background_error": ERROR,
            "radial_interval": [str(left), str(right)],
            "source_energy_fraction_upper": ".999",
            "partition_points": checked+1, "refinements": refinements,
            "curvature_zero_candidate_cells_count": mixed,
            "curvature_zero_candidate_ranges": mixed_cells,
            "normalized_Schaaf_test_lower_bound": positive_margin,
            "normalized_negative_Uprime_Uthird_lower_bound_at_zero_cells": zero_margin if mixed else None,
            "log_optical_slope_lower_on_nonconvex_cells": increasing_branch_margin,
            "metric_factor_lower_bound": horizon_margin,
            "scaled_endpoint_energy_margins": margins,
            "scope": "Schaaf criterion and band coverage on the certified exact star; no wave pole or leakage sign"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--symbolic-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 25 <= args.digits <= 60:
        parser.error("Use 25 through 60 interval digits.")
    start = time.perf_counter()
    symbolic_checks()
    result = {"exact_checks": len(CHECKS)}
    if not args.symbolic_only:
        result["certificate"] = certify(args.digits)
    result["seconds"] = round(time.perf_counter()-start, 3)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="ascii")
    summary = {key: value for key, value in result.items() if key != "certificate"}
    if "certificate" in result:
        summary.update({key: value for key, value in result["certificate"].items()
                        if key not in ("input_sha256", "curvature_zero_candidate_ranges")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
