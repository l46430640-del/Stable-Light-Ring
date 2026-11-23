"""Certify the near-top positive orbit source on the exact original star.

Positive Bernstein coefficients enclose the energy integrals. Continuous
radial cell bounds include the background error and the moving light-ring
energies; no collocation solution or sampled positivity is used as proof.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Run source certificates without python -O.")

import numpy as np
import sympy as sp
from mpmath import iv, mp

from check_asymptotic_exterior import lower
from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import upper
from check_near_top_orientation import angle_normalized_source
from check_physical_barrier import physical_box, optical_quantities, STABLE, UNSTABLE
from check_polar_coercivity import check_inputs, physical_cell
from polar_band_moments import band_moments, maximum_fraction
from trapped_axial import Background

ROOT = Path(__file__).parent
DATA = ROOT/"data/solitonic-screen"
DOMAIN = (Fraction("15.62"), Fraction("16.525"))


def endpoints(value):
    return [lower(value), upper(value)]


def algebra_checks():
    t, s, x = sp.symbols("t s x", real=True)
    B = lambda n, j: sp.binomial(n, j)*t**j*(1-t)**(n-j)
    for m, n in ((2, 2), (8, 8), (16, 2)):
        for i in range(m+1):
            for j in range(n+1):
                factor = sp.binomial(m, i)*sp.binomial(n, j)/sp.binomial(m+n, i+j)
                assert sp.expand(B(m, i)*B(n, j)-factor*B(m+n, i+j)) == 0
    for n in (14, 16, 18):
        assert all(sp.integrate(B(n, j), (t, 0, 1)) == sp.Rational(1, n+1) for j in range(n+1))
    shape = (4*(x+s*s)*(1-x-s*s))**4
    assert sp.expand(sp.diff(s*shape, s)-shape-2*s*s*sp.diff(shape, x)) == 0
    P = 256*sum(sp.binomial(4, k)*x**(4-k)*(1-x)**k*sp.factorial(4)
                /sp.prod(sp.Rational(1, 2)+k+n for n in range(5)) for k in range(5))
    Q = 1792*x**4-448*x**3-48*x*x-8*x-1
    assert sp.expand((1-x)*sp.diff(P, x)-sp.Rational(9, 2)*P+sp.Rational(32768, 45045)*Q) == 0
    assert sp.polys.polytools.count_roots(Q, 0, 1) == 1
    r, n, rho, pressure = sp.symbols("r n rho pressure", nonzero=True)
    pressure_prime = (-2/r-2*n)*pressure-(rho-pressure)*(n-1/r)
    assert sp.expand(pressure_prime+n*(rho+pressure)+(3*pressure-rho)/r) == 0
    omega, zeta, N, J0, J1, depth = sp.symbols("omega zeta N J0 J1 depth", nonzero=True)
    alpha, W = omega*sp.exp(-zeta), sp.exp(-2*zeta)/r**2
    rho = N*omega**2*(W*J0+depth*J1)/(4*sp.pi*r*r*alpha**2)
    pressure = N*sp.exp(2*zeta)*depth*J1/(4*sp.pi*r*r)
    assert sp.simplify(rho-pressure-N*J0/(4*sp.pi*r**4)) == 0
    cases = []
    for point in (sp.Rational(-9, 16), sp.Rational(0), sp.Rational(3, 4), sp.Rational(1)):
        left, right = sp.sqrt(max(-point, 0)), sp.sqrt(max(1-point, 0))
        expressions = [2*sp.integrate((shape*s**(2*j)).subs(x, point), (s, left, right)) for j in (0, 1)]
        expressions.append(2*sp.integrate(sp.diff(shape, x).subs(x, point), (s, left, right)))
        values = band_moments(exact(Fraction(point)), 0, 1, derivative=True)
        for expected, interval in zip(expressions, values):
            rational = exact(Fraction(expected))
            assert interval.a <= rational.a and rational.b <= interval.b
        cases.append(str(point))
    # Independent high-precision integration across both branch boundaries.
    mp.dps = 80
    a, b = mp.mpf(".998"), mp.mpf(".999")
    largest, enclosure_error = 0., 0.
    for text in ("-.1", "0", ".9979", ".998", ".9982", ".9989", ".999"):
        xx = mp.mpf(text)
        l, r = mp.sqrt(max(a-xx, 0)), mp.sqrt(max(b-xx, 0))
        def integrand(ss, j):
            fraction = (xx+ss*ss-a)/(b-a)
            return 2*(4*fraction*(1-fraction))**4*ss**(2*j)
        values = band_moments(iv.mpf(text))
        for j, interval in enumerate(values):
            reference = mp.quad(lambda ss: integrand(ss, j), [l, r])
            reference_box = iv.mpf(mp.nstr(reference, 75))
            assert interval.a <= reference_box.b and reference_box.a <= interval.b
            midpoint = mp.mpf(float(interval.mid))
            largest = max(largest, float(abs(midpoint-reference)))
            uncertainty = mp.mpf(str(upper(abs(interval-iv.mpf(mp.nstr(reference, 75))))))
            enclosure_error = max(enclosure_error, float(uncertainty))
            assert uncertainty < mp.mpf("2e-14"), (text, j, interval, uncertainty)
    crossing = 0
    for center in (".998", ".999"):
        lo, hi = iv.mpf(center)-iv.mpf("1e-8"), iv.mpf(center)+iv.mpf("1e-8")
        box = enclosure(lo.a, hi.b)
        bounds = band_moments(box, derivative=True)
        for point in (box.a, box.mid, box.b):
            for value, bound in zip(band_moments(point, derivative=True), bounds):
                assert bound.a <= value.a and value.b <= bound.b
                crossing += 1
    return {"Bernstein_product_identity": True, "Bernstein_integrals": True,
            "moment_conservation_identity": True, "exact_rational_cases": cases,
            "radial_stress_conservation": True, "source_frequency_cancellation": True,
            "unique_moment_maximum": list(map(str, maximum_fraction())),
            "independent_moment_comparisons": 14, "maximum_midpoint_difference": largest,
            "maximum_independent_comparison_radius": enclosure_error,
            "crossing_containment_checks": crossing}


def log_potential_cell(bg, knots, left, right):
    middle = (left+right)/2
    piece = min(bisect_right(knots, middle)-1, len(knots)-2)
    assert knots[piece] <= left <= right <= knots[piece+1]
    y = physical_cell(bg, piece, left, right)
    ym = physical_cell(bg, piece, middle, middle)
    r = enclosure(left, right)
    H, _, f = optical_quantities(r, y, iv.mpf(".06"), iv.mpf(".08"))
    log_mid = -2*ym[3]-2*iv.ln(exact(middle))
    return log_mid+enclosure(-(right-left)/2, (right-left)/2)*2*H/(r*f), y


def ring_potential(bg, location, bracket):
    a, b = map(Fraction, bracket)
    r0 = Fraction(float(location))
    assert a < r0 < b
    y = physical_box(bg, a, b)
    H, _, f = optical_quantities(enclosure(a, b), y, iv.mpf(".06"), iv.mpf(".08"))
    variation = exact(max(r0-a, b-r0))*abs(2*H/(enclosure(a, b)*f))
    y0 = physical_box(bg, r0, r0)
    log_value = -2*y0[3]-2*iv.ln(exact(r0))+enclosure(-variation, variation)
    assert variation.b < iv.mpf("1e-10").a
    return iv.exp(log_value), upper(variation)


def source_partition(bg, source, count):
    nodes = {DOMAIN[0], DOMAIN[1]}
    edges = [float(DOMAIN[0]), *source.knots, float(DOMAIN[1])]
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        points = count if index == 2 else 32
        mesh = a+(b-a)*np.sin(np.linspace(0, np.pi/2, points+1))**2
        nodes.update(Fraction(float(x)) for x in mesh if DOMAIN[0] < Fraction(float(x)) < DOMAIN[1])
    nodes.update(Fraction(float(x)) for x in bg.sol.x if DOMAIN[0] < Fraction(float(x)) < DOMAIN[1])
    if len(nodes) > 4000:
        raise RuntimeError("Source normalization exceeds 4000 radial partition points.")
    return sorted(nodes)


def numerical_source_check(bg, source):
    radii = np.unique(np.r_[np.linspace(source.a+1e-8, source.b-1e-8, 61), source.knots[1:3]])
    g = bg.fields(radii)
    rho, pressure, D, _ = source.fields(radii)
    reference = [4*radii*radii*D/source.factor,
                 4*np.pi*radii*radii*g["alpha"]**2*pressure/(source.factor*source.depth)]
    maximum = 0.
    for i, r in enumerate(radii):
        x = float((g["U"][i]-bg.Us)/source.depth)
        exact_values = band_moments(exact(x), source.floor, source.ceiling)
        for j, value in enumerate(exact_values):
            difference = abs(float(value.mid)-reference[j][i])
            maximum = max(maximum, difference)
            assert difference < 2e-12
    return {"compared_radii": len(radii), "maximum_moment_difference": maximum,
            "scope": "Floating consistency with the existing twelve-point energy quadrature."}


def certificate(bg, count):
    hashes = check_inputs()
    global_record = json.loads((ROOT/"data/s08a06/global-background.json").read_text())
    assert [row["radius_interval"] for row in global_record["light_rings"]] == [list(STABLE), list(UNSTABLE)]
    omega = enclosure(*global_record["endpoint_graphs"]["frequency_bound_on_the_graph"])
    Ws, stable_shift = ring_potential(bg, bg.rs, STABLE)
    Wu, unstable_shift = ring_potential(bg, bg.ru, UNSTABLE)
    depth = Wu-Ws
    assert depth.a > 0
    source = angle_normalized_source(bg, (.998, .999))
    cache = DATA/"polar-residual-fields/degree-002-tol-2e-07.npz"
    with np.load(cache) as data:
        reference_factor = float(data["source_factor"])
    assert reference_factor == source.factor
    hashes["reference_response.npz"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    nodes = source_partition(bg, source, count)
    knots = [Fraction(float(x)) for x in bg.sol.x]
    margins = []
    for r in DOMAIN:
        logW, _ = log_potential_cell(bg, knots, r, r)
        margins.append((iv.exp(logW)-Ws)/depth-iv.mpf(".999"))
    assert all(v.a > 0 for v in margins)
    total, rows = iv.mpf(0), []
    for a, b in zip(nodes[:-1], nodes[1:]):
        logW, y = log_potential_cell(bg, knots, a, b)
        W, f = iv.exp(logW), 1-2*y[2]
        x = (W-Ws)/depth
        J0, J1 = band_moments(x)
        integrand = iv.exp(y[3])*(W*J0+depth*J1)/iv.sqrt(f)
        contribution = exact(b-a)*integrand
        total += contribution
        r = enclosure(a, b)
        pressure = iv.exp(2*y[3])*depth*J1/(4*iv.pi*r*r)
        density = J0/(4*iv.pi*r**4)+pressure
        rows.append({"interval": [float(a), float(b)], "potential_fraction": endpoints(x),
                     "interval_exact": [str(a), str(b)],
                     "J0": endpoints(J0), "J1": endpoints(J1),
                     "density_per_amplitude": endpoints(density),
                     "pressure_per_amplitude": endpoints(pressure),
                     "anisotropy_per_amplitude": endpoints(J0/(4*r*r)),
                     "reduced_energy_integrand": endpoints(integrand)})
    energy = omega*total
    assert energy.a > 0
    factor = 1/energy
    reference_ratio = exact(reference_factor)/factor
    assert reference_ratio.a < 1 < reference_ratio.b
    return {"status": "positive_band_source_and_normalization_enclosed", "input_sha256": hashes,
            "scope": "Exact positive conserved orbit mixture on the original star; normalization and continuous source moments enclosed, response sign not certified.",
            "band": [".998", ".999"], "radial_domain": list(map(str, DOMAIN)),
            "source_zero_boundary_margins": [endpoints(v) for v in margins],
            "stable_reduced_potential": endpoints(Ws), "unstable_reduced_potential": endpoints(Wu),
            "reduced_well_depth": endpoints(depth), "frequency": endpoints(omega),
            "stationary_log_shift_bounds": [stable_shift, unstable_shift],
            "partition_points": len(nodes), "requested_middle_cells": count,
            "unnormalized_Killing_energy": endpoints(energy), "normalization_factor": endpoints(factor),
            "reference_normalization_factor": reference_factor,
            "reference_factor_exact_rational": str(Fraction(reference_factor)),
            "reference_to_exact_factor": endpoints(reference_ratio),
            "existing_source_comparison": numerical_source_check(bg, source),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, choices=(30, 50), default=30)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 32 <= args.cells <= 2500:
        parser.error("Use 32 through 2500 middle cells within the total partition budget.")
    iv.dps = args.digits
    started = perf_counter()
    checks = algebra_checks()
    report = certificate(Background(), args.cells)
    report.update(arithmetic_digits=args.digits, algebra_checks=checks, elapsed_seconds=perf_counter()-started)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
