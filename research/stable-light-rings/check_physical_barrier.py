"""Enclose the trapping geometry and aligned coefficients of the exact star.

The global-background certificate supplies the stationary solution and its
uniform error. This check does not yet bound its fixed-charge response.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from mpmath import iv

from check_asymptotic_exterior import lower
from check_background_defect import background_jets, field_equations
from check_barrier_certificate import Jet, exact, enclosure, matrix_formula
from check_finite_static_inverse import upper, interval_norm
from finite_static_inverse import WEIGHTS
from trapped_axial import Background


BACKGROUND_ERROR = "7.2e-9"
STABLE = ("15.942938", "15.942947")
UNSTABLE = ("16.527600", "16.527617")
OUTER = ("16.7977", "16.7979")


def represented_box(background, left, right):
    knots = [Fraction(float(x)) for x in background.sol.x]
    assert knots[0] <= left <= right <= knots[-1]
    cuts = sorted({left, right} | {x for x in knots if left < x < right})
    pairs = list(zip(cuts[:-1], cuts[1:])) if left < right else [(left, right)]
    boxes = []
    for a, b in pairs:
        piece = min(bisect_right(knots, (a+b)/2)-1, len(knots)-2)
        _, y, _ = background_jets(background, piece, enclosure(a, b), order=1)
        boxes.append([v.c[0] for v in y])
    if left < right and right in knots[:-1]:
        _, y, _ = background_jets(background, bisect_right(knots, right)-1, exact(right), order=1)
        boxes.append([v.c[0] for v in y])
    return [iv.mpf([min(v[j].a for v in boxes), max(v[j].b for v in boxes)]) for j in range(4)]


def physical_box(background, left, right):
    rho = iv.mpf(BACKGROUND_ERROR)
    return [v+enclosure(-rho, rho)*exact(float(w))
            for v, w in zip(represented_box(background, left, right), WEIGHTS)]


def optical_quantities(r, y, A0, scale):
    A0, scale, pi = [Jet(v) for v in (A0, scale, iv.pi)] if isinstance(r, Jet) else (A0, scale, iv.pi)
    a, b, u, zeta = y
    f, scalar2 = 1-2*u, A0*A0*a*a
    ratio = scalar2/(scale*scale)
    potential = scalar2*(1-2*ratio)*(1-2*ratio)
    omega2 = (2*zeta).exp() if isinstance(zeta, Jet) else iv.exp(2*zeta)
    pressure = (omega2*scalar2+f*A0*A0*b*b-potential)/2
    H = 3*u+4*pi*r*r*pressure-1
    root_derivative = 8*pi*r*r*omega2*scalar2-1
    return H, root_derivative, f


def optical_H_derivative(r, box, A0, scale):
    field = field_equations(Jet(r), [Jet(v) for v in box], A0, scale, iv.pi)
    jets = [Jet(0, [v, F.c[0], iv.mpf(0), iv.mpf(0)]) for v, F in zip(box, field)]
    H, _, _ = optical_quantities(Jet(0, [r, iv.mpf(1), iv.mpf(0), iv.mpf(0)]), jets, A0, scale)
    return H.c[1]


def partition(background, left, right, extra=()):
    result = sorted({left, right, *extra} |
                    {Fraction(float(x)) for x in background.sol.x if left < Fraction(float(x)) < right})
    assert result[0] == left and result[-1] == right
    if len(result) > 4000:
        raise RuntimeError("Barrier verification exceeds 4000 radial points.")
    return result


def certify_geometry(background):
    A0, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    edges = list(map(Fraction, ("15.93", "15.96", "16.51", "16.55", "16.803")))
    nodes = partition(background, edges[0], edges[-1], edges[1:-1])
    minimum = [float("inf")]*4
    for a, b in zip(nodes[:-1], nodes[1:]):
        box = physical_box(background, a, b)
        H, root_derivative, _ = optical_quantities(enclosure(a, b), box, A0, scale)
        zone = bisect_right(edges, (a+b)/2)-1
        if zone in (1, 3):
            center = (a+b)/2
            midpoint_H = optical_quantities(exact(center), physical_box(background, center, center), A0, scale)[0]
            derivative = optical_H_derivative(enclosure(a, b), box, A0, scale)
            H = midpoint_H+enclosure(-(b-a)/2, (b-a)/2)*derivative
        quantity = (root_derivative, H, -root_derivative, -H)[zone]
        assert quantity.a > 0, f"Optical monotonicity not resolved on {(float(a), float(b))}."
        minimum[zone] = min(minimum[zone], lower(quantity))
    assert all(exact(v) > iv.mpf(bound) for v, bound in zip(minimum, ("0.19", "0.0002", "0.23", "0.0003")))

    # Integrate the exact logarithmic derivative near the minimum, retaining
    # its vanishing slope instead of subtracting independent endpoint boxes.
    rs_left, rs_right = map(Fraction, STABLE)
    reference_rs = Fraction(float(background.rs))
    assert rs_left < reference_rs < rs_right
    displacement = max(abs(reference_rs-rs_left), abs(reference_rs-rs_right))
    H, _, f = optical_quantities(enclosure(rs_left, rs_right), physical_box(background, rs_left, rs_right), A0, scale)
    log_shift = exact(displacement)*abs(2*H/(enclosure(rs_left, rs_right)*f))
    assert log_shift < iv.mpf("5e-12")
    y0 = physical_box(background, reference_rs, reference_rs)
    log_Ws = -2*y0[3]-2*iv.ln(exact(reference_rs))+enclosure(-log_shift, log_shift)
    gaps = []
    for endpoint in OUTER:
        r = Fraction(endpoint)
        y = physical_box(background, r, r)
        gaps.append(iv.exp(-2*y[3]-2*iv.ln(exact(r))-log_Ws)-1)
    assert gaps[0].a > 0 and gaps[1].b < 0
    assert gaps[0].a > iv.mpf("1.19e-7") and gaps[0].b < iv.mpf("1.49e-7")
    assert gaps[1].a > iv.mpf("-1.87e-7") and gaps[1].b < iv.mpf("-1.57e-7")
    width = Fraction(OUTER[1])-rs_left
    assert width < Fraction("0.855") < Fraction("0.86")
    return {"partition_points": len(nodes), "critical_neighborhoods": [["15.93", "15.96"], ["16.51", "16.55"]],
            "positive_bounds_by_zone": minimum,
            "stable_radius_interval": list(STABLE), "unstable_radius_interval": list(UNSTABLE),
            "outer_turning_interval": list(OUTER), "barrier_width_upper": "0.855",
            "light_ring_displacement_bound": upper(exact(displacement)),
            "log_potential_shift_bound": upper(log_shift),
            "relative_outer_endpoint_gaps": [[lower(v), upper(v)] for v in gaps]}, displacement


def derivative_bounds(r, box, A0, scale):
    weights = [exact(float(w)) for w in WEIGHTS]
    state_jets = [Jet(0, [v, enclosure(-w, w), iv.mpf(0), iv.mpf(0)]) for v, w in zip(box, weights)]
    state_matrix = matrix_formula(Jet(r), state_jets, A0, scale)
    state_bound = interval_norm([[v.c[1]*weights[j]/weights[i] for j, v in enumerate(row)]
                                 for i, row in enumerate(state_matrix)])
    field = field_equations(Jet(r), [Jet(v) for v in box], A0, scale, iv.pi)
    flow_jets = [Jet(0, [v, F.c[0], iv.mpf(0), iv.mpf(0)]) for v, F in zip(box, field)]
    flow_matrix = matrix_formula(Jet(0, [r, iv.mpf(1), iv.mpf(0), iv.mpf(0)]), flow_jets, A0, scale)
    flow_bound = interval_norm([[v.c[1]*weights[j]/weights[i] for j, v in enumerate(row)]
                                for i, row in enumerate(flow_matrix)])
    return state_bound, flow_bound


def certify_alignment(background, displacement):
    start = Fraction(float(background.rs))
    nodes = partition(background, start, start+Fraction("0.86"))
    A0, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    rho, shift = iv.mpf(BACKGROUND_ERROR), exact(displacement)
    state_max, flow_max, difference = 0., 0., 0.
    for a, b in zip(nodes[:-1], nodes[1:]):
        left, right = a-displacement, b+displacement
        box = physical_box(background, left, right)
        state, flow = derivative_bounds(enclosure(left, right), box, A0, scale)
        state_max, flow_max = max(state_max, state), max(flow_max, flow)
        difference = max(difference, upper(rho*exact(state)+shift*exact(flow)))
    conversion = max(upper(exact(float(w))/iv.mpf(s)) for w, s in zip(WEIGHTS, ("1", "2", "0.4", "0.5")))
    difference = upper(exact(difference)*exact(conversion))
    assert exact(state_max)*exact(conversion) < iv.mpf("70.37")
    assert exact(flow_max)*exact(conversion) < iv.mpf("5.803")
    assert exact(difference) < iv.mpf("2.7e-5")
    return {"partition_points": len(nodes), "interval_length": "0.86",
            "weighted_state_derivative_bound": state_max, "weighted_flow_derivative_bound": flow_max,
            "aligned_coefficient_difference_bound": difference,
            "verified_coefficient_difference_upper": "0.000027"}


def certify_jump(background, frequency_interval, reference):
    left, right = map(Fraction, STABLE)
    r = enclosure(left, right)
    a, b, u, zeta = physical_box(background, left, right)
    omega = enclosure(*map(float, frequency_interval))
    factor = iv.sqrt(1-2*u)*iv.exp(zeta)/(r*omega)
    physical = [iv.mpf(0), factor*b/(1-2*u), factor, iv.mpf(0)]
    with np.load(reference) as data:
        mesh, rhs = data["mesh"], data["rhs"]
        i = int(np.flatnonzero(mesh == background.rs)[0])
        nominal = rhs[4*(i-1):4*i]
    errors = [upper(abs(v/exact(float(w))-exact(float(n)))) for v, w, n in zip(physical, WEIGHTS, nominal)]
    conversion = max(upper(exact(float(w))/iv.mpf(s)) for w, s in zip(WEIGHTS, ("1", "2", "0.4", "0.5")))
    error = upper(exact(max(errors))*exact(conversion))
    assert exact(error) < iv.mpf("1.6e-7")
    return {"scaled_jump_error_components": errors, "decimal_weight_jump_error_bound": error,
            "verified_jump_error_upper": "0.00000016",
            "scope": "Jump amplitudes at their respective radii; source displacement still needs Green propagation."}


def check_formulas(background):
    discontinuous = SimpleNamespace(sol=SimpleNamespace(x=np.array([0., 1., 2.]),
        c=np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])))
    box = represented_box(discontinuous, Fraction(1, 2), Fraction(1))
    assert all(v.a <= j+1 <= j+5 <= v.b for j, v in enumerate(box))
    point = represented_box(discontinuous, Fraction(1), Fraction(1))
    assert all(v.a <= j+5 <= v.b for j, v in enumerate(point))
    direction = np.array([.2, -.3, .1, -.2])
    A0, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    for r in (15.94, 16.2, 16.79):
        y = background.sol(r)
        F = field_equations(Jet(r), [Jet(float(v)) for v in y], A0, scale, iv.pi)
        field = np.array([float(v.c[0].mid) for v in F])
        H, _, f = optical_quantities(exact(r), list(map(exact, map(float, y))), A0, scale)
        identity = -2*F[3].c[0]-2/exact(r)-2*H/(exact(r)*f)
        assert identity.a <= 0 <= identity.b
        derivative_H = optical_H_derivative(exact(r), list(map(exact, map(float, y))), A0, scale)
        h = 1e-5
        endpoints_H = [optical_quantities(exact(r+sign*h), list(map(exact, map(float, y+sign*h*field))), A0, scale)[0]
                       for sign in (-1, 1)]
        expected_H = float(((endpoints_H[1]-endpoints_H[0])/(2*h)).mid)
        assert abs(float(derivative_H.mid)-expected_H) < 1e-8
        for radial, d in ((0, direction), (1, field)):
            jets = [Jet(0, [exact(float(v)), exact(float(dj)), iv.mpf(0), iv.mpf(0)]) for v, dj in zip(y, d)]
            matrix = matrix_formula(Jet(0, [exact(r), iv.mpf(radial), iv.mpf(0), iv.mpf(0)]), jets, A0, scale)
            actual = np.array([[float(v.c[1].mid) for v in row] for row in matrix])
            endpoints = []
            h = 1e-5
            for sign in (-1, 1):
                values = matrix_formula(Jet(r+sign*h*radial), [Jet(float(v)) for v in y+sign*h*d], A0, scale)
                endpoints.append(np.array([[float(v.c[0].mid) for v in row] for row in values]))
            expected = (endpoints[1]-endpoints[0])/(2*h)
            assert np.max(np.abs(actual-expected)/(1+np.abs(actual))) < 2e-7
    print("PASS: optical logarithmic derivative and matrix derivatives", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, choices=(30, 50), default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    iv.dps = args.digits
    start, background = perf_counter(), Background()
    directory = Path(__file__).parent/"data/s08a06"
    global_path = directory/"global-background.json"
    certificate = json.loads(global_path.read_text())
    reference = directory/"finite-static-reference.npz"
    hashes = {"background_sha256": hashlib.sha256((directory/"background.npz").read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    assert all(certificate[key] == value for key, value in hashes.items())
    assert certificate["status"] == "nonlinear_global_background_validated"
    assert exact(certificate["fixed_point"]["fixed_point_error_bound"]) < iv.mpf(BACKGROUND_ERROR)
    assert [v["radius_interval"] for v in certificate["light_rings"]] == [list(STABLE), list(UNSTABLE)]
    check_formulas(background)
    geometry, displacement = certify_geometry(background)
    alignment = certify_alignment(background, displacement)
    jump = certify_jump(background, certificate["endpoint_graphs"]["frequency_bound_on_the_graph"], reference)
    result = {"status": "physical_barrier_geometry_and_coefficients_validated",
              "scope": "Exact background geometry and coefficient alignment; the fixed-charge response interface error remains open.",
              "background_error_used": BACKGROUND_ERROR, "geometry": geometry, "coefficient_alignment": alignment,
              "concentrated_jump": jump, "arithmetic_digits": args.digits, **hashes,
              "global_background_certificate_sha256": hashlib.sha256(global_path.read_bytes()).hexdigest(),
              "elapsed_seconds": perf_counter()-start}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
