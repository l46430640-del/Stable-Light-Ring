"""Verify nonlinear matching of a regular center to an AF boson-star exterior.

The scalar theory is fixed. The ADM mass is that of the exact nominal
exterior, and the central amplitude is allowed to adjust near 0.06.
This certifies a stationary background, not its dynamical stability.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpmath import iv

from check_asymptotic_exterior import compare_saved_endpoint, endpoint_enclosure, lower
from check_background_defect import background_jets, field_equations
from check_barrier_certificate import Jet, exact, enclosure, matrix_formula
from check_exterior_response import certify_derivatives, endpoint_changes
from check_finite_static_inverse import upper, interval_norm, positive_product_bound, validate_inverse, gamma
from check_regular_center import certify_center, leading
from finite_static_inverse import WEIGHTS, discrete_inverse
from trapped_axial import Background


BALL_RADIUS = "2e-8"


def absmax(values):
    return max(upper(abs(v)) for v in values)


def complementary_minor(rows):
    a, b = [exact(float(v)) for v in rows[0, [1, 2]]]
    c, d = [exact(float(v)) for v in rows[1, [1, 2]]]
    determinant = a*d-b*c
    assert determinant.a > 0 or determinant.b < 0, "The boundary projection loses a graph condition."
    return lower(abs(determinant))


def close_ball(beta, contraction, radius):
    assert beta >= 0 and 0 <= contraction < 1 and radius > 0, "The nonlinear map is not contracting."
    image = upper(beta+contraction*radius)
    assert exact(image) < radius, "The nonlinear ball is not invariant."
    return image, upper(beta/(1-contraction))


def graph_certificates(background, reference, left, right, left_radius, digits):
    center, basis = certify_center(background, left, float(left_radius), digits)
    _, db, dm, deta = certify_derivatives(digits)
    endpoint_report, delta_left, delta_right, _ = endpoint_changes(background, reference, db, dm, deta, digits)
    nominal_tail = compare_saved_endpoint(background)
    rho = iv.mpf(BALL_RADIUS)
    selected = [0, 3]
    central_contraction = interval_norm(np.eye(2, dtype=int)-basis[selected])
    trace_error = [exact(v) for v in center["nominal_trace_error_against_represented_background"]]
    central_drive = max(upper(trace_error[j]+rho*exact(float(WEIGHTS[j]))) for j in selected)
    central_image = upper(exact(central_drive)+exact(central_contraction)*iv.mpf("1e-4"))
    assert exact(central_contraction) < iv.mpf("5e-6") and exact(central_image) < iv.mpf("1e-4")
    central_leading = leading([iv.mpf(1), exact(float(background.sol(0)[3]))],
                              iv.mpf(str(background.central)), iv.mpf(str(background.scale)), iv.pi)
    central_scalar_lower = lower(1-iv.mpf("1e-4")-exact(float(left_radius))**2*(abs(central_leading[0])+iv.mpf("0.001")))
    assert central_scalar_lower > .999
    minors = [complementary_minor(rows) for rows in (left, right)]
    assert min(minors) > .86

    a, b, u, zeta = endpoint_enclosure(background)
    A0, R = iv.mpf(str(background.central)), iv.mpf(40)
    Mstar = R*u+enclosure(0, exact(nominal_tail["ADM_mass_minus_mass_R_bound"]))
    A_R = A0*(a+enclosure(-rho, rho)*exact(float(WEIGHTS[0])))
    zeta_R = zeta+enclosure(-rho, rho)*exact(float(WEIGHTS[3]))
    MR, omega = enclosure(iv.mpf("5.64"), iv.mpf("5.66")), enclosure(iv.mpf("0.139"), iv.mpf("0.141"))
    f = 1-2*MR/R
    amplitude_bound = exact(absmax([A_R]))
    eta_R = enclosure(-400*amplitude_bound*amplitude_bound, 0)
    mass_image = Mstar-enclosure(0, 14000*amplitude_bound*amplitude_bound)
    omega_image = iv.exp(zeta_R)*iv.sqrt(f)*iv.exp(eta_R)
    assert amplitude_bound < iv.mpf("1e-8")
    assert mass_image.a > MR.a and mass_image.b < MR.b
    assert omega_image.a > omega.a and omega_image.b < omega.b
    # Fixed point: MR=Mstar-tail_mass; omega=exp(zeta_R)*sqrt(f_R)*exp(eta_R).
    exterior_matrix = [[abs(dm[1]-1), abs(dm[2])],
                       [abs(omega_image*(deta[1]-1/(R*f))), abs(omega_image*deta[2])]]
    exterior_contraction = interval_norm(exterior_matrix)
    assert exact(exterior_contraction) < iv.mpf("0.005")
    frequency_bound = iv.exp(zeta_R)*iv.sqrt(1-2*mass_image/R)*iv.exp(eta_R)
    central_amplitude_error = upper(A0*exact(central_drive)/(1-exact(central_contraction)))
    assert exact(central_amplitude_error) < iv.mpf("1.3e-9")

    scaling = [exact(float(v)) for v in WEIGHTS]
    center_error = [v/w for v, w in zip(trace_error, scaling)]
    outer_error = [iv.mpf(0), exact(nominal_tail["normalized_scalar_derivative_error_bound"])/scaling[1],
                   iv.mpf(0), exact(nominal_tail["local_log_frequency_error_bound"])/scaling[3]]
    report = {"radius": BALL_RADIUS, "central_parameter_contraction_bound": central_contraction,
              "central_parameter_image_radius": central_image,
              "central_parameter_displacement_bound": upper(exact(central_drive)/(1-exact(central_contraction))),
              "central_amplitude_displacement_bound": central_amplitude_error,
              "central_normalized_scalar_lower_bound": central_scalar_lower,
              "complementary_boundary_minor_lower_bounds": minors,
              "exterior_parameter_contraction_bound": exterior_contraction,
              "exterior_amplitude_bound": upper(amplitude_bound),
              "exterior_mass_image": [lower(mass_image), upper(mass_image)],
              "exterior_frequency_image": [lower(omega_image), upper(omega_image)],
              "frequency_bound_on_the_graph": [lower(frequency_bound), upper(frequency_bound)],
              "ADM_mass_interval": [lower(Mstar), upper(Mstar)],
              "endpoint_derivative_enclosures": endpoint_report,
              "scope": "Nonlinear endpoint graphs over all selected endpoint states in the stated weighted ball."}
    return report, delta_left, delta_right, center_error, outer_error


def nonlinear_transport(background, mesh, K, radius, reference_bounds=True):
    """Enclose nonlinear remainders; reference_bounds adds the A0=.06 thresholds."""
    knots, nodes = [Fraction(float(v)) for v in background.sol.x], [Fraction(float(v)) for v in mesh]
    partition = sorted(set(nodes) | {v for v in knots if nodes[0] < v < nodes[-1]})
    if len(partition) > 4000:
        raise RuntimeError("Nonlinear verification exceeds 4000 radial points.")
    assert Fraction(20) in partition
    weights = [exact(float(v)) for v in WEIGHTS]
    A0, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    transported = [iv.mpf(0) for _ in K]
    max_hessian, metric_lower, scalar_lower, outer_Q_lower = 0., 1., float("inf"), float("inf")
    for count, (a, b) in enumerate(zip(partition[:-1], partition[1:])):
        piece, cell = bisect_right(knots, (a+b)/2)-1, bisect_right(nodes, (a+b)/2)-1
        r = enclosure(a, b)
        _, represented, _ = background_jets(background, piece, r, order=2)
        box = [v.c[0]+enclosure(-radius, radius)*w for v, w in zip(represented, weights)]
        metric_lower = min(metric_lower, lower(1-2*box[2]))
        assert 1-2*box[2] > 0
        directions = [Jet(0, [v, enclosure(-w, w), iv.mpf(0)]) for v, w in zip(box, weights)]
        values = field_equations(Jet(r, order=2), directions, A0, scale, iv.pi)
        # Real polarization bounds H[u,v] by twice the supremum of H[w,w].
        hessian = 4*exact(max(upper(abs(v.c[2])/w) for v, w in zip(values, weights)))
        max_hessian = max(max_hessian, upper(hessian))
        k, length = exact(float(K[cell])), exact(b-a)
        transported[cell] += iv.exp(k*exact(nodes[cell+1]-b))*(iv.exp(k*length)-1)/k*hessian
        if b <= 20:
            scalar_lower = min(scalar_lower, lower(box[0]))
        if a >= 20:
            ratio = A0*A0*box[0]*box[0]/(scale*scale)
            Q = ((1-2*ratio)*(1-6*ratio)-iv.exp(2*box[3]))/(1-2*box[2])
            outer_Q_lower = min(outer_Q_lower, lower(Q))
        if (count+1) % 1000 == 0:
            print(f"bounded {count+1} of {len(partition)-1} nonlinear intervals", flush=True)
    assert metric_lower > 0 and scalar_lower > 0 and outer_Q_lower > 0
    if reference_bounds:
        assert metric_lower > .37444 and scalar_lower > .00394 and outer_Q_lower > 1.35
        assert max(upper(v) for v in transported) < 23.404
    return np.array([upper(v) for v in transported]), {
        "partition_points": len(partition), "weighted_hessian_bound": max_hessian,
        "maximum_transported_hessian_bound": max(upper(v) for v in transported),
        "metric_factor_lower_bound": metric_lower, "normalized_scalar_lower_up_to_20": scalar_lower,
        "scalar_ODE_Q_lower_from_20_to_40": outer_Q_lower}


def fixed_point_bounds(H, eta, direct, signed_lower, signed_upper, growth, nonlinear,
                       left, right, delta_left, delta_right, center_error, outer_error, radius):
    boundary, derivative = [], []
    for rows, delta, error in ((left, delta_left, center_error), (right, delta_right, outer_error)):
        for row, change in zip(rows, delta):
            # Reparametrizing a nominal exact germ to the represented selected
            # endpoint coordinates contributes the additional delta-row term.
            boundary.append(upper(sum(((abs(exact(float(row[j])))+abs(change[j]))*error[j]
                                       for j in range(4)), iv.mpf(0))))
            derivative.append(upper(sum(map(abs, change), iv.mpf(0))))
    rhs = [enclosure(float(lo), float(hi)) for lo, hi in zip(signed_lower.flat, signed_upper.flat)]
    rhs += [enclosure(-v, v) for v in boundary]
    midpoint = np.array([float(v.mid) for v in rhs])
    radii = np.array([upper(abs(v-exact(float(m)))) for v, m in zip(rhs, midpoint)])
    nominal = H@midpoint
    arithmetic = positive_product_bound(np.abs(H), np.abs(midpoint))
    uncertainty = positive_product_bound(np.abs(H), radii)
    total = max(upper(abs(exact(float(v)))+exact(float(error))+gamma(2*len(H))*exact(float(arithmetic[j]))
                      +iv.mpf(2)**-1022*len(H)*2) for j, (v, error) in enumerate(zip(nominal, uncertainty)))
    nodal_zero = upper(exact(total)/(1-eta))
    zero = max(upper(exact(float(g))*exact(nodal_zero)+exact(float(d))) for g, d in zip(growth, direct))
    nonlinear_input = np.array([upper(radius*exact(float(v))) for v in nonlinear])
    gains = positive_product_bound(np.abs(H), np.r_[np.repeat(nonlinear_input, 4), derivative])
    nodal_gain = upper(exact(float(max(gains)))/(1-eta))
    contraction = max(upper(exact(float(g))*exact(nodal_gain)+exact(float(v))) for g, v in zip(growth, nonlinear_input))
    image, error = close_ball(exact(zero), exact(contraction), radius)
    conversion = max(upper(exact(float(v))/iv.mpf(s)) for v, s in zip(WEIGHTS, ("1", "2", "0.4", "0.5")))
    decimal_error = upper(exact(error)*exact(conversion))
    static_inverse_gain = upper(iv.mpf("19000")/(1-exact(contraction))*exact(conversion))
    assert exact(zero) < iv.mpf("5.75e-9") and exact(contraction) < iv.mpf("0.198")
    assert exact(image) < iv.mpf("9.71e-9") and exact(decimal_error) < iv.mpf("7.2e-9")
    assert exact(static_inverse_gain) < iv.mpf("24000")
    return {"weighted_ball_radius": BALL_RADIUS, "nodal_zero_image_bound": nodal_zero,
            "zero_image_bound": zero, "nodal_derivative_bound": nodal_gain,
            "contraction_bound": contraction, "image_radius_bound": image,
            "fixed_point_error_bound": error, "decimal_weight_error_bound": decimal_error,
            "physical_static_inverse_gain_bound": static_inverse_gain,
            "nonlinear_boundary_defect_bounds": boundary, "boundary_derivative_row_bounds": derivative}


def certify_light_rings(background, radius, intervals=None):
    A0, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    weights = [exact(float(v)) for v in WEIGHTS]
    results = []
    intervals = intervals or ((("15.942938", "15.942947"), True), (("16.527600", "16.527617"), False))
    for endpoints, stable in intervals:
        signs = []
        for endpoint in endpoints:
            r = iv.mpf(endpoint)
            piece = int(np.searchsorted(background.sol.x, float(r.mid), side="right")-1)
            _, represented, _ = background_jets(background, piece, r, order=2)
            a, b, u, zeta = [v.c[0]+enclosure(-radius, radius)*w for v, w in zip(represented, weights)]
            f, s = 1-2*u, A0*A0*a*a
            V = s*(1-2*s/(scale*scale))*(1-2*s/(scale*scale))
            pressure = (iv.exp(2*zeta)*s+f*A0*A0*b*b-V)/2
            H = 3*u+4*iv.pi*r*r*pressure-1
            signs.append(H)
        assert (signs[0].b < 0 < signs[1].a) if stable else (signs[1].b < 0 < signs[0].a)
        lower_r, upper_r = map(Fraction, endpoints)
        cuts = sorted({lower_r, upper_r} | {Fraction(float(x)) for x in background.sol.x if lower_r < Fraction(float(x)) < upper_r})
        derivatives = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            piece = int(np.searchsorted(background.sol.x, float((a+b)/2), side="right")-1)
            r = enclosure(a, b)
            _, represented, _ = background_jets(background, piece, r, order=2)
            y = [v.c[0]+enclosure(-radius, radius)*w for v, w in zip(represented, weights)]
            derivatives.append(8*iv.pi*r*r*iv.exp(2*y[3])*A0*A0*y[0]*y[0]-1)
        assert all(v.a > 0 for v in derivatives) if stable else all(v.b < 0 for v in derivatives)
        results.append({"radius_interval": list(endpoints), "stable": stable,
                        "left_H_interval": [lower(signs[0]), upper(signs[0])],
                        "right_H_interval": [lower(signs[1]), upper(signs[1])],
                        "r_Hprime_at_a_root_interval": [min(lower(v) for v in derivatives), max(upper(v) for v in derivatives)]})
    return results


def check_hessians(background, digits):
    iv.dps = digits
    close_ball(iv.mpf("0.1"), iv.mpf("0.2"), iv.mpf(1))
    for beta, q in ((iv.mpf("0.1"), iv.mpf(1)), (iv.mpf("0.9"), iv.mpf("0.2"))):
        try:
            close_ball(beta, q, iv.mpf(1))
        except AssertionError:
            pass
        else:
            raise AssertionError("A nonclosing ball was accepted.")
    assert complementary_minor(np.array([[0., 1., 0., 0.], [0., 0., 1., 0.]])) > .99
    try:
        complementary_minor(np.zeros((2, 4)))
    except AssertionError:
        pass
    else:
        raise AssertionError("A singular boundary projection was accepted.")
    direction = np.array([.2, -.3, .1, -.2])
    for r in (.002, 1., 15.94, 20., 39.):
        y = background.sol(r)
        jets = [Jet(0, [exact(float(v)), exact(float(d)), iv.mpf(0)]) for v, d in zip(y, direction)]
        values = field_equations(Jet(r, order=2), jets, background.central, background.scale, iv.pi)
        h = 1e-5
        columns = []
        for sign in (-1, 1):
            matrix = matrix_formula(Jet(r), [Jet(float(v)) for v in y+sign*h*direction], background.central, background.scale)
            columns.append(np.array([[float(v.c[0].mid) for v in row] for row in matrix])@direction)
        expected = (columns[1]-columns[0])/(2*h)
        actual = np.array([float((2*v.c[2]).mid) for v in values])
        assert np.max(np.abs(actual-expected)/(1+np.abs(actual))) < 2e-7
    print("PASS: directional Hessians, ball rejection, and complementary boundary projections", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.digits >= 25
    start, background = perf_counter(), Background()
    check_hessians(background, args.digits)
    directory = Path(__file__).parent/"data/s08a06"
    reference, defect_path = directory/"finite-static-reference.npz", directory/"background-defect.npz"
    certificate = json.loads((directory/"background-defect.json").read_text())
    hashes = {"reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "background_sha256": hashlib.sha256((directory/"background.npz").read_bytes()).hexdigest(),
              "defect_enclosure_sha256": hashlib.sha256(defect_path.read_bytes()).hexdigest()}
    assert all(certificate[key] == value for key, value in hashes.items())
    flow_certificate = json.loads((directory/"finite-static-inverse.json").read_text())
    assert all(flow_certificate[key] == hashes[key] for key in ("reference_sha256", "background_sha256"))
    assert exact(flow_certificate["flow_bounds"]["maximum_transition_error_bound"]) < iv.mpf("2.6e-8")
    assert flow_certificate["verified_decimal_bounds"]["continuous_forcing_gain"] == "19000"
    with np.load(reference) as data:
        mesh, transitions, left, right = [data[key].copy() for key in ("mesh", "transitions", "left_rows", "right_rows")]
    with np.load(defect_path) as data:
        assert np.array_equal(mesh, data["mesh"])
        K, direct, growth, lo, hi = [data[key].copy() for key in
            ("matrix_norm_bounds", "direct_forcing_bounds", "growth_bounds", "signed_forcing_lower", "signed_forcing_upper")]
    graph, delta_left, delta_right, center_error, outer_error = graph_certificates(background, reference, left, right, mesh[0], args.digits)
    radius = iv.mpf(BALL_RADIUS)
    nonlinear, interior = nonlinear_transport(background, mesh, K, radius)
    H, _ = discrete_inverse(mesh, transitions, left, right)
    inverse, _ = validate_inverse(mesh, transitions, left, right, H,
        np.full(len(mesh)-1, upper(iv.mpf("2.6e-8"))), np.zeros(len(mesh)-1), 1.,
        np.zeros((len(mesh), 4)), np.zeros(4*len(mesh)))
    fixed_point = fixed_point_bounds(H, exact(inverse["inverse_contraction_bound"]), direct, lo, hi, growth,
        nonlinear, left, right, delta_left, delta_right, center_error, outer_error, radius)
    geometry_radius = iv.mpf("7.2e-9")
    assert exact(fixed_point["fixed_point_error_bound"]) < geometry_radius
    rings = certify_light_rings(background, geometry_radius)
    result = {"status": "nonlinear_global_background_validated", "scope": "Existence and local uniqueness at fixed ADM mass; no radial or nonlinear dynamical stability theorem.",
              "endpoint_graphs": graph, "nonlinear_interior": interior, "fixed_point": fixed_point,
              "light_rings": rings, "light_ring_background_error_bound": "7.2e-9",
              "rechecked_inverse_contraction": inverse["inverse_contraction_bound"],
              "arithmetic_digits": args.digits, **hashes, "elapsed_seconds": perf_counter()-start}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
