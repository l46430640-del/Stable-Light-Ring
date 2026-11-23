"""Bound the nonlinear background defect through the verified static inverse.

This is a linear correction estimate on the represented background. A
nonlinear existence claim additionally requires a closed neighborhood
estimate and nonlinear matching of the central and exterior families.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import comb, factorial
from pathlib import Path
from time import perf_counter

import numpy as np
from mpmath import iv

from boson_star import equations, SINGULAR
from check_asymptotic_exterior import compare_saved_endpoint, lower
from check_barrier_certificate import Jet, exact, enclosure, polynomial, matrix_formula
from check_finite_static_inverse import upper, interval_norm, positive_product_bound, p_jets, gamma, validate_inverse
from check_regular_center import certify_center, Dual, exponential, promote
from check_exterior_response import certify_derivatives, endpoint_changes
from finite_static_inverse import WEIGHTS, discrete_inverse
from trapped_axial import Background


def field_equations(r, y, central, scale, pi):
    if any(isinstance(value, Jet) for value in [r, *y]):
        order = next(value.order for value in [r, *y] if isinstance(value, Jet))
        r, central, scale, pi = [Jet.coerce(value, order) for value in (r, central, scale, pi)]
        a, b, u, zeta = [Jet.coerce(value, order) for value in y]
    else:
        r, a, b, u, zeta, central, scale, pi = promote([r, *y, central, scale, pi])
    f, A = 1-2*u, central*a
    s = A*A
    ratio = s/(scale*scale)
    V, Vs = s*(1-2*ratio)*(1-2*ratio), (1-2*ratio)*(1-6*ratio)
    omega2 = (2*zeta).exp() if isinstance(zeta, (Jet, Dual)) else exponential(2*zeta)
    rho = (omega2*s+f*central*central*b*b+V)/2
    pressure = (omega2*s+f*central*central*b*b-V)/2
    return [b, ((Vs-omega2)*a-(2*u/r-4*pi*r*V)*b)/f-2*b/r,
            4*pi*r*rho-u/r, -(u/r+4*pi*r*pressure)/f]


def background_jets(background, piece, radius, order=5):
    r = Jet(0, [radius, iv.mpf(1)]+[iv.mpf(0)]*(order-1))
    argument = r-exact(float(background.sol.x[piece]))
    y = [polynomial(background.sol.c[:, piece, j], argument) for j in range(4)]
    dy = [polynomial(background.sol.c[:, piece, j], argument, 1) for j in range(4)]
    return r, y, dy


def inverse_polynomial_jets(P, radius, middle, half, order=5):
    coefficients = p_jets(P, (radius-middle)/half, half, order)
    nominal = np.linalg.inv(np.array([[float(v.mid) for v in row] for row in coefficients[0]]))
    B = np.array([[exact(float(v)) for v in row] for row in nominal], dtype=object)
    eta = interval_norm(np.eye(4, dtype=int)-B@coefficients[0])
    assert eta < .8, "A local polynomial inverse is not enclosed."
    # Preconditioned interval elimination preserves small off-diagonal entries.
    matrix = B@coefficients[0]
    for k in range(4):
        pivot = matrix[k, k]
        assert pivot.a > 0 or pivot.b < 0
        for j in range(4):
            B[k, j] /= pivot
            if j != k:
                matrix[k, j] /= pivot
        matrix[k, k] = iv.mpf(1)
        for i in range(4):
            if i == k:
                continue
            multiplier = matrix[i, k]
            for j in range(4):
                B[i, j] -= multiplier*B[k, j]
                if j != k:
                    matrix[i, j] -= multiplier*matrix[k, j]
            matrix[i, k] = iv.mpf(0)
    inverse = [B]
    for k in range(1, order+1):
        inverse.append(-inverse[0]@sum((coefficients[j]@inverse[k-j] for j in range(1, k+1)),
                                      np.full((4, 4), iv.mpf(0), dtype=object)))
    return inverse


def certify_residual(background, mesh, polynomials, digits=30, fundamental_defect_bound="7.1e-8"):
    """Use a previously enclosed bound on P'-A*P over every transport cell."""
    iv.dps = digits
    assert exact(fundamental_defect_bound).a >= 0
    knots, nodes = [Fraction(float(x)) for x in background.sol.x], [Fraction(float(x)) for x in mesh]
    partition = sorted(set(nodes) | {x for x in knots if nodes[0] < x < nodes[-1]})
    if len(partition) > 4000:
        raise RuntimeError("Background verification exceeds 4000 radial points.")
    central, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    weights = [exact(float(value)) for value in WEIGHTS]
    records, K = [], np.zeros(len(nodes)-1)
    plus = [p_jets(P, iv.mpf(1), iv.mpf(1), 0)[0] for P in polynomials]
    largest_residual, jump_total = iv.mpf(0), iv.mpf(0)
    for count, (a, b) in enumerate(zip(partition[:-1], partition[1:])):
        c, h = (a+b)/2, (b-a)/2
        piece, cell = bisect_right(knots, c)-1, bisect_right(nodes, c)-1
        residuals = []
        for radius in (exact(c), enclosure(a, b)):
            r, y, dy = background_jets(background, piece, radius)
            F = field_equations(r, y, central, scale, iv.pi)
            residuals.append([(dy[j]-F[j])/weights[j] for j in range(4)])
        center, ranged = residuals
        h_iv, width = exact(h), exact(b-a)
        offset = enclosure(-h, h)
        # Preserve cancellation also in the derivatives of the residual used
        # by the Green integral remainder, not just in its zeroth derivative.
        for j in range(4):
            order = center[j].order
            for k in range(order):
                narrowed = sum((comb(n, k)*center[j].c[n]*offset**(n-k) for n in range(k, order)), iv.mpf(0))
                narrowed += comb(order, k)*ranged[j].c[order]*offset**(order-k)
                ranged[j].c[k] = iv.mpf([max(ranged[j].c[k].a, narrowed.a), min(ranged[j].c[k].b, narrowed.b)])
        defect, integral = [], []
        for j in range(4):
            coefficients = center[j].c
            order = center[j].order
            bound = sum((abs(coefficients[k])*h_iv**k for k in range(order)), iv.mpf(0))
            bound += abs(ranged[j].c[order])*h_iv**order
            remainder = upper(2*abs(ranged[j].c[order])*h_iv**(order+1)/(order+1))
            moment = sum((2*h_iv**(k+1)/(k+1)*coefficients[k] for k in range(0, order, 2)), iv.mpf(0))
            integral.append(moment+enclosure(-remainder, remainder))
            defect.append(upper(bound))
        max_residual = max(defect)
        largest_residual = max(largest_residual, exact(max_residual))
        matrix = matrix_formula(Jet(r.c[0]), [Jet(v.c[0]) for v in y], central, scale)
        scaled_matrix = [[matrix[i][j].c[0]*weights[j]/weights[i] for j in range(4)] for i in range(4)]
        K[cell] = max(K[cell], interval_norm(scaled_matrix))

        middle, half = exact((nodes[cell]+nodes[cell+1])/2), exact((nodes[cell+1]-nodes[cell])/2)
        products, inverse_norm = [], None
        for radius, residual in ((exact(c), center), (enclosure(a, b), ranged)):
            inverse = inverse_polynomial_jets(polynomials[cell], radius, middle, half)
            R = [np.array([v.c[k] for v in residual], dtype=object) for k in range(6)]
            products.append([sum((inverse[j]@R[k-j] for j in range(k+1)),
                                  np.full(4, iv.mpf(0), dtype=object)) for k in range(6)])
            inverse_norm = interval_norm(inverse[0])
        center_product, range_product = products
        weighted_integral = sum((center_product[k]*(2*h_iv**(k+1)/(k+1)) for k in (0, 2, 4)),
                                np.full(4, iv.mpf(0), dtype=object))
        for j in range(4):
            remainder = upper(abs(range_product[5][j])*h_iv**6/3)
            weighted_integral[j] += enclosure(-remainder, remainder)

        # At a polynomial join the correction cancels the represented jump.
        jump = [iv.mpf(0)]*4
        if b in knots[1:-1]:
            next_piece = bisect_right(knots, b)-1
            _, previous, _ = background_jets(background, piece, exact(b))
            for j in range(4):
                jump[j] = (exact(float(background.sol.c[-1, next_piece, j]))-previous[j].c[0])/weights[j]
        jump_norm = max(upper(abs(value)) for value in jump)
        jump_total += exact(jump_norm)
        if jump_norm > 1e-300:
            inverse_at_join = inverse_polynomial_jets(polynomials[cell], exact(b), middle, half, order=0)[0]
            weighted_integral += inverse_at_join@np.array(jump, dtype=object)
        transported_integral = plus[cell]@weighted_integral
        records.append((cell, a, b, max_residual, integral, jump, transported_integral, inverse_norm))
        if (count+1) % 1000 == 0:
            print(f"bounded {count+1} of {len(partition)-1} background intervals", flush=True)

    ordinary, cancelled = [iv.mpf(0) for _ in K], [iv.mpf(0) for _ in K]
    signed = np.full((len(K), 4), iv.mpf(0), dtype=object)
    for cell, a, b, residual, moment, jump, transported, inverse_norm in records:
        k, width = exact(float(K[cell])), exact(b-a)
        to_end = iv.exp(k*exact(nodes[cell+1]-b))
        ordinary[cell] += to_end*((iv.exp(k*width)-1)/k*exact(residual)
                                  +exact(max(upper(abs(v)) for v in jump)))
        # Integration by parts retains the small signed residual integral.
        remainder = (iv.exp(k*width)-1-k*width)/k*exact(residual)
        cancelled[cell] += to_end*(exact(max(upper(abs(moment[j]+jump[j])) for j in range(4)))+remainder)
        kernel_error = exact(fundamental_defect_bound)*exact(inverse_norm)*(iv.exp(k*exact(nodes[cell+1]-a))-1)/k
        error = upper(kernel_error*(width*exact(residual)+exact(max(upper(abs(v)) for v in jump))))
        signed[cell] -= transported+enclosure(-error, error)
    lengths = [exact(b-a) for a, b in zip(nodes[:-1], nodes[1:])]
    growth = [upper(iv.exp(exact(float(k))*h)) for k, h in zip(K, lengths)]
    report = {"partition_points": len(partition), "weighted_residual_bound": upper(largest_residual),
              "weighted_polynomial_jump_sum_bound": upper(jump_total),
              "maximum_cell_matrix_norm_bound": float(max(K)),
              "maximum_direct_forcing_bound": max(upper(v) for v in ordinary),
              "maximum_cancelled_forcing_bound": max(upper(v) for v in cancelled),
              "maximum_signed_forcing_width": max(upper(v.b-v.a) for v in signed.flat),
              "green_integral_taylor_order": 5,
              "maximum_cell_growth_bound": max(growth), "arithmetic_digits": digits}
    return report, np.array([upper(v) for v in ordinary]), np.array([upper(v) for v in cancelled]), growth, signed, K


def check_formulas(background, digits=30):
    iv.dps = digits
    for order in (3, 5, 7):
        x = Jet(0, [iv.mpf(0), iv.mpf(1)]+[iv.mpf(0)]*(order-1))
        assert all((x.exp().c[k]-iv.mpf(1)/factorial(k)).a <= 0 <=
                   (x.exp().c[k]-iv.mpf(1)/factorial(k)).b for k in range(order+1))
        assert all(value == 1 for value in (1/(1-x)).c)
        p = polynomial([2., -3., 5., 7.], x+1)
        assert all(p.c[j] == value for j, value in enumerate([11, 5, 3, 2]+[0]*(order-3)))
    try:
        Jet(0)+Jet(0, order=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal Taylor orders were silently combined.")
    P = np.zeros((13, 4, 4))
    P[0], P[1] = np.eye(4), np.diag([.125, .25, .375, .5])
    radius = enclosure(iv.mpf("-0.1"), iv.mpf("0.1"))
    inverse = inverse_polynomial_jets(P, radius, iv.mpf(0), iv.mpf(1))
    for k, matrix in enumerate(inverse):
        for j in range(4):
            for x in (iv.mpf("-0.09"), iv.mpf(0), iv.mpf("0.09")):
                expected = (-exact(float(P[1, j, j])))**k/(1+exact(float(P[1, j, j]))*x)**(k+1)
                assert matrix[j, j].a <= expected.a and matrix[j, j].b >= expected.b
    at_center = inverse_polynomial_jets(P, iv.mpf(0), iv.mpf(0), iv.mpf(1))
    h = iv.mpf("0.1")
    for j in range(4):
        c = exact(float(P[1, j, j]))
        # Manufactured Green integral with P=1+c*r and residual R=r.
        center_product = [iv.mpf(0)]+[at_center[k][j, j] for k in range(4)]
        last = inverse[5][j, j]*radius+inverse[4][j, j]
        value = sum((2*h**(k+1)/(k+1)*center_product[k] for k in (0, 2, 4)), iv.mpf(0))
        error = upper(abs(last)*h**6/3)
        value = (1+c*h)*(value+enclosure(-error, error))
        expected = (1+c*h)*(2*h/c-(iv.ln(1+c*h)-iv.ln(1-c*h))/(c*c))
        assert value.a <= expected.a and value.b >= expected.b
    try:
        inverse_polynomial_jets(np.zeros_like(P), iv.mpf(0), iv.mpf(0), iv.mpf(1))
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("A singular polynomial inverse was accepted.")
    for r in (.002, .1, 4., 12., 16., 30.):
        y = background.sol(r)
        actual = np.array(field_equations(r, y, background.central, background.scale, np.pi))
        expected = equations(np.array([r]), y[:, None], background.central, background.scale)[:, 0]+SINGULAR@y/r
        assert np.max(np.abs(actual-expected)) < 1e-13
        jacobian = matrix_formula(Jet(r), [Jet(float(v)) for v in y], background.central, background.scale)
        for j in range(4):
            shifted = y.astype(complex)
            shifted[j] += 1e-25j
            column = np.imag(field_equations(r, shifted, background.central, background.scale, np.pi))/1e-25
            expected = [float(jacobian[i][j].c[0].mid) for i in range(4)]
            assert np.max(np.abs(column-expected)) < 2e-12
    print("PASS: Taylor orders, interval inverses, manufactured Green integrals, field equations and Jacobian", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.digits >= 25
    start, background = perf_counter(), Background()
    check_formulas(background, args.digits)
    directory = Path(__file__).parent/"data/s08a06"
    reference = directory/"finite-static-reference.npz"
    with np.load(reference) as data:
        mesh, transitions, left, right, polynomials = [data[key].copy() for key in
            ("mesh", "transitions", "left_rows", "right_rows", "fundamental_coefficients")]
    certificate = json.loads((directory/"finite-static-inverse.json").read_text())
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    background_hash = hashlib.sha256((directory/"background.npz").read_bytes()).hexdigest()
    assert certificate["reference_sha256"] == reference_hash and certificate["background_sha256"] == background_hash
    assert certificate["verified_decimal_bounds"]["inverse_contraction"] == "0.000021"
    assert exact(certificate["flow_bounds"]["maximum_fundamental_defect_bound"]) < iv.mpf("7.1e-8")
    assert exact(certificate["flow_bounds"]["maximum_transition_error_bound"]) < iv.mpf("2.6e-8")
    local, direct, cancelled, growth, signed, matrix_norms = certify_residual(background, mesh, polynomials, args.digits)
    result = {"scope": "Linear correction to the nonlinear ODE defect on the represented background; nonlinear existence not certified.",
              "local_defect": local}
    if not args.local_only:
        H, _ = discrete_inverse(mesh, transitions, left, right)
        # The newly reconstructed inverse is checked afresh. Only the exact
        # flow enclosure, tied to the unchanged data hashes, is reused.
        inverse_report, _ = validate_inverse(mesh, transitions, left, right, H,
            np.full(len(mesh)-1, upper(iv.mpf("2.6e-8"))), np.zeros(len(mesh)-1), 1.,
            np.zeros((len(mesh), 4)), np.zeros(4*len(mesh)))
        center, _ = certify_center(background, left, float(mesh[0]), args.digits)
        exterior = compare_saved_endpoint(background)
        center_error = [exact(v)/exact(float(weight)) for v, weight in zip(center["nominal_trace_error_against_represented_background"], WEIGHTS)]
        outer_error = [iv.mpf(0), exact(exterior["normalized_scalar_derivative_error_bound"])/exact(float(WEIGHTS[1])),
                       iv.mpf(0), exact(exterior["local_log_frequency_error_bound"])/exact(float(WEIGHTS[3]))]
        boundary = np.array([upper(sum((abs(exact(float(row[j])))*error[j] for j in range(4)), iv.mpf(0)))
                             for rows, error in ((left, center_error), (right, outer_error)) for row in rows])
        eta = exact(inverse_report["inverse_contraction_bound"])
        estimates = {}
        for name, forcing in (("direct", direct), ("cancelled", cancelled)):
            effects = positive_product_bound(np.abs(H), np.r_[np.repeat(forcing, 4), boundary])
            nodal = upper(exact(float(max(effects)))/(1-eta))
            continuous = max(upper(exact(float(g))*exact(nodal)+exact(float(d))) for g, d in zip(growth, direct))
            estimates[name] = {"nodal_correction_bound": nodal, "continuous_correction_bound": continuous}
        rhs = [*signed.flat, *(enclosure(-float(v), float(v)) for v in boundary)]
        midpoint = np.array([float(v.mid) for v in rhs])
        radii = np.array([upper(abs(v-exact(float(m)))) for v, m in zip(rhs, midpoint)])
        nominal = H@midpoint
        arithmetic = positive_product_bound(np.abs(H), np.abs(midpoint))
        uncertainty = positive_product_bound(np.abs(H), radii)
        total = max(upper(abs(exact(float(v)))+exact(float(error))+gamma(2*len(H))*exact(float(arithmetic[j]))
                          +iv.mpf(2)**-1022*len(H)*2) for j, (v, error) in enumerate(zip(nominal, uncertainty)))
        nodal = upper(exact(total)/(1-eta))
        continuous = max(upper(exact(float(g))*exact(nodal)+exact(float(d))) for g, d in zip(growth, direct))
        estimates["signed_green"] = {"nominal_nodal_correction_maximum": float(max(abs(nominal))),
                                     "nodal_correction_bound": nodal, "continuous_correction_bound": continuous,
                                     "propagated_interval_uncertainty": float(max(uncertainty))}
        _, db, dm, deta = certify_derivatives(args.digits)
        _, delta_left, delta_right, _ = endpoint_changes(background, reference, db, dm, deta, args.digits)
        differences = [*delta_left, *delta_right]
        row_bounds = [exact(upper(sum(map(abs, row), iv.mpf(0)))) for row in differences]
        data_bounds = [exact(upper(sum((abs(row[j])*error[j] for j in range(4)), iv.mpf(0))))
                       for rows, error in ((delta_left, center_error), (delta_right, outer_error)) for row in rows]
        column_bounds = np.array([[abs(exact(float(v)))+eta/(1-eta)*exact(float(max(abs(H[:, -4+j]))))
                                  for j, v in enumerate(row)] for row in H[:, -4:]], dtype=object)
        boundary_gain = max(upper(sum((v*bound for v, bound in zip(row, row_bounds)), iv.mpf(0))) for row in column_bounds)
        boundary_offset = max(upper(sum((v*bound for v, bound in zip(row, data_bounds)), iv.mpf(0))) for row in column_bounds)
        assert exact(boundary_gain) < iv.mpf("0.08")
        corrected_node = upper((exact(nodal)+exact(boundary_offset))/(1-exact(boundary_gain)))
        corrected_continuous = max(upper(exact(float(g))*exact(corrected_node)+exact(float(d))) for g, d in zip(growth, direct))
        conversion = max(upper(exact(float(v))/iv.mpf(s)) for v, s in zip(WEIGHTS, ("1", "2", "0.4", "0.5")))
        corrected_continuous = upper(exact(corrected_continuous)*exact(conversion))
        assert exact(nodal) < iv.mpf("7e-10")
        assert exact(corrected_node) < iv.mpf("8e-10")
        assert exact(corrected_continuous) < iv.mpf("6e-9")
        estimates["exact_endpoint_tangents"] = {"boundary_perturbation_gain": boundary_gain,
            "boundary_anchor_offset_bound": boundary_offset, "nodal_correction_bound": corrected_node,
            "decimal_weight_continuous_correction_bound": corrected_continuous,
            "scope": "Linearized matching to the exact nominal central/exterior germs, with ADM mass fixed to that of the nominal exterior."}
        result.update({"endpoint_defect_bounds": boundary.tolist(), "linear_correction": estimates,
                       "rechecked_inverse": {key: inverse_report[key] for key in
                           ("nominal_inverse_defect_bound", "flow_perturbation_bound", "inverse_contraction_bound", "node_inverse_norm_bound")},
                       "reference_sha256": reference_hash, "background_sha256": background_hash})
    result["elapsed_seconds"] = perf_counter()-start
    if args.output:
        array_path = args.output.with_suffix(".npz")
        np.savez_compressed(array_path, mesh=mesh, matrix_norm_bounds=matrix_norms,
            direct_forcing_bounds=direct, signed_forcing_lower=np.array([[lower(v) for v in row] for row in signed]),
            signed_forcing_upper=np.array([[upper(v) for v in row] for row in signed]), growth_bounds=growth,
            reference_sha256=reference_hash, background_sha256=background_hash)
        with np.load(array_path) as data:
            assert all(exact(float(lo)).a <= v.a and v.b <= exact(float(hi)).b for lo, hi, v in
                       zip(data["signed_forcing_lower"].flat, data["signed_forcing_upper"].flat, signed.flat))
        result["defect_enclosure_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
