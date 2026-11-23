"""Validate the finite reference static inverse using interval flow defects.

This reference problem retains approximate center and tail boundary rows
on a represented background. It is not an existence proof for an exact
asymptotically flat boson star. Its local flow polynomials are validated
on every background interpolation interval, including their joins.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import comb, inf, nextafter
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import mpmath
from mpmath import iv

from check_barrier_certificate import exact, enclosure
from finite_static_inverse import WEIGHTS, SCALED_SINGULAR, build_discrete_inverse, coefficient_jets, discrete_inverse
from static_transversality import static_matrix
from trapped_axial import Background


if not __debug__:
    raise RuntimeError("Run the certificate with assertions enabled.")


def upper(value):
    result = nextafter(float(value.b), inf)
    if not np.isfinite(result):
        raise ArithmeticError("An interval bound is not finite.")
    return result


def interval_norm(matrix):
    return max(upper(sum((abs(entry) for entry in row), iv.mpf(0))) for row in matrix)


def p_jets(P, argument, half, order):
    result = []
    for derivative in range(order+1):
        matrix = np.empty((4, 4), dtype=object)
        for i in range(4):
            for j in range(4):
                value = iv.mpf(0)
                for k in range(len(P)-1, derivative-1, -1):
                    value = value * argument + exact(float(P[k, i, j])) * comb(k, derivative)
                matrix[i, j] = value / half**derivative
        result.append(matrix)
    return result


def validate_flows(background, mesh, polynomials, regular_coefficients, transitions, digits=30):
    assert digits >= 25
    iv.dps = digits
    assert len(mesh) >= 2 and np.all(np.diff(mesh) > 0)
    assert 0 < mesh[0] < mesh[-1] <= background.radius
    assert polynomials.shape == (len(mesh)-1, 13, 4, 4)
    assert regular_coefficients.shape == (len(mesh)-1, 9, 4, 4)
    assert transitions.shape == (len(mesh)-1, 4, 4)
    assert all(np.all(np.isfinite(array)) for array in (mesh, polynomials, regular_coefficients, transitions))
    bg_knots = [Fraction(float(r)) for r in background.sol.x]
    meshq = [Fraction(float(r)) for r in mesh]
    partition = sorted(set(meshq) | {x for x in bg_knots if meshq[0] < x < meshq[-1]})
    if len(partition) > 4000:
        raise RuntimeError("Flow verification exceeds 4000 radial points.")
    K = np.zeros(len(mesh)-1)
    defects = np.zeros_like(K)
    polynomial_defects, polynomial_norms = [], []
    Si = np.array([[exact(float(entry)) for entry in row] for row in SCALED_SINGULAR], dtype=object)
    for cell, (left, right) in enumerate(zip(meshq[:-1], meshq[1:])):
        c, h = exact((left+right)/2), exact((right-left)/2)
        P = np.array([[[exact(float(entry)) for entry in row] for row in matrix]
                      for matrix in polynomials[cell]], dtype=object)
        G = np.array([[[exact(float(entry)) for entry in row] for row in matrix]
                      for matrix in regular_coefficients[cell]], dtype=object)
        remainder = np.full((len(P)+len(G), 4, 4), iv.mpf(0), dtype=object)
        for n, matrix in enumerate(P):
            remainder[n] += n*matrix - Si@matrix
            if n:
                remainder[n-1] += matrix*(c/h*n)
            for k, regular_matrix in enumerate(G):
                product = regular_matrix@matrix
                remainder[k+n] -= product*c
                remainder[k+n+1] -= product*h
        polynomial_defects.append(upper(sum((exact(interval_norm(v)) for v in remainder), iv.mpf(0))/exact(left)))
        polynomial_norms.append(upper(sum((exact(interval_norm(v)) for v in P), iv.mpf(0))))
    for count, (a, b) in enumerate(zip(partition[:-1], partition[1:])):
        c = (a+b) / 2
        cell = bisect_right(meshq, c) - 1
        piece = bisect_right(bg_knots, c) - 1
        middle, halfq = (meshq[cell]+meshq[cell+1])/2, (meshq[cell+1]-meshq[cell])/2
        half = exact(halfq)

        def residual_jets(radius, order):
            matrix = coefficient_jets(background, radius, piece, regular=True)
            A = [np.array([[matrix[i][j].c[k] for j in range(4)] for i in range(4)], dtype=object)
                 for k in range(order+1)]
            G = p_jets(regular_coefficients[cell], (radius-exact(middle))/half, half, order)
            return A[0] + Si/radius, [G[k]-A[k] for k in range(order+1)]

        A, ranged = residual_jets(enclosure(a, b), 3)
        _, centered = residual_jets(exact(c), 2)
        delta = enclosure(-(b-a)/2, (b-a)/2)
        R = centered[0] + centered[1]*delta + centered[2]*delta**2 + ranged[3]*delta**3
        K[cell] = max(K[cell], interval_norm(A))
        defect_bound = upper(exact(polynomial_defects[cell]) + exact(interval_norm(R))*exact(polynomial_norms[cell]))
        defects[cell] = max(defects[cell], defect_bound)
        if (count+1) % 1000 == 0:
            print(f"verified {count+1} of {len(partition)-1} coefficient intervals", flush=True)

    errors, forcing_weights, growth = [], [], []
    for i, (left, right) in enumerate(zip(meshq[:-1], meshq[1:])):
        P = polynomials[i]
        minus = p_jets(P, iv.mpf(-1), iv.mpf(1), 0)[0]
        plus = p_jets(P, iv.mpf(1), iv.mpf(1), 0)[0]
        B = np.linalg.inv(sum((-1)**k * P[k] for k in range(len(P))))
        Bi = np.array([[exact(float(entry)) for entry in row] for row in B], dtype=object)
        Ti = np.array([[exact(float(entry)) for entry in row] for row in transitions[i]], dtype=object)
        eta = interval_norm(np.eye(4, dtype=int) - Bi @ minus)
        assert eta < 1e-10
        inverse_bound = exact(interval_norm(Bi)) / (1-exact(eta))
        h, k = exact(right-left), exact(float(K[i]))
        expkh = iv.exp(k*h)
        integral = (expkh-1)/k
        endpoint_defect = interval_norm(Ti @ minus - plus)
        error = inverse_bound * (integral * exact(float(defects[i])) + exact(endpoint_defect))
        errors.append(upper(error))
        forcing_weights.append(upper(integral))
        growth.append(upper(expkh))
    return np.array(errors), np.array(forcing_weights), max(growth), {
        "partition_points": len(partition), "maximum_matrix_norm_bound": float(max(K)),
        "maximum_fundamental_defect_bound": float(max(defects)),
        "maximum_transition_error_bound": float(max(errors))}


def gamma(n):
    u = iv.mpf(2)**-53
    return n*u/(1-n*u)


def positive_product_bound(matrix, vector):
    assert np.all(matrix >= 0) and np.all(vector >= 0)
    result = matrix @ vector
    allowance = iv.mpf(2)**-1022 * (2*len(vector)+1)
    return np.array([upper(exact(float(v))/(1-gamma(2*len(vector))) + allowance) for v in result])


def row_norm_bound(matrix):
    return upper(exact(float(np.max(np.sum(np.abs(matrix), axis=1)))) / (1-gamma(matrix.shape[1])))


def validate_inverse(mesh, transitions, left, right, H, flow_errors, forcing, growth, state, rhs):
    assert all(array.dtype == np.float64 for array in (transitions, left, right, H, state, rhs))
    size = len(H)
    product = np.zeros_like(H)
    for k, T in enumerate(transitions):
        product[:, 4*k:4*k+4] -= H[:, 4*k:4*k+4] @ T
        product[:, 4*(k+1):4*(k+1)+4] += H[:, 4*k:4*k+4]
    assert len(left)+len(right) == 4 and 0 < len(left) < 4
    product[:, :4] += H[:, -4:-len(right)] @ left
    product[:, -4:] += H[:, -len(right):] @ right
    product -= np.eye(size)
    Hnorm = row_norm_bound(H)
    Lnorm = upper(1 + exact(max(row_norm_bound(T) for T in transitions)))
    Lnorm = max(Lnorm, row_norm_bound(left), row_norm_bound(right))
    # Every entry uses at most 16 products/additions; gamma_32 also covers
    # the diagonal subtraction. Row summation is bounded separately.
    roundoff = gamma(32) * (exact(Hnorm)*exact(Lnorm)+1) + iv.mpf(2)**-1022 * 32*size**2
    eta0 = upper(exact(row_norm_bound(product)) + roundoff)
    del product
    absH = np.abs(H)
    eta_rows = positive_product_bound(absH, np.r_[np.repeat(flow_errors, 4), np.zeros(4)])
    eta = upper(exact(eta0) + exact(float(max(eta_rows))))
    assert eta < .5, f"The reference inverse contraction did not close: {eta}"
    norm_bound = upper(exact(Hnorm)/(1-exact(eta)))
    forcing_gain = max(positive_product_bound(absH, np.r_[np.repeat(forcing, 4), np.ones(4)]))
    continuous_gain = upper(exact(growth)*exact(float(forcing_gain))/(1-exact(eta)) + exact(float(max(forcing))))
    # A posteriori nodal error for the concentrated-source reference solution.
    y = state / WEIGHTS
    residual = np.zeros(size)
    for k, T in enumerate(transitions):
        value = y[k+1] - T@y[k] - rhs[4*k:4*k+4]
        radius = exact(float(flow_errors[k])) * exact(float(np.max(np.abs(y[k]))))
        arithmetic = gamma(16) * (exact(row_norm_bound(T))*exact(row_norm_bound(y[k:k+1]))
                                  + exact(row_norm_bound(y[k+1:k+2])) + exact(row_norm_bound(rhs[None, 4*k:4*k+4])))
        arithmetic += iv.mpf(2)**-1022 * 32
        for i in range(4):
            residual[4*k+i] = upper(abs(exact(float(value[i]))) + radius + arithmetic)
    boundary = np.r_[left@y[0], right@y[-1]] - rhs[-4:]
    for i in range(4):
        arithmetic = gamma(16)*(exact(Lnorm)*exact(row_norm_bound(y)) + abs(exact(float(rhs[-4+i]))))
        residual[-4+i] = upper(abs(exact(float(boundary[i]))) + arithmetic + iv.mpf(2)**-1022*32)
    effects = positive_product_bound(absH, residual)
    global_error = upper(exact(float(max(effects)))/(1-exact(eta)))
    point_errors = np.array([upper(exact(float(effect)) + (exact(float(eta_row))+exact(eta0))*exact(global_error))
                             for effect, eta_row in zip(effects, eta_rows)])
    return {"nominal_inverse_defect_bound": eta0, "flow_perturbation_bound": float(max(eta_rows)),
            "inverse_contraction_bound": eta, "node_inverse_norm_bound": norm_bound,
            "continuous_forcing_gain_bound": continuous_gain,
            "weighted_node_error_bound": global_error}, point_errors


def check_small_examples(background):
    iv.dps = 30
    for r in (.002, .1, 1., 10., 15.94, 16.8, 30.):
        piece = np.searchsorted(background.sol.x, r, side="right")-1
        regular = coefficient_jets(background, exact(r), piece, regular=True)
        values = np.array([[float(entry.c[0].mid) for entry in row] for row in regular])
        expected = static_matrix(background, r, regular=True)[0, :4, :4]
        expected *= WEIGHTS[None, :]/WEIGHTS[:, None]
        assert np.max(np.abs(values-expected)) < 1e-12
    mesh = np.array([1., 2.])
    T = np.eye(4)[None, :, :]
    y = np.tile([.1, .2, .3, .4], (2, 1)) * WEIGHTS
    rhs = np.r_[np.zeros(4), [.1, .2, .3, .4]]
    for count in (1, 2, 3):
        left, right = np.eye(4)[:count], np.eye(4)[count:]
        H, factor = discrete_inverse(mesh, T, left, right)
        assert np.max(np.abs(factor.solve(rhs).reshape(-1, 4)-y/WEIGHTS)) < 1e-15
        report, _ = validate_inverse(mesh, T, left, right, H, np.zeros(1), np.ones(1), 1., y, rhs)
        assert report["inverse_contraction_bound"] < 1e-12
    try:
        validate_inverse(mesh, T, left, right, np.zeros_like(H), np.zeros(1), np.ones(1), 1., y, rhs)
    except AssertionError:
        pass
    else:
        raise AssertionError("The inverse verifier accepted a zero preconditioner.")
    print("PASS: regular coefficient matrix, manufactured inverse, rejection of invalid preconditioner", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, help="Recheck saved polynomial and endpoint data.")
    args = parser.parse_args()
    bg = Background()
    start = perf_counter()
    check_small_examples(bg)
    if args.reference is None:
        mesh, P, G, transitions, left, right, H, state, rhs, diagnostic = build_discrete_inverse(bg)
    else:
        with np.load(args.reference) as data:
            mesh, P, G, transitions, left, right, state, rhs = [data[key].copy() for key in
                ("mesh", "fundamental_coefficients", "regular_coefficients", "transitions", "left_rows", "right_rows", "node_state", "rhs")]
        H, _ = discrete_inverse(mesh, transitions, left, right)
        index = int(np.flatnonzero(mesh == bg.rs)[0])
        diagnostic = {"radial_nodes": len(mesh), "matrix_dimension": len(H),
                      "inverse_storage_MiB": H.nbytes/2**20, "right_interface_state": state[index].tolist()}
    errors, forcing, growth, flow_report = validate_flows(bg, mesh, P, G, transitions)
    inverse_report, point_errors = validate_inverse(mesh, transitions, left, right, H, errors, forcing, growth, state, rhs)
    index = int(np.flatnonzero(mesh == bg.rs)[0])
    interface_bound = float(max(point_errors[4*index:4*index+4]))
    barrier_path = Path(__file__).parent / "data/s08a06/barrier-response.npz"
    with np.load(barrier_path) as data:
        barrier_value = data["coefficients"][-1, 0]
        assert data["location"] == bg.rs
    # Convert the exact binary reference scaling to the decimal barrier weights.
    decimal_weights = list(map(iv.mpf, ("1", "2", "0.4", "0.5")))
    nominal_scaled = state[index] / WEIGHTS
    difference = [abs(exact(float(WEIGHTS[j]))*exact(float(nominal_scaled[j]))-exact(float(barrier_value[j]))) / decimal_weights[j]
                  for j in range(4)]
    conversion = max(upper(exact(float(WEIGHTS[j]))/decimal_weights[j]) for j in range(4))
    barrier_error = upper(exact(interface_bound)*exact(conversion) + exact(max(upper(v) for v in difference)))
    assert inverse_report["inverse_contraction_bound"] < .000021
    assert inverse_report["continuous_forcing_gain_bound"] < 19000
    assert barrier_error < .0000065
    result = {"status": "finite_reference_inverse_validated", "scope": " ".join(__doc__.splitlines()[2:]),
              "diagnostics": diagnostic, "flow_bounds": flow_report, "inverse_bounds": inverse_report,
              "right_interface_weighted_error_bound": interface_bound,
              "interface_error_against_barrier_polynomial_bound": barrier_error,
              "verified_decimal_bounds": {"inverse_contraction": "0.000021", "continuous_forcing_gain": "19000",
                                           "interface_error_against_barrier_polynomial": "0.0000065"},
              "software": {"numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mpmath.__version__},
              "arithmetic": "30-digit intervals for flows; IEEE binary64 gamma bounds for sparse matrix products and sums",
              "barrier_response_sha256": hashlib.sha256(barrier_path.read_bytes()).hexdigest(),
              "background_sha256": hashlib.sha256((Path(__file__).parent / "data/s08a06/background.npz").read_bytes()).hexdigest()}
    if args.output:
        path = args.output.with_name("finite-static-reference.npz")
        np.savez_compressed(path, mesh=mesh, fundamental_coefficients=P, regular_coefficients=G,
                            transitions=transitions, left_rows=left, right_rows=right, node_state=state, rhs=rhs)
        result["reference_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["elapsed_seconds"] = perf_counter()-start
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    elif args.reference is not None:
        result["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
