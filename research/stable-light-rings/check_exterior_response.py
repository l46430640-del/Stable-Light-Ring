"""Enclose exterior boundary derivatives and replace both reference endpoints.

The interior coefficient matrix and concentrated source remain the saved
reference data. Exact admissible endpoint subspaces alone do not certify
a global nonlinear boson star or its physical fixed-charge response.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from mpmath import iv

from check_asymptotic_exterior import certify as certify_exterior, lower, scaled_kernels, check_finite_exterior
from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import upper, positive_product_bound, validate_flows, validate_inverse
from check_regular_center import Dual, promote, certify_center, matrix_inverse2
from finite_static_inverse import WEIGHTS, discrete_inverse
from trapped_axial import Background


def symmetric(radius):
    return enclosure(-exact(radius), exact(radius))


def nonnegative_resolvent(matrix, source, terms=80):
    """Upper bound on (I-matrix)^(-1) source, with a geometric tail."""
    matrix, source = np.asarray(matrix, dtype=object), np.asarray(source, dtype=object)
    assert matrix.ndim == source.ndim == 2 and matrix.shape[0] == matrix.shape[1] == source.shape[0]
    assert all(exact(entry).a >= 0 for entry in matrix.flat) and all(exact(entry).a >= 0 for entry in source.flat)
    q = exact(max(upper(sum(row, iv.mpf(0))) for row in matrix))
    assert upper(q) < 1, "The nonnegative resolvent did not close."
    result, term = np.full(source.shape, iv.mpf(0), dtype=object), source.copy()
    for _ in range(terms):
        result += term
        term = matrix@term
    for j in range(source.shape[1]):
        tail = q**terms/(1-q)*exact(max(upper(value) for value in source[:, j]))
        result[:, j] += tail
    return np.array([[exact(upper(value)) for value in row] for row in result], dtype=object), upper(q)


def frozen_kernels(invr, decay2, radial_decay2, state, mass_R, omega, *parameters):
    kernels, coefficients = scaled_kernels(invr, decay2, radial_decay2, state, mass_R, omega, *parameters)
    numeric = all(isinstance(v, (int, float, complex, np.number)) for v in [*state, mass_R, omega])
    U, w, k2 = promote([state[0], omega, .9801 if numeric else iv.mpf("0.9801")])
    kernels[0] += (1-w*w-k2)*U
    return kernels, coefficients


def certify_derivatives(digits=30):
    exterior = certify_exterior(digits)
    R, beta, kappa0 = iv.mpf(40), iv.mpf("0.5"), iv.mpf("0.99")
    amplitude, mass_scale, lapse_scale, sigma = map(iv.mpf, ("1e-8", "1e-10", "1e-11", "0.08"))
    mass_R, omega = enclosure(iv.mpf("5.64"), iv.mpf("5.66")), enclosure(iv.mpf("0.139"), iv.mpf("0.141"))
    radial = [enclosure(0, 1/R), enclosure(0, 1), enclosure(0, R)]
    box = [enclosure(-3, 3), enclosure(-3, 3), enclosure(-1, 1), enclosure(-1, 1)]
    constants = [amplitude, mass_scale, lapse_scale, sigma, iv.pi]
    gain = 1/(kappa0*kappa0-beta*beta)
    factors = [gain, R*R/(2*beta)+2*R/(2*beta)**2+2/(2*beta)**3, R/(2*beta)+1/(2*beta)**2]
    derivatives, coefficients = frozen_kernels(*radial, [Dual(value, np.eye(4, dtype=int)[j]) for j, value in enumerate(box)],
                                              mass_R, omega, *constants)
    C = np.array([[exact(upper(factor*(abs(entry.d[0])+abs(entry.d[1])))),
                   exact(upper(factor*abs(entry.d[2]))), exact(upper(factor*abs(entry.d[3])))]
                  for factor, entry in zip(factors, derivatives)], dtype=object)
    parameter_derivatives, _ = frozen_kernels(*radial, box, Dual(mass_R, [1, 0]), Dual(omega, [0, 1]), *constants)
    source = np.full((3, 3), iv.mpf(0), dtype=object)
    source[0, 0] = 1/amplitude
    for i in range(3):
        for j in range(2):
            source[i, j+1] = exact(upper(factors[i]*abs(parameter_derivatives[i].d[j])))
    sensitivity, contraction = nonnegative_resolvent(C, source)
    assert exact(contraction) < iv.mpf("0.614")

    # The amplitude derivative is close to the unit Dirichlet solution of
    # the scalar equation with its coefficients frozen on the exact tail.
    P, Q = coefficients[:2]
    scalar_contraction = upper(gain*(abs(P.v)+abs(Q.v-kappa0*kappa0)))
    assert exact(scalar_contraction) < iv.mpf("0.613")
    amplitude_direction = [sensitivity[0, 0], sensitivity[0, 0], sensitivity[1, 0], sensitivity[2, 0]]
    coefficient_change = sum(((abs(P.d[j])+abs(Q.d[j]))*amplitude_direction[j] for j in range(4)), iv.mpf(0))
    derivative_error = upper(gain*3*amplitude*coefficient_change/(1-exact(scalar_contraction)))
    assert exact(derivative_error) < iv.mpf("8e-11")
    db = [enclosure(-iv.mpf("1.21"), -iv.mpf("0.98"))+symmetric(derivative_error),
          symmetric(upper(amplitude*sensitivity[0, 1])), symmetric(upper(amplitude*sensitivity[0, 2]))]
    dm = [symmetric(upper(mass_scale*sensitivity[1, j]))+int(j == 1) for j in range(3)]
    deta = [symmetric(upper(lapse_scale*sensitivity[2, j])) for j in range(3)]
    for row, bounds in (([db[1], db[2]], ("1.05e-8", "5.9e-8")),
                        ([dm[0], dm[1]-1, dm[2]], ("0.00334", "1.4e-11", "8e-11")),
                        (deta, ("8.4e-5", "3.4e-13", "2.15e-12"))):
        assert all(abs(value) < iv.mpf(bound) for value, bound in zip(row, bounds))
    derivative_intervals = {name: [[lower(value), upper(value)] for value in row]
                            for name, row in (("A_prime_R", db), ("ADM_mass", dm), ("log_z_R", deta))}
    report = {"exterior_existence_contraction": exterior["verified_decimal_bounds"]["contraction"],
              "parameter_order": ["A_R", "M_R", "omega"], "state_block_order": ["scalar_pair", "mass_tail", "log_z"],
              "fixed_kernel_kappa": "0.99", "derivative_contraction_bound": contraction,
              "block_matrix_bounds": [[upper(v) for v in row] for row in C],
              "scaled_sensitivity_bounds": [[upper(v) for v in row] for row in sensitivity],
              "frozen_scalar_contraction_bound": scalar_contraction,
              "frozen_scalar_amplitude_derivative_error_bound": derivative_error,
              "boundary_derivative_intervals": derivative_intervals,
              "arithmetic": f"{digits}-digit intervals on the entire previously certified exterior parameter box"}
    return report, db, dm, deta


def endpoint_changes(background, reference, db, dm, deta, digits=30):
    with np.load(reference) as data:
        left, right, radius = data["left_rows"], data["right_rows"], float(data["mesh"][0])
    center_report, basis = certify_center(background, left, radius, digits)
    scaling = [exact(float(value)) for value in WEIGHTS]
    scaled = np.array([[entry/scaling[i] for entry in row] for i, row in enumerate(basis)], dtype=object)
    left_iv = np.array([[exact(float(entry)) for entry in row] for row in left], dtype=object)
    right_iv = np.array([[exact(float(entry)) for entry in row] for row in right], dtype=object)
    selector = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=int)
    delta_left = -left_iv@scaled@matrix_inverse2(scaled[[0, 3]])@selector

    R, A0 = iv.mpf(40), iv.mpf(str(background.central))
    M_R, omega = enclosure(iv.mpf("5.64"), iv.mpf("5.66")), enclosure(iv.mpf("0.139"), iv.mpf("0.141"))
    zeta_derivative = [-deta[0], 1/(R*(1-2*M_R/R))-deta[1], 1/omega-deta[2]]
    chart = np.array([[zeta_derivative[1], zeta_derivative[2]], [dm[1], dm[2]]], dtype=object)
    mass_pivot = dm[1]-dm[2]*zeta_derivative[1]/zeta_derivative[2]
    assert lower(zeta_derivative[2]) > 7 and lower(mass_pivot) > .99
    # Columns are the scaled scalar value, scaled local frequency, and ADM
    # mass variation. This chart includes the inhomogeneous unit-energy datum.
    endpoint_map = np.full((4, 3), iv.mpf(0), dtype=object)
    for j in range(3):
        da, dzeta, dADM = A0*scaling[0]*int(j == 0), scaling[3]*int(j == 1), iv.mpf(int(j == 2))
        frequency_rhs = dzeta-zeta_derivative[0]*da
        dM = (dADM-dm[0]*da-dm[2]*frequency_rhs/zeta_derivative[2])/mass_pivot
        dw = (frequency_rhs-zeta_derivative[1]*dM)/zeta_derivative[2]
        dB = db[0]*da+db[1]*dM+db[2]*dw
        endpoint_map[:, j] = [iv.mpf(int(j == 0)), dB/(A0*scaling[1]), dM/(R*scaling[2]), iv.mpf(int(j == 1))]
    delta_right = -right_iv@endpoint_map[:, :2]@selector
    new_rhs = right_iv@endpoint_map[:, 2]
    report = {"central_boundary_row_difference_bound": center_report["central_boundary_row_difference_bound"],
              "mass_chart_pivot_lower_bound": lower(mass_pivot),
              "frequency_chart_pivot_lower_bound": lower(zeta_derivative[2]),
              "exterior_chart_determinant_interval": [lower(chart[0, 0]*chart[1, 1]-chart[0, 1]*chart[1, 0]),
                                                      upper(chart[0, 0]*chart[1, 1]-chart[0, 1]*chart[1, 0])],
              "exterior_row_difference_bounds": [upper(sum(map(abs, row), iv.mpf(0))) for row in delta_right],
              "exterior_unit_mass_rhs_intervals": [[lower(v), upper(v)] for v in new_rhs]}
    return report, delta_left, delta_right, new_rhs


def verify_endpoint_replacement(background, reference, delta_left, delta_right, new_rhs):
    with np.load(reference) as data:
        mesh, P, G, transitions, left, right, state, rhs = [data[key].copy() for key in
            ("mesh", "fundamental_coefficients", "regular_coefficients", "transitions", "left_rows", "right_rows", "node_state", "rhs")]
    H, _ = discrete_inverse(mesh, transitions, left, right)
    flow_errors, forcing, growth, flow_report = validate_flows(background, mesh, P, G, transitions)
    inverse_report, point_errors = validate_inverse(mesh, transitions, left, right, H, flow_errors, forcing, growth, state, rhs)
    eta = exact(inverse_report["inverse_contraction_bound"])
    eta0 = exact(inverse_report["nominal_inverse_defect_bound"])
    theta = positive_product_bound(np.abs(H), np.r_[np.repeat(flow_errors, 4), np.zeros(4)])
    # Enclose only the central, exterior, and light-ring rows of the four
    # endpoint-data columns of the exact reference inverse.
    ring = int(np.flatnonzero(mesh == background.rs)[0])
    indices = [*range(4), *range(len(H)-4, len(H)), *range(4*ring, 4*ring+4)]
    K = np.empty((12, 4), dtype=object)
    for j in range(4):
        column = H[:, -4+j]
        global_gain = exact(float(np.max(np.abs(column))))/(1-eta)
        for k, index in enumerate(indices):
            K[k, j] = abs(exact(float(column[index])))+(eta0+exact(float(theta[index])))*global_gain
    delta = np.full((4, 8), iv.mpf(0), dtype=object)
    delta[:2, :4], delta[2:, 4:] = delta_left, delta_right
    absolute_delta = np.array([[exact(upper(abs(v))) for v in row] for row in delta], dtype=object)
    schur = absolute_delta@K[:8]
    y = state/WEIGHTS
    endpoint_values = np.array([exact(float(y[index//4, index % 4]))+symmetric(float(point_errors[index]))
                                for index in indices[:8]], dtype=object)
    drhs = np.array([iv.mpf(0), iv.mpf(0), *(new_rhs[j]-exact(float(rhs[-2+j])) for j in range(2))], dtype=object)
    defect = drhs-delta@endpoint_values
    defect_bound = np.array([[exact(upper(abs(value)))] for value in defect], dtype=object)
    boundary_error, contraction = nonnegative_resolvent(schur, defect_bound)
    change = K[8:]@boundary_error
    conversion = max(upper(exact(float(weight))/iv.mpf(decimal)) for weight, decimal in zip(WEIGHTS, ("1", "2", "0.4", "0.5")))
    interface_error = upper(exact(max(upper(v) for v in change[:, 0]))*exact(conversion))
    assert exact(contraction) < iv.mpf("0.076")
    assert exact(interface_error) < iv.mpf("1e-10")
    return {"flow_bounds": flow_report, "reference_inverse_bounds": inverse_report,
            "endpoint_schur_matrix_bounds": [[upper(v) for v in row] for row in schur],
            "endpoint_schur_contraction_bound": contraction,
            "endpoint_defect_bounds": [upper(v) for v in defect_bound[:, 0]],
            "endpoint_correction_bounds": [upper(v) for v in boundary_error[:, 0]],
            "joint_endpoint_replacement_interface_error_bound": interface_error,
            "verified_decimal_bounds": {"endpoint_contraction": "0.076", "interface_change": "1e-10"}}


def check_formulas():
    iv.dps = 30
    M = np.array([[iv.mpf("0.2"), iv.mpf("0.1")], [iv.mpf("0.1"), iv.mpf("0.3")]], dtype=object)
    source = np.array([[iv.mpf(1)], [iv.mpf(2)]], dtype=object)
    result, _ = nonnegative_resolvent(M, source)
    inverse = matrix_inverse2(np.eye(2, dtype=int)-M)
    expected = inverse@source
    assert all(result[j, 0].a >= expected[j, 0].b for j in range(2))
    assert max(abs(float(result[j, 0].mid)-float(expected[j, 0].mid)) for j in range(2)) < 1e-12
    try:
        nonnegative_resolvent(np.eye(2, dtype=int)*iv.mpf(1), source)
    except AssertionError:
        pass
    else:
        raise AssertionError("A noncontracting endpoint map was accepted.")
    state = [.5, -.6, .1, -.02]
    constants = [.003, .01, .001, .08, np.pi]
    parameters = [5.65, .14]
    radial = [1/42, np.exp(-2), 42*np.exp(-2)]
    dual, _ = frozen_kernels(*radial, state, Dual(parameters[0], [1, 0]), Dual(parameters[1], [0, 1]), *constants)
    for j in range(2):
        shifted = np.array(parameters, dtype=complex)
        shifted[j] += 1e-25j
        numeric, _ = frozen_kernels(*radial, state, *shifted, *constants)
        expected = np.imag(numeric)/1e-25
        actual = np.array([float(entry.d[j].mid) for entry in dual])
        assert np.max(np.abs(expected-actual)) < 1e-12
    print("PASS: fixed-kernel parameter derivatives, finite resolvent, noncontraction rejection", flush=True)


def check_boundary_differences(background, db, dm, deta):
    """Finite-domain one-sided differences, independent of the interval proof."""
    base_parameters = np.array([5e-9, background.mass, background.omega])
    steps = np.array([1e-9, 1e-4, 1e-5])
    nodes = []
    def values(parameters):
        amp, mass, omega = parameters
        model = SimpleNamespace(mass=mass, omega=omega, sol=background.sol)
        report = check_finite_exterior(model, amplitude=amp)
        nodes.append(report["radial_nodes"])
        return np.array([-amp*report["minus_Aprime_over_A_at_R"], amp*amp*report["mass_tail_over_A_R_squared"],
                         -amp*amp*report["minus_log_z_R_over_A_R_squared"]])
    base = values(base_parameters)
    differences = np.empty((3, 3))
    for j, step in enumerate(steps):
        shifted = base_parameters.copy()
        shifted[j] += step
        differences[:, j] = (values(shifted)-base)/step
    differences[1, 1] += 1
    assert sum(nodes) <= 4000
    for i, row in enumerate((db, dm, deta)):
        assert all(lower(row[j]) < differences[i, j] < upper(row[j]) for j in range(3))
    return {"parameter_order": ["A_R", "M_R", "omega"], "row_order": ["A_prime_R", "ADM_mass", "log_z_R"],
            "one_sided_differences": differences.tolist(), "nodes_per_problem": nodes, "total_radial_nodes": sum(nodes),
            "scope": "Finite-domain floating-point consistency only; not used by derivative or endpoint certificates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.digits >= 25
    start, background = perf_counter(), Background()
    check_formulas()
    derivative_report, db, dm, deta = certify_derivatives(args.digits)
    reference = Path(__file__).parent/"data/s08a06/finite-static-reference.npz"
    endpoint_report, delta_left, delta_right, new_rhs = endpoint_changes(background, reference, db, dm, deta, args.digits)
    result = {"scope": "Exact central/exterior endpoint subspaces on the represented interior; global physical background not certified.",
              "exterior_derivatives": derivative_report, "endpoint_charts": endpoint_report,
              "finite_domain_derivative_crosscheck": check_boundary_differences(background, db, dm, deta),
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "background_sha256": hashlib.sha256(reference.with_name("background.npz").read_bytes()).hexdigest()}
    if not args.local_only:
        result["endpoint_replacement"] = verify_endpoint_replacement(background, reference, delta_left, delta_right, new_rhs)
    result["elapsed_seconds"] = perf_counter()-start
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
