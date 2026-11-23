"""Enclose exact nonlinear central germs and their static boundary subspaces.

The global comparison changes only the central boundary rows of the saved
finite reference problem. Its interior background and exterior rows are
still reference data, not a validated asymptotically flat boson star.
"""

import argparse
import hashlib
import json
from math import inf, nextafter
from pathlib import Path
from time import perf_counter

import numpy as np
import mpmath
from mpmath import iv

from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import (
    interval_norm, positive_product_bound, upper, validate_flows, validate_inverse,
)
from finite_static_inverse import WEIGHTS, discrete_inverse
from static_transversality import central_basis
from trapped_axial import Background


class Dual:
    """First derivatives with interval values and a fixed number of directions."""

    def __init__(self, value, derivative):
        self.v = exact(value.item() if isinstance(value, np.generic) else value)
        self.d = [exact(entry.item() if isinstance(entry, np.generic) else entry) for entry in derivative]

    def coerce(self, other):
        if isinstance(other, Dual):
            assert len(other.d) == len(self.d)
            return other
        return Dual(other, [0] * len(self.d))

    def __add__(self, other):
        other = self.coerce(other)
        return Dual(self.v + other.v, [a+b for a, b in zip(self.d, other.d)])

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, [-a for a in self.d])

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return Dual(self.v * other.v, [a*other.v+self.v*b for a, b in zip(self.d, other.d)])

    __rmul__ = __mul__

    def reciprocal(self):
        return Dual(1/self.v, [-a/(self.v*self.v) for a in self.d])

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def exp(self):
        value = iv.exp(self.v)
        return Dual(value, [value*a for a in self.d])


def exponential(value):
    if isinstance(value, Dual):
        return value.exp()
    if isinstance(value, (complex, float, np.number)):
        return np.exp(value)
    return iv.exp(value)


def promote(values):
    prototype = next((value for value in values if isinstance(value, Dual)), None)
    return values if prototype is None else [prototype.coerce(value) for value in values]


def matter(a, q, s, mass, zeta, central, scale):
    a, q, s, mass, zeta, central, scale = promote([a, q, s, mass, zeta, central, scale])
    f = 1-2*s*mass
    amplitude = central*a
    field2 = amplitude*amplitude
    ratio = field2/(scale*scale)
    V = field2*(1-2*ratio)*(1-2*ratio)
    Vs = (1-2*ratio)*(1-6*ratio)
    Omega2 = exponential(2*zeta)
    W, G = Omega2*field2, f*central*central*s*q*q
    return f, V, Vs, Omega2, (W+G+V)/2, (W+G-V)/2


def integrands(s, X, parameters, central, scale, pi):
    p, q, mass, z, ac, zc, s, central, scale, pi = promote([*X, *parameters, s, central, scale, pi])
    a, zeta = ac+s*p, zc+s*z
    f, V, Vs, Omega2, rho, pressure = matter(a, q, s, mass, zeta, central, scale)
    H = ((Vs-Omega2)*a-s*(2*mass-4*pi*V)*q)/f
    return [q, H, 4*pi*rho, -(mass+4*pi*pressure)/f]


def leading(parameters, central, scale, pi):
    ac, zc, central, scale, pi = promote([*parameters, central, scale, pi])
    _, _, Vs, Omega2, rho, pressure = matter(ac, 0, 0, 0, zc, central, scale)
    q = (Vs-Omega2)*ac/3
    mass = 4*pi*rho/3
    return [q/2, q, mass, -(mass+4*pi*pressure)/2]


def matrix_inverse2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a*d-b*c
    assert float(determinant.a) > 0 or float(determinant.b) < 0
    return np.array([[d/determinant, -b/determinant],
                     [-c/determinant, a/determinant]], dtype=object)


def certify_center(background, left_rows, radius=.002, digits=30, reference_bounds=True):
    """Prove the germ bounds; optionally retain the original A0=.06 calibration."""
    iv.dps = digits
    assert 0 < radius < background.radius
    assert left_rows.shape == (2, 4) and np.all(np.isfinite(left_rows))
    central, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    r, parameter_radius, box_radius = exact(radius), iv.mpf("0.0001"), iv.mpf("0.001")
    nominal = [iv.mpf(1), exact(float(background.sol(0)[3]))]
    parameters = [value+enclosure(-parameter_radius, parameter_radius) for value in nominal]
    baseline = leading([Dual(nominal[j], np.eye(2, dtype=int)[j]) for j in range(2)],
                       central, scale, iv.pi)
    X0, J0 = [entry.v for entry in baseline], [entry.d for entry in baseline]
    box = [value+enclosure(-box_radius, box_radius) for value in X0]
    s = enclosure(0, r*r)
    factors = [iv.mpf(1)/2, iv.mpf(1)/3, iv.mpf(1)/3, iv.mpf(1)/2]
    image = integrands(s, box, parameters, central, scale, iv.pi)
    image_radius = max(upper(abs(factor*value-base)) for factor, value, base in zip(factors, image, X0))
    assert exact(image_radius) < box_radius, "The central box is not invariant."
    derivative = integrands(s, [Dual(box[j], np.eye(4, dtype=int)[j]) for j in range(4)],
                            parameters, central, scale, iv.pi)
    contraction = max(upper(factor*sum(map(abs, value.d), iv.mpf(0)))
                      for factor, value in zip(factors, derivative))
    assert contraction < 1, "The central Volterra map is not contracting."

    nominal_image = integrands(s, X0, nominal, central, scale, iv.pi)
    nominal_defect = max(upper(abs(factor*value-base)) for factor, value, base in zip(factors, nominal_image, X0))
    nominal_error = upper(exact(nominal_defect)/(1-exact(contraction)))
    powers = [r*r, r, r*r, r*r]
    nominal_trace = [powers[i]*X0[i]+(nominal[0] if i == 0 else nominal[1] if i == 3 else 0) for i in range(4)]
    piece = int(np.searchsorted(background.sol.x, radius, side="right")-1)
    argument = r-exact(float(background.sol.x[piece]))
    represented = []
    for i in range(4):
        value = iv.mpf(0)
        for coefficient in background.sol.c[:, piece, i]:
            value = value*argument+exact(float(coefficient))
        represented.append(value)
    trace_error = [upper(abs(nominal_trace[i]-represented[i])+powers[i]*exact(nominal_error))
                   for i in range(4)]

    # Insert the leading parameter derivative into the exact differentiated
    # Volterra map; its residual bounds the derivative of the fixed point.
    inserted = integrands(s, [Dual(value, row) for value, row in zip(box, J0)],
                          [Dual(parameters[j], np.eye(2, dtype=int)[j]) for j in range(2)],
                          central, scale, iv.pi)
    errors = [max(upper(abs(factors[i]*inserted[i].d[j]-J0[i][j])) for i in range(4)) for j in range(2)]
    errors = [upper(exact(error)/(1-exact(contraction))) for error in errors]
    J = [[J0[i][j]+enclosure(-errors[j], errors[j]) for j in range(2)] for i in range(4)]
    basis = np.array([[powers[i]*J[i][j]+int((i, j) in ((0, 0), (3, 1)))
                       for j in range(2)] for i in range(4)], dtype=object)
    scaled = np.array([[entry/exact(float(WEIGHTS[i])) for entry in row]
                       for i, row in enumerate(basis)], dtype=object)
    chart_inverse = matrix_inverse2(scaled[[0, 3]])
    B = np.array([[exact(float(entry)) for entry in row] for row in left_rows], dtype=object)
    adjustment = B @ scaled @ chart_inverse
    row_error = interval_norm(adjustment)
    metric_lower = nextafter(float((1-2*s*box[2]).a), -inf)
    assert metric_lower > 0
    if reference_bounds:
        assert contraction < .501
        assert exact(image_radius) < iv.mpf("0.000503")
        assert exact(row_error) < iv.mpf("0.0000034")
        assert exact(max(trace_error)) < iv.mpf("0.000000000015")
        assert exact(metric_lower) > iv.mpf("0.99999997")
    report = {
        "radius": radius, "central_parameters": {"a_center": 1., "zeta_center": float(background.sol(0)[3]),
        "parameter_radius": "0.0001", "normalization_A0": str(background.central), "potential_scale": str(background.scale)},
        "volterra_box_radius": "0.001", "image_radius_bound": image_radius,
        "contraction_bound": contraction, "leading_derivative_error_bounds": errors,
        "nominal_fixed_point_error_bound": nominal_error,
        "nominal_trace_error_against_represented_background": trace_error,
        "central_boundary_row_difference_bound": row_error,
        "minimum_metric_factor_bound": metric_lower,
        "arithmetic": f"{digits}-digit interval arithmetic, no radial subdivision",
    }
    return report, basis


def compare_boundary(background, reference, center_report):
    with np.load(reference) as data:
        mesh, P, G, transitions, left, right, state, rhs = [data[key].copy() for key in
            ("mesh", "fundamental_coefficients", "regular_coefficients", "transitions", "left_rows", "right_rows", "node_state", "rhs")]
    assert mesh[0] == center_report["radius"]
    assert np.all(rhs[-4:-2] == 0)
    H, _ = discrete_inverse(mesh, transitions, left, right)
    flow_errors, forcing, growth, flow_report = validate_flows(background, mesh, P, G, transitions)
    inverse_report, point_errors = validate_inverse(mesh, transitions, left, right, H,
                                                   flow_errors, forcing, growth, state, rhs)
    theta = positive_product_bound(np.abs(H), np.r_[np.repeat(flow_errors, 4), np.zeros(4)])
    eta0 = exact(inverse_report["nominal_inverse_defect_bound"])
    eta = exact(inverse_report["inverse_contraction_bound"])
    data_selector = np.zeros(len(H))
    data_selector[-4:-2] = 1
    c = positive_product_bound(np.abs(H), data_selector)
    global_gain = upper(exact(float(max(c)))/(1-eta))
    i = int(np.flatnonzero(mesh == background.rs)[0])
    interface_gain = max(upper(exact(float(c[j]))+(eta0+exact(float(theta[j])))*exact(global_gain))
                         for j in range(4*i, 4*i+4))
    epsilon = exact(center_report["central_boundary_row_difference_bound"])
    scaled_state = state/WEIGHTS
    center_norm = max(upper(abs(exact(float(scaled_state[0, j])))+exact(float(point_errors[j]))) for j in range(4))
    perturbation = exact(global_gain)*epsilon
    assert perturbation < 1
    global_error = upper(perturbation*exact(center_norm)/(1-perturbation))
    interface_error = upper(exact(interface_gain)*epsilon*(exact(center_norm)+exact(global_error)))
    conversion = max(upper(exact(float(WEIGHTS[j]))/iv.mpf(v)) for j, v in enumerate(("1", "2", "0.4", "0.5")))
    interface_error = upper(exact(interface_error)*exact(conversion))
    assert exact(global_gain) < iv.mpf("1.001")
    assert exact(interface_gain) < iv.mpf("0.0039")
    assert exact(interface_error) < iv.mpf("0.000000025")
    return {"flow_bounds": flow_report, "reference_inverse_bounds": inverse_report,
            "central_data_global_node_gain_bound": global_gain,
            "central_data_interface_gain_bound": interface_gain,
            "central_reference_state_norm_bound": center_norm,
            "central_boundary_perturbation_contraction_bound": upper(perturbation),
            "central_replacement_global_node_error_bound": global_error,
            "central_replacement_interface_error_bound": interface_error}


def check_formulas(background):
    iv.dps = 30
    X = np.array([-.0004, -.0008, .0024, -.0044])
    p = np.array([1., -.603])
    s = 1e-6
    for dimension in (2, 4):
        xd = [Dual(v, np.eye(4, dtype=int)[j]) for j, v in enumerate(X)] if dimension == 4 else list(X)
        pd = [Dual(v, np.eye(2, dtype=int)[j]) for j, v in enumerate(p)] if dimension == 2 else list(p)
        result = integrands(exact(s), xd, pd, exact(.06), exact(.08), exact(np.pi))
        for column in range(dimension):
            xc, pc = X.astype(complex), p.astype(complex)
            (pc if dimension == 2 else xc)[column] += 1e-25j
            expected = np.imag(integrands(s, xc, pc, .06, .08, np.pi))/1e-25
            actual = np.array([float(value.d[column].mid) for value in result])
            assert np.max(np.abs(actual-expected)) < 1e-12
    leading_dual = leading([Dual(1, [1, 0]), Dual(float(background.sol(0)[3]), [0, 1])],
                           exact(.06), exact(.08), exact(np.pi))
    r = .002
    basis = np.array([[float(v.d[j].mid)*power+int((i, j) in ((0, 0), (3, 1)))
                       for j in range(2)] for i, (v, power) in enumerate(zip(leading_dual, [r*r, r, r*r, r*r]))])
    assert np.max(np.abs(basis-central_basis(background, r)[:4])) < 1e-14
    # Check the Volterra integrands against the independently implemented
    # original nonlinear Einstein--scalar ODE away from the singular center.
    from boson_star import equations, SINGULAR
    r = .001
    y = np.array([p[0]+r*r*X[0], r*X[1], r*r*X[2], p[1]+r*r*X[3]])
    g = integrands(r*r, X, p, .06, .08, np.pi)
    reconstructed = np.array([r*g[0], g[1]-2*X[1], r*(g[2]-X[2]), r*g[3]])
    expected = equations(np.array([r]), y[:, None], .06, .08)[:, 0]+SINGULAR@y/r
    assert np.max(np.abs(reconstructed-expected)) < 1e-14
    try:
        matrix_inverse2(np.array([[iv.mpf(1), iv.mpf(1)], [iv.mpf(1), iv.mpf(1)]], dtype=object))
    except AssertionError:
        pass
    else:
        raise AssertionError("A singular central chart was accepted.")
    print("PASS: interval derivatives, leading tangent, nonlinear ODE reconstruction, singular chart rejection", flush=True)


def check_local_integration(background, basis):
    """Independent floating-point consistency test, not used by the proof."""
    from scipy.integrate import solve_ivp
    from boson_star import equations, SINGULAR
    start, end = 1e-6, .002
    total_nodes = 0
    for offset in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        parameters = np.array([1., background.sol(0)[3]])+9e-5*np.array(offset)
        seed = leading([Dual(value, np.eye(2, dtype=int)[j]) for j, value in enumerate(parameters)],
                       exact(.06), exact(.08), exact(np.pi))
        powers = np.array([start*start, start, start*start, start*start])
        y = np.array([float(value.v.mid) for value in seed])*powers
        y[[0, 3]] += parameters
        tangent = np.array([[float(v.mid) for v in value.d] for value in seed])*powers[:, None]
        tangent[0, 0] += 1
        tangent[3, 1] += 1

        def rhs(r, state):
            y, C = state[:4], state[4:].reshape(4, 2)
            radius = np.array([r])
            value = equations(radius, y[:, None], .06, .08)[:, 0]+SINGULAR@y/r
            jacobian = []
            for column in range(4):
                shifted = y.astype(complex)
                shifted[column] += 1e-25j
                jacobian.append(equations(radius, shifted[:, None], .06, .08)[:, 0].imag/1e-25)
            matrix = np.array(jacobian).T+SINGULAR/r
            return np.r_[value, (matrix@C).ravel()]

        result = solve_ivp(rhs, (start, end), np.r_[y, tangent.ravel()], method="DOP853",
                           rtol=1e-10, atol=1e-14, max_step=end/16)
        assert result.success
        total_nodes += len(result.t)
        assert total_nodes <= 4000
        value = result.y[4:, -1].reshape(4, 2)
        assert all(float(basis[i, j].a) < value[i, j] < float(basis[i, j].b) for i in range(4) for j in range(2))
    print(f"PASS: independent local nonlinear and variational integration ({total_nodes} nodes total)", flush=True)
    return {"parameter_samples": 5, "total_radial_nodes": total_nodes,
            "scope": "Floating-point consistency only; existence and tangent enclosure use the Volterra contraction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-only", action="store_true", help="Check the central germ without global flow replay.")
    parser.add_argument("--digits", type=int, default=30)
    args = parser.parse_args()
    assert args.digits >= 25
    start, background = perf_counter(), Background()
    check_formulas(background)
    root = Path(__file__).parent/"data/s08a06"
    reference = root/"finite-static-reference.npz"
    with np.load(reference) as data:
        center, basis = certify_center(background, data["left_rows"], float(data["mesh"][0]), args.digits)
    result = {"scope": "Exact central nonlinear family; only central rows are replaced in the global reference comparison.",
              "central_germ": center, "mpmath_version": mpmath.__version__,
              "local_integration_crosscheck": check_local_integration(background, basis),
              "verified_decimal_bounds": {"central_contraction": "0.501", "central_row_difference": "0.0000034",
                  "nominal_trace_error": "0.000000000015"},
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "background_sha256": hashlib.sha256((root/"background.npz").read_bytes()).hexdigest()}
    if not args.local_only:
        result["boundary_comparison"] = compare_boundary(background, reference, center)
        result["verified_decimal_bounds"]["central_replacement_interface_error"] = "0.000000025"
    result["elapsed_seconds"] = perf_counter()-start
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
