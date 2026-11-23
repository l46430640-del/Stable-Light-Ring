"""Interval bounds for represented mean fields on the outer barrier.

This certifies inequalities for the saved piecewise polynomials and their
ODE defect. Bounds for the exact background and global matching data are
separate hypotheses, not inferred from the collocation residual.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path

import mpmath
from mpmath import iv
import numpy as np

from concentrated_axial import ConcentratedResponse
from static_transversality import static_matrix
from trapped_axial import Background


if not __debug__:
    raise RuntimeError("Certificate checks require assertions; run Python without -O.")


ORDER = 3
SCALES = ("1", "2", "0.4", "0.5")


def exact(value):
    """Enclose a rational or an exact binary float without decimal conversion."""
    if isinstance(value, Fraction):
        return iv.mpf(value.numerator) / value.denominator
    return iv.mpf(value)


def enclosure(left, right):
    return iv.mpf([exact(left).a, exact(right).b])


class Jet:
    """Interval Taylor coefficients, through degree three by default."""

    def __init__(self, value, coefficients=None, order=ORDER):
        self.c = ([exact(value)] + [iv.mpf(0) for _ in range(order)]
                  if coefficients is None else coefficients)
        self.order = len(self.c)-1

    @staticmethod
    def coerce(other, order=ORDER):
        if isinstance(other, Jet):
            if other.order != order:
                raise ValueError("Taylor operations require equal orders.")
            return other
        return Jet(other, order=order)

    def __add__(self, other):
        other = self.coerce(other, self.order)
        return Jet(0, [a + b for a, b in zip(self.c, other.c)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(0, [-a for a in self.c])

    def __sub__(self, other):
        return self + (-self.coerce(other, self.order))

    def __rsub__(self, other):
        return self.coerce(other, self.order) - self

    def __mul__(self, other):
        other = self.coerce(other, self.order)
        return Jet(0, [sum((self.c[k] * other.c[n-k] for k in range(n+1)), iv.mpf(0))
                       for n in range(self.order+1)])

    __rmul__ = __mul__

    def reciprocal(self):
        result = [1 / self.c[0]]
        for n in range(1, self.order+1):
            result.append(-sum((self.c[k] * result[n-k] for k in range(1, n+1)), iv.mpf(0)) / self.c[0])
        return Jet(0, result)

    def __truediv__(self, other):
        return self * self.coerce(other, self.order).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other, self.order) / self

    def exp(self):
        result = [iv.exp(self.c[0])]
        for n in range(1, self.order+1):
            result.append(sum((k * self.c[k] * result[n-k] for k in range(1, n+1)), iv.mpf(0)) / n)
        return Jet(0, result)


def polynomial(coefficients, argument, derivative=0):
    degree = len(coefficients) - 1
    result = Jet(0, order=argument.order)
    for j, coefficient in enumerate(coefficients[:len(coefficients)-derivative]):
        factor = 1
        for k in range(derivative):
            factor *= degree - j - k
        result = result * argument + exact(float(coefficient)) * factor
    return result


def rational_polynomial_check(coefficients, knots, width):
    """Independent Bernstein bounds using exact rational arithmetic only."""
    w = Fraction(float(width))
    end = Fraction(43, 50) / w
    scales = list(map(Fraction, SCALES))
    minimum_u, minimum_slope, jump_sum = None, None, Fraction(0)

    def bernstein_bounds(power, left, right):
        n = len(power) - 1
        transformed = [sum(power[k] * comb(k, j) * left**(k-j) * (right-left)**j
                           for k in range(j, n+1)) for j in range(n+1)]
        bernstein = [sum(transformed[j] * Fraction(comb(i, j), comb(n, j)) for j in range(i+1))
                     for i in range(n+1)]
        return min(bernstein), max(bernstein)

    for i in range(len(knots)-1):
        left, right = Fraction(float(knots[i])), Fraction(float(knots[i+1]))
        if left >= end:
            break
        width_t = min(right, end) - left
        power = [list(map(lambda x: Fraction(float(x)), coefficients[::-1, i, j])) for j in range(4)]
        bounds = [bernstein_bounds(p, 0, width_t) for p in power]
        derivative = [-(j+1) * power[3][j+1] / w for j in range(3)]
        slope, _ = bernstein_bounds(derivative, 0, width_t)
        assert bounds[2][0] > Fraction("0.24") and slope > Fraction("0.043")
        assert all(max(abs(lo), abs(hi)) < Fraction("1.8") * d for (lo, hi), d in zip(bounds, scales))
        minimum_u = bounds[2][0] if minimum_u is None else min(minimum_u, bounds[2][0])
        minimum_slope = slope if minimum_slope is None else min(minimum_slope, slope)
        if right < end:
            jumps = [abs(sum(p[j] * (right-left)**j for j in range(4))
                         - Fraction(float(coefficients[-1, i+1, k]))) / scales[k]
                     for k, p in enumerate(power)]
            jump_sum += max(jumps)
    assert jump_sum < Fraction("0.0000000001")
    return {"arithmetic": "exact rational Bernstein coefficients",
            "verified_bounds": {"response_u_lower": "0.24", "lapse_slope_lower": "0.043",
                                "weighted_state_upper": "1.8", "weighted_jump_sum_upper": "0.0000000001"},
            "rounded_diagnostics": {"u_lower": float(minimum_u), "slope_lower": float(minimum_slope),
                                    "jump_sum_upper": float(jump_sum)}}


def matrix_formula(r, y, central, scale, regular=False):
    """Four-state Jacobian; regular=True removes the explicit singular S/r."""
    central, scale, pi = Jet.coerce(central), Jet.coerce(scale), Jet(iv.pi)
    a, b, u, zeta = y
    f = 1 - 2 * u
    omega2 = (2 * zeta).exp()
    A02 = central * central
    s = A02 * a * a
    sigma2 = scale * scale
    V = s * (1 - 2 * s / sigma2) * (1 - 2 * s / sigma2)
    Vs = 1 - 8 * s / sigma2 + 12 * s * s / (sigma2 * sigma2)
    Vss = -8 / sigma2 + 24 * s / (sigma2 * sigma2)
    Va = 2 * A02 * a * Vs
    H = (Vs - omega2) * a - (2 * u / r - 4 * pi * r * V) * b
    rho_a = A02 * a * (omega2 + Vs)
    pressure_a = A02 * a * (omega2 - Vs)
    common_b = f * A02 * b
    common_u = -A02 * b * b
    common_zeta = omega2 * s
    pressure = (omega2 * s + f * A02 * b * b - V) / 2
    G = u / r + 4 * pi * r * pressure
    return [
        [Jet(0), Jet(1), Jet(0), Jet(0)],
        [(Vs - omega2 + 2 * s * Vss + 4 * pi * r * Va * b) / f,
         -(2 * u / r - 4 * pi * r * V) / f - (0 if regular else 2 / r),
         -2 * b / (r * f) + 2 * H / (f * f), -2 * omega2 * a / f],
        [4 * pi * r * rho_a, 4 * pi * r * common_b,
         4 * pi * r * common_u - (0 if regular else 1 / r), 4 * pi * r * common_zeta],
        [-4 * pi * r * pressure_a / f, -4 * pi * r * common_b / f,
         -((2 * u if regular else 1) / r + 4 * pi * r * common_u) / f - 2 * G / (f * f),
         -4 * pi * r * common_zeta / f],
    ]


def certify(background, coefficients, knots, location, width, digits=30):
    assert coefficients.shape == (4, len(knots)-1, 4)
    assert np.all(np.isfinite(coefficients)) and np.all(np.isfinite(knots))
    assert knots[0] == 0 and np.all(np.diff(knots) > 0) and width > 0 and location > 0
    assert Fraction(float(width)) * Fraction(float(knots[-1])) >= Fraction(43, 50)
    iv.dps = digits
    rs, w = Fraction(float(location)), Fraction(float(width))
    length = Fraction(43, 50)
    background_knots = list(map(lambda x: Fraction(float(x)), background.sol.x))
    response_knots = [rs + w * Fraction(float(t)) for t in knots]
    partition = sorted({rs, rs + length} | {x for x in background_knots + response_knots if rs < x < rs + length})
    if len(partition) > 4000:
        raise RuntimeError("Interval partition exceeds the retained 4000-point ceiling.")
    scales = list(map(iv.mpf, SCALES))
    central, scale = iv.mpf(str(background.central)), iv.mpf(str(background.scale))
    thresholds = {"response_u_lower": "0.24", "lapse_slope_lower": "0.043",
                  "weighted_state_upper": "1.8", "weighted_matrix_norm_upper": "2.5",
                  "weighted_residual_upper": "0.000001", "weighted_jump_sum_upper": "0.0000000001"}
    extrema = {"u_lower": float("inf"), "slope_lower": float("inf"),
               "state_upper": 0., "matrix_upper": 0., "residual_upper": 0.}

    def evaluate(center, bi, ri):
        r = Jet(center, [center, iv.mpf(1), iv.mpf(0), iv.mpf(0)])
        bg_argument = r - exact(background_knots[bi])
        bg_y = [polynomial(background.sol.c[:, bi, j], bg_argument) for j in range(4)]
        argument = (r - exact(rs)) / exact(w) - exact(float(knots[ri]))
        y = [polynomial(coefficients[:, ri, j], argument) for j in range(4)]
        dy = [polynomial(coefficients[:, ri, j], argument, 1) / exact(w) for j in range(4)]
        matrix = matrix_formula(r, bg_y, central, scale)
        residual = [(dy[i] - sum((matrix[i][j] * y[j] for j in range(4)), Jet(0))) / scales[i]
                    for i in range(4)]
        return y, dy, matrix, residual

    for left, right in zip(partition[:-1], partition[1:]):
        middle = (left + right) / 2
        bi = bisect_right(background_knots, middle) - 1
        ri = bisect_right(response_knots, middle) - 1
        y, dy, matrix, ranged_residual = evaluate(enclosure(left, right), bi, ri)
        _, _, _, center_residual = evaluate(exact(middle), bi, ri)
        delta = enclosure(-(right-left)/2, (right-left)/2)
        defect = [center_residual[i].c[0] + center_residual[i].c[1] * delta
                  + center_residual[i].c[2] * delta**2 + ranged_residual[i].c[3] * delta**3 for i in range(4)]
        row_norms = [sum((abs(matrix[i][j].c[0]) * scales[j] / scales[i] for j in range(4)), iv.mpf(0))
                     for i in range(4)]
        assert y[2].c[0] > iv.mpf(thresholds["response_u_lower"])
        assert -dy[3].c[0] > iv.mpf(thresholds["lapse_slope_lower"])
        assert all(abs(y[i].c[0]) / scales[i] < iv.mpf(thresholds["weighted_state_upper"]) for i in range(4))
        assert all(value < iv.mpf(thresholds["weighted_matrix_norm_upper"]) for value in row_norms)
        assert all(abs(value) < iv.mpf(thresholds["weighted_residual_upper"]) for value in defect)
        # These rounded diagnostics are not used in any certificate comparison.
        extrema["u_lower"] = min(extrema["u_lower"], float(y[2].c[0].a))
        extrema["slope_lower"] = min(extrema["slope_lower"], float((-dy[3].c[0]).a))
        extrema["state_upper"] = max(extrema["state_upper"], *(float((abs(y[i].c[0]) / scales[i]).b) for i in range(4)))
        extrema["matrix_upper"] = max(extrema["matrix_upper"], *(float(value.b) for value in row_norms))
        extrema["residual_upper"] = max(extrema["residual_upper"], *(float(abs(value).b) for value in defect))

    jump_sum = iv.mpf(0)
    for ri, node in enumerate(response_knots[1:-1], 1):
        if not rs < node < rs + length:
            continue
        step = exact(float(knots[ri])) - exact(float(knots[ri-1]))
        jumps = [abs(polynomial(coefficients[:, ri-1, j], Jet(step)).c[0] - exact(float(coefficients[-1, ri, j]))) / scales[j]
                 for j in range(4)]
        jump_sum += sum((value.b for value in jumps), iv.mpf(0))
    assert jump_sum < iv.mpf(thresholds["weighted_jump_sum_upper"])

    # A conditional enclosure for an exact response with nearby coefficients
    # and right interface state; it does not establish those two hypotheses.
    epsilon_A, epsilon_0 = iv.mpf("0.0001"), iv.mpf("0.0001")
    K = iv.mpf(thresholds["weighted_matrix_norm_upper"]) + epsilon_A
    d = iv.mpf(thresholds["weighted_residual_upper"]) + epsilon_A * iv.mpf(thresholds["weighted_state_upper"])
    growth = iv.exp(K * exact(length))
    error = growth * (epsilon_0 + iv.mpf(thresholds["weighted_jump_sum_upper"])) + (growth - 1) * d / K
    u_lower = iv.mpf(thresholds["response_u_lower"]) - scales[2] * error
    slope_lower = iv.mpf(thresholds["lapse_slope_lower"]) - scales[3] * (K * error + d)
    assert error < iv.mpf("0.00142")
    assert u_lower > iv.mpf("0.239") and slope_lower > iv.mpf("0.041")
    return {"status": "interval_inequalities_passed", "decimal_precision": digits,
            "scope": "Bounds for represented polynomials and their ODE defect. Exact-background coefficient and global interface enclosures remain hypotheses.",
            "interval_length": "0.86", "partition_points": len(partition),
            "reference_parameters": {"central_amplitude": str(background.central), "potential_scale": str(background.scale)},
            "weighted_coordinates": "Y_i / d_i", "weights": list(SCALES),
            "verified_decimal_bounds": thresholds, "rounded_interval_diagnostics": extrema,
            "rounded_jump_sum_bound": float(jump_sum.b),
            "conditional_exact_response": {"coefficient_error_assumption": "0.0001",
                                           "right_interface_error_assumption": "0.0001",
                                           "weighted_state_error_upper": "0.00142",
                                           "response_u_lower": "0.239", "lapse_slope_lower": "0.041"}}


def check_formula(background):
    iv.dps = 30
    for r in np.linspace(background.rs, background.rs + .86, 9):
        matrix = matrix_formula(Jet(exact(float(r))), [Jet(exact(float(v))) for v in background.sol(r)],
                                exact(background.central), exact(background.scale))
        values = np.array([[float(entry.c[0].mid) for entry in row] for row in matrix])
        expected = static_matrix(background, r)[0, :4, :4]
        assert np.max(np.abs(values - expected)) < 1e-12
    x = Jet(0, [iv.mpf(0), iv.mpf(1), iv.mpf(0), iv.mpf(0)])
    assert (x.exp().c[3] - iv.mpf(1) / 6).a <= 0 <= (x.exp().c[3] - iv.mpf(1) / 6).b
    assert all((1 / (1 - x)).c[j] == 1 for j in range(4))
    p = polynomial([2., -3., 5., 7.], x + 1)
    assert all(p.c[j] == value for j, value in enumerate((11, 5, 3, 2)))
    bad = np.zeros((4, 1, 4))
    bad[:, 0, 2] = [0, 2, -1.72, .3]
    bad[:, 0, 3] = [0, 0, -.05, 0]
    endpoint = Fraction(43, 50)
    assert Fraction(.3) > Fraction("0.24")
    assert 2 * endpoint**2 + Fraction(-1.72) * endpoint + Fraction(.3) > Fraction("0.24")
    try:
        rational_polynomial_check(bad, np.array([0., 1.]), 1.)
    except AssertionError:
        pass
    else:
        raise AssertionError("The rational check missed an interior sign violation with positive endpoints.")
    print("PASS: interval Taylor arithmetic and independent Jacobian formula")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--response", type=Path, help="Verify a saved response without a new BVP solve.")
    parser.add_argument("--digits", type=int, default=30, choices=(30, 50))
    args = parser.parse_args()
    background = Background()
    check_formula(background)
    if args.response is None:
        response = ConcentratedResponse(background)
        coefficients, knots = response.solution.sol.c[:, :, 5:9], response.solution.x
        location, width = response.location, response.widths[1]
    else:
        with np.load(args.response) as data:
            coefficients, knots = data["coefficients"].copy(), data["knots"].copy()
            location, width = float(data["location"]), float(data["width"])
    result = certify(background, coefficients, knots, location, width, digits=args.digits)
    result["rational_polynomial_check"] = rational_polynomial_check(coefficients, knots, width)
    result["software"] = {"mpmath": mpmath.__version__, "numpy": np.__version__}
    result["response_representation"] = {"coordinate": "t=(r-location)/width",
                                         "coefficient_order": "descending cubic powers of t-knots[i]",
                                         "columns": ["B/A0", "B_prime/A0", "x/r", "w/omega-v"],
                                         "paired_BVP_radial_points": int(2 * len(knots))}
    result["background_sha256"] = hashlib.sha256((Path(__file__).parent / "data/s08a06/background.npz").read_bytes()).hexdigest()
    if args.output_directory is not None:
        args.output_directory.mkdir(parents=True, exist_ok=True)
        path = args.output_directory / "barrier-response.npz"
        np.savez_compressed(path, coefficients=coefficients, knots=knots, location=location, width=width)
        result["response_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output_directory / "barrier-certificate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.response is not None:
        result["response_sha256"] = hashlib.sha256(args.response.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
