"""Positive orbital-plane distributions and their finite polar action kernels."""

from fractions import Fraction
from math import comb, factorial

import numpy as np
from numpy.polynomial import Legendre
from scipy.linalg import eigh


def concentrated_coefficients(power, at_poles=False):
    """Exact Legendre coefficients of normalized (1-z^2)^power or z^(2*power)."""
    if not isinstance(power, int) or not 0 <= power <= 6:
        raise ValueError("Use powers 0 through 6 in this bounded calculation.")
    monomials = {2*power: 1} if at_poles else {2*j: (-1)**j*comb(power, j) for j in range(power+1)}
    norm = 1/sum(Fraction(value, exponent+1) for exponent, value in monomials.items())
    result = [Fraction(0)]*(2*power+1)
    for degree in range(0, 2*power+1, 2):
        value = Fraction(0)
        for j in range(degree//2+1):
            coefficient = Fraction((-1)**j*factorial(2*degree-2*j),
                                   2**degree*factorial(j)*factorial(degree-j)*factorial(degree-2*j))
            value += sum(coefficient*Fraction(weight, degree-2*j+exponent+1)
                         for exponent, weight in monomials.items())
        result[degree] = (2*degree+1)*norm*value
    return result


def action_kernel(monopole, multipoles):
    """K(z)=D0+sum (2l+1)*Dl*Pl(z), with supplied even degrees only."""
    coefficients = np.zeros(max(multipoles, default=0)+1)
    coefficients[0] = monopole
    for degree, value in multipoles.items():
        if degree < 2 or degree % 2:
            raise ValueError("Use even degrees at least two.")
        coefficients[degree] = (2*degree+1)*value
    return Legendre(coefficients)


def kernel_minimum(kernel):
    roots = kernel.deriv().roots()
    stationary = [float(np.real(root)) for root in roots
                  if abs(np.imag(root)) < 1e-10 and -1 < np.real(root) < 1]
    points = np.r_[-1., stationary, 1.]
    values = kernel(points)
    index = int(np.argmin(values))
    return {"minimum": float(values[index]), "normal_cosine": float(abs(points[index])),
            "stationary_points": stationary, "values_at_candidates": values.tolist()}


def distribution_action(monopole, multipoles, power, at_poles=False):
    coefficients = concentrated_coefficients(power, at_poles)
    return float(monopole+sum(float(coefficients[degree])*multipoles[degree]
                             for degree in range(2, len(coefficients), 2)))


def square_distribution_minimum(kernel, maximum_degree=12, order=32):
    """Minimize over all normalized nonnegative even polynomials of degree <= N.

    Markov--Lukacs and parity reduce the full cone to four square families.
    The input kernel must contain every physical coefficient through N;
    an absent coefficient is interpreted as zero, not as an unknown response.
    """
    if (isinstance(maximum_degree, (bool, np.bool_))
            or not isinstance(maximum_degree, (int, np.integer))
            or maximum_degree not in range(2, 257, 2)
            or isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer))
            or not maximum_degree+1 <= order <= 512):
        raise ValueError("Use even N=2..256 and integer quadrature order N+1..512.")
    if (not isinstance(kernel, Legendre) or kernel.degree() > maximum_degree
            or not np.array_equal(kernel.domain, [-1, 1])
            or not np.array_equal(kernel.window, [-1, 1])
            or not np.isrealobj(kernel.coef) or not np.all(np.isfinite(kernel.coef))
            or np.any(kernel.coef[1::2] != 0)):
        raise ValueError("Use a finite real even Legendre kernel on [-1,1] of degree <= N.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    candidates = []
    for endpoint_weight in (0, 1):
        for parity in (0, 1):
            degrees = list(range(parity, maximum_degree//2-endpoint_weight+1, 2))
            if not degrees:
                continue
            basis = np.array([Legendre.basis(degree)(nodes) for degree in degrees])
            measure = weights*(1-nodes*nodes)**endpoint_weight/2
            norm = (basis*measure)@basis.T
            energy = (basis*(measure*kernel(nodes)))@basis.T
            values, vectors = eigh(energy, norm)
            vector = vectors[:, 0]
            density = (1-nodes*nodes)**endpoint_weight*(vector@basis)**2
            residual = np.linalg.norm(energy@vector-values[0]*(norm@vector))
            residual /= max(1., (np.linalg.norm(energy)+abs(values[0])*np.linalg.norm(norm))
                            *np.linalg.norm(vector))
            candidates.append({"action": float(values[0]), "endpoint_weight": endpoint_weight,
                               "degrees": degrees, "polynomial_coefficients": vector.tolist(),
                               "normalization": float(weights@density/2),
                               "mean_squared_normal_cosine": float(weights@(density*nodes*nodes)/2),
                               "eigenpair_residual": float(residual),
                               "gram_condition_number": float(np.linalg.cond(norm)),
                               "direct_action_difference": float(abs(weights@(density*kernel(nodes))/2-values[0]))})
    result = dict(min(candidates, key=lambda row: row["action"]))
    result["blocks"] = [{key: row[key] for key in ("action", "endpoint_weight", "degrees",
                         "eigenpair_residual", "gram_condition_number")} for row in candidates]
    return result


def complete_orientation_minimum(monopole, multipoles, maximum_degree, order):
    """Physical finite-degree minimum, rejecting unknown intermediate responses."""
    if (isinstance(maximum_degree, (bool, np.bool_))
            or not isinstance(maximum_degree, (int, np.integer))
            or maximum_degree not in range(2, 257, 2)):
        raise ValueError("Use an even maximum degree from 2 through 256.")
    if set(multipoles) != set(range(2, maximum_degree+1, 2)):
        raise ValueError("Every even response through the maximum degree is required.")
    return square_distribution_minimum(action_kernel(monopole, multipoles), maximum_degree, order)
