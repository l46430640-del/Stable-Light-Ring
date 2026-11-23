"""Power-series coefficients for the prepared radiation."""
import sympy as sp


def efficiency(e2, e3, e4, b2, b3, b4):
    return (b4 - b2 * e4 / e2) / e2**2 - 3 * e3 * (b3 - b2 * e3 / e2) / (2 * e2**3)


def laurent(expr, variables):
    result = {}
    for term in sp.expand(expr).as_ordered_terms():
        powers = term.as_powers_dict()
        charge = tuple(int(powers.get(z, 0)) for z in variables)
        coefficient = sp.simplify(term / sp.prod(z**q for z, q in zip(variables, charge)))
        result[charge] = result.get(charge, 0) + coefficient
    return {q: sp.simplify(c) for q, c in result.items() if c != 0}
