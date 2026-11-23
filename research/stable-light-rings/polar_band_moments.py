"""Positive Bernstein evaluation of compact orbit-band moments.

All interval operations enclose continuous input intervals. No energy
quadrature or differentiation of a moving square-root endpoint is used.
"""

from fractions import Fraction
from functools import lru_cache
from math import comb, isfinite

from mpmath import iv

from check_barrier_certificate import exact


def product(left, right):
    """Multiply Bernstein polynomials without converting to power form."""
    m, n = len(left)-1, len(right)-1
    return [sum((left[j]*right[k-j]*comb(m, j)*comb(n, k-j)
                 /comb(m+n, k) for j in range(max(0, k-n), min(m, k)+1)), iv.mpf(0))
            for k in range(m+n+1)]


def hull(values):
    return iv.mpf([min(v.a for v in values), max(v.b for v in values)])


def nonnegative(value):
    return iv.mpf([max(iv.mpf(0).a, value.a), max(iv.mpf(0).b, value.b)])


def _one_branch(x, a, b, below, derivative):
    width = b-a
    if below:
        gap = nonnegative(a-x)
        left, right = iv.sqrt(gap), iv.sqrt(nonnegative(b-x))
        total = left+right
        length = width/total
        ratio = lambda d, w: iv.sqrt(d)/(iv.sqrt(d)+iv.sqrt(d+w))
        middle = iv.mpf([ratio(gap.a, width.b).a, ratio(gap.b, width.a).b])
        p, q = [iv.mpf(0), middle, iv.mpf(1)], [iv.mpf(1), 1-middle, iv.mpf(0)]
    else:
        gap = nonnegative(b-x)
        left, right = iv.mpf(0), iv.sqrt(gap)
        length = right
        t0, s0 = nonnegative(x-a)/width, gap/width
        t = iv.mpf([min(iv.mpf(1).b, t0.a), min(iv.mpf(1).b, t0.b)])
        s = iv.mpf([min(iv.mpf(1).b, s0.a), min(iv.mpf(1).b, s0.b)])
        p, q = [t, t, iv.mpf(1)], [s, s, iv.mpf(0)]
    p2, q2 = product(p, p), product(q, q)
    p4, q4 = product(p2, p2), product(q2, q2)
    shape = product(p4, q4)
    J0 = 512*length*sum(shape, iv.mpf(0))/17
    J1 = 512*length*sum(product(shape, [left*left, left*right, right*right]), iv.mpf(0))/19
    result = [J0, J1]
    if derivative:
        positive = product(product(p2, p), q4)
        negative = product(p4, product(q2, q))
        result.append(2048*length/width*(sum(positive, iv.mpf(0))-sum(negative, iv.mpf(0)))/15)
    return result


def _raw_band_moments(x, lower=".998", upper=".999", derivative=False):
    """Enclose J_j(x)=integral F(e)*(e-x)^(j-1/2) de, j=0,1.

    F(e)=[4t(1-t)]^4, t=(e-lower)/(upper-lower), with integration
    from max(x,lower) to upper. A third result optionally encloses J0'.
    Band endpoints are exact scalars; x may straddle either endpoint.
    """
    try:
        a_exact, b_exact = Fraction(lower), Fraction(upper)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Specify finite scalar band endpoints, not uncertain bands.") from error
    a, b, x = exact(a_exact), exact(b_exact), exact(x)
    if a_exact >= b_exact or not a.b < b.a or not all(isfinite(float(v)) for v in (x.a, x.b)):
        raise ValueError("Use a finite argument and an ordered band resolved at the working precision.")
    pieces = []
    if x.a < a.b:
        # Slight overlap covers rounding of rational branch boundaries.
        below_x = iv.mpf([x.a, min(x.b, a.b)])
        pieces.append(_one_branch(below_x, a, b, True, derivative))
    if x.b >= a.a and x.a <= b.b:
        middle = iv.mpf([max(x.a, a.a), min(x.b, b.b)])
        # Use an outward scalar endpoint hull through nonnegative gaps.
        pieces.append(_one_branch(middle, a, b, False, derivative))
    if x.b >= b.a:
        pieces.append([iv.mpf(0)]*(3 if derivative else 2))
    return [hull([row[j] for row in pieces]) for j in range(3 if derivative else 2)]


@lru_cache(maxsize=1)
def maximum_fraction():
    """The unique positive zero of the derivative polynomial, rationally bracketed."""
    polynomial = lambda z: 1792*z**4-448*z**3-48*z**2-8*z-1
    left, right = Fraction(0), Fraction(1)
    for _ in range(80):
        middle = (left+right)/2
        if polynomial(middle) < 0:
            left = middle
        else:
            right = middle
    assert polynomial(left) < 0 < polynomial(right)
    return left, right


def band_moments(x, lower=".998", upper=".999", derivative=False):
    """Sharp range enclosure using unimodality of J0 and monotonicity of J1."""
    x = exact(x)
    left = _raw_band_moments(x.a, lower, upper)
    right = _raw_band_moments(x.b, lower, upper)
    z0, z1 = maximum_fraction()
    a, b = exact(Fraction(lower)), exact(Fraction(upper))
    peak = a+(b-a)*iv.mpf([exact(z0).a, exact(z1).b])
    candidates = [left[0], right[0]]
    if x.a <= peak.b and x.b >= peak.a:
        candidates.append(_raw_band_moments(peak, lower, upper)[0])
    result = [iv.mpf([min(left[0].a, right[0].a), max(v.b for v in candidates)]),
              iv.mpf([right[1].a, left[1].b])]
    if derivative:
        pieces = []
        if x.a < a.b:
            # J0'' > 0 below the band; endpoint values enclose J0'.
            lo = _one_branch(x.a, a, b, True, True)[2]
            hi = _one_branch(min(x.b, a.b), a, b, True, True)[2]
            pieces.append(iv.mpf([lo.a, hi.b]))
        if x.b >= a.a and x.a <= b.b:
            z = nonnegative(iv.mpf([max(x.a, a.a), min(x.b, b.b)])-a)/(b-a)
            z = iv.mpf([min(z.a, iv.mpf(1).a), min(z.b, iv.mpf(1).b)])
            s = nonnegative(1-z)
            polynomial = (((1792*z-448)*z-48)*z-8)*z-1
            pieces.append(-iv.mpf(32768)/45045/iv.sqrt(b-a)*s**3*iv.sqrt(s)*polynomial)
        if x.b >= b.a:
            pieces.append(iv.mpf(0))
        result.append(hull(pieces))
    return result
