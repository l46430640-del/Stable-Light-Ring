"""Exact identities and order counts in the four-variation proof (not a PDE solver)."""
from fractions import Fraction

import sympy as sp


def check_cones():
    r, q, lam, radius = sp.symbols("r q lam radius", positive=True)
    y = q / r
    side = q * (1 + y**lam / 2) - radius
    slope = lam*q*y**lam / (2*r*(1 + (1+lam)*y**lam/2))
    assert sp.simplify(sp.diff(side, r) + slope*sp.diff(side, q)) == 0
    # y<1 implies q>2D/3. On the smaller domain the larger trace exists.
    assert Fraction(2, 3) > Fraction(3, 5) > Fraction(1, 2)
    assert 64 > 48 and 48 * Fraction(2, 3) == 32
    assert 2*Fraction(1, 8)*32/3 == Fraction(8, 3) > 2
    # g^qq <= slope/2 after choosing the low ball; slope < lambda/2.
    s = sp.symbols("s", positive=True)
    assert sp.expand(s/2 - 2*sp.Rational(3, 4)*s + 2*s**2 + s*(1-2*s)) == 0
    assert Fraction(1, 3)/2 < Fraction(1, 2)


def check_weights_and_mass():
    y, lam, z, m = sp.symbols("y lam z m", positive=True)
    assert sp.integrate(y**(2*lam-1), (y, 0, 1)) == 1/(2*lam)
    assert Fraction(1, 3) - Fraction(1, 4) == Fraction(1, 12)
    # The old permitted layer exponent cannot use the old eight-order budget.
    assert 10 - Fraction(1, 3)/Fraction(1, 36) == -2
    v = sp.Function("v")(y)
    dy = lambda x: y*sp.diff(x, y)
    f = 1 - 2*m*z*y
    radial = -y*f*(dy(dy(v))+dy(v)) + 2*m*z*y**2*(dy(v)+v)
    assert sp.simplify(sp.diff(radial, m)-2*z*y**2*(dy(dy(v))+2*dy(v)+v)) == 0
    assert sp.diff(z**-2*y**-1, m) == 0


def check_orders():
    shift, loss = 112, 20
    theorem_order = 16*loss**2 + 43*loss + 24
    assert (shift+2*loss, shift+3*loss, theorem_order+shift+loss) == (152, 172, 7416)
    orders = [160-24*j for j in range(5)]
    assert orders[0] <= shift+3*loss and orders[-1] >= 48
    for previous, target in zip(orders, orders[1:]):
        source = target+2+8  # restriction and vacuum recovery
        assert source+5 == target+15  # high trial norm from weak matter estimate
        assert source+8 <= previous and target+15 <= shift+3*loss


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O.")
    check_cones()
    check_weights_and_mass()
    check_orders()
    print("Cone geometry, restriction weight, mass variation and four order counts verified.")
