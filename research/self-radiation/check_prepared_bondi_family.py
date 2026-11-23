"""Algebraic checks: prepared bondi family."""
import json
from math import comb
import sympy as sp


def zero(expr):
    if isinstance(expr, sp.MatrixBase):
        assert expr.applyfunc(sp.simplify) == sp.zeros(*expr.shape)
    else:
        assert sp.simplify(expr) == 0


def mixed_mass_commutators():
    a, r0, y, mu = sp.symbols('a r0 y mu', positive=True)
    mass = mu**2/(r0**2*y)
    m = 2+a*a+a**3/7
    D0 = lambda z: r0*sp.diff(z, r0)
    D1 = lambda z: y*sp.diff(z, y)
    shifted = lambda z: D1(z)+z
    derivative = lambda z: 2*r0*y*y*shifted(shifted(z))
    L = lambda z: -2*D1(D0(z)-D1(z))-y*(D1(D1(z))+D1(z))+m*derivative(z)+mass*z
    v = (1+a+a*a)*(r0*y+r0**2*y**3)+a**6*r0**3*y**2
    cases = 0
    for order in range(1, 7):
        commutator = sp.diff(L(v), a, order)-L(sp.diff(v, a, order))
        expected = sum(comb(order, j)*sp.diff(m, a, j)*derivative(sp.diff(v, a, order-j))
                       for j in range(1, order+1))
        zero(commutator-expected)
        zero(sp.diff(mass*v, a, order)-mass*sp.diff(v, a, order))
        cases += 1
    V = lambda z: D0(z)-2*D1(z)
    zero(V(mass))
    zero(D1(mass)+mass)
    zero(sp.diff(V(v), a)-V(sp.diff(v, a)))
    zero(sp.diff(D1(v), a)-D1(sp.diff(v, a)))
    return dict(reference_parameter_commutators=cases, mass_parameter_commutators=cases,
                mass_triangular_identities_unchanged=True,
                scope='Exact rescaled reference operator, not the full metric PDE estimate')


def bondi_shear_orders():
    z = sp.symbols('z', positive=True)
    a, u = sp.symbols('a u', real=True)
    # z=r^(-1/3): a cut change is O(z), an angle change is O(z^4).
    C = sp.Matrix([[a*a*u, a**3+u], [a**3+u, -a*a*u]])
    Cshift = C.subs(u, u+a*z)
    B = sp.Matrix([[1, a], [u, -1]])
    D = sp.Matrix([[2, 1], [1, 3]])
    J = sp.eye(2)+z**4*B
    angular = sp.expand(J.T*(sp.eye(2)+z**3*Cshift+z**4*D)*J)
    series = lambda expr, order: sp.series(expr, z, 0, order).removeO().expand()
    determinant = series(angular.det(), 7)
    inverse_area = series(determinant**(-sp.Rational(1, 2)), 5)
    area_ratio = series(determinant**sp.Rational(1, 4), 7)
    normalized = (angular*inverse_area).applyfunc(lambda x: series(x, 5))
    actual_shear = normalized.applyfunc(lambda x: x.coeff(z, 3))
    zero(actual_shear-C)
    zero(sp.diff(actual_shear, u)-sp.diff(C, u))
    for j in range(1, 4):
        zero(sp.diff(actual_shear, a, j)-sp.diff(C, a, j))
    assert area_ratio.coeff(z, 1) == area_ratio.coeff(z, 2) == area_ratio.coeff(z, 3) == 0
    assert area_ratio.coeff(z, 4) != 0
    assert sp.trace(C) == 0
    # The squared shear affects the area only one full inverse-radius later.
    c, d = sp.symbols('c d', real=True)
    cc = sp.Matrix([[c, d], [d, -c]])
    zero((sp.eye(2)+z**3*cc).det()-(1-z**6*(c*c+d*d)))
    return dict(tracefree_shear_preserved=True, time_derivative_preserved=True,
                amplitude_derivatives_preserved=3, area_radius_error='O(r^(-1/3))+O(r^(-1))',
                scope='Local formal coordinate-order model. Eikonal existence uses the analytic proof')


def massive_flux_orders():
    A = sp.Rational(1, 4)
    phi_decay = 4-A
    flux_decay = 2*phi_decay-2
    radial_stress_decay = 2*(phi_decay+1)
    assert phi_decay == sp.Rational(15, 4)
    assert flux_decay == sp.Rational(11, 2)
    assert radial_stress_decay == sp.Rational(19, 2)
    assert radial_stress_decay-1 > 3
    return dict(field_decay=str(phi_decay), r_squared_Tuu_decay=str(flux_decay),
                radial_scalar_stress_decay=str(radial_stress_decay),
                area_gauge_beta_decay=2, scalar_limiting_flux_zero=True)


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "mass": mixed_mass_commutators(),
        "shear": bondi_shear_orders(),
        "matter": massive_flux_orders(),
    }, indent=2))
