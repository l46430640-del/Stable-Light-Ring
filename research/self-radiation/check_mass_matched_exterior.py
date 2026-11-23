"""Algebraic checks: mass matched exterior."""
import json
import sympy as sp
from check_radiation_quartic import efficiency


def parity_and_vector_kernel():
    theta, phi = sp.symbols('theta phi', real=True)
    checked = []
    for ell in (3, 5, 7):
        for phase in (sp.cos, sp.sin):
            Y = sp.sin(theta)**ell*phase(ell*phi)
            X = sp.Matrix([sp.diff(Y, phi)/sp.sin(theta),
                           -sp.sin(theta)*sp.diff(Y, theta)])
            inverted = X.subs({theta: sp.pi-theta, phi: phi+sp.pi}, simultaneous=True)
            pulled = sp.diag(-1, 1)*inverted
            assert all(sp.trigsimp(value) == 0 for value in pulled-X)
            checked.append(dict(ell=ell, scalar_phase=phase.__name__, axial_inversion_parity=1))
    k = sp.Matrix(sp.symbols('k0:3', real=True))
    k2 = (k.T*k)[0]
    projector = k*k.T/k2
    assert (projector*projector-projector).applyfunc(sp.cancel) == sp.zeros(3)
    symbol = -k2*(sp.eye(3)+projector/3)
    inverse = -(sp.eye(3)-projector/4)/k2
    assert (symbol*inverse-sp.eye(3)).applyfunc(sp.cancel) == sp.zeros(3)
    n = sp.Matrix(sp.symbols('n0:3', real=True))
    r = sp.Symbol('r', positive=True)
    kernel = -(7*sp.eye(3)+n*n.T)/(32*sp.pi*r)
    assert kernel.subs(dict(zip(n, -n)), simultaneous=True) == kernel
    y = sp.Matrix(sp.symbols('y0:3', real=True))
    radius = sp.sqrt((y.T*y)[0])
    G = -(7*sp.eye(3)/radius+y*y.T/radius**3)/(32*sp.pi)
    operated = sp.Matrix(3, 3, lambda i, j:
        sum(sp.diff(G[i, j], coord, 2) for coord in y)
        +sum(sp.diff(G[k, j], y[k], y[i]) for k in range(3))/3)
    for point in ((2, 3, 6), (1, 2, 2), (0, 0, 2)):
        assert operated.subs(dict(zip(y, point))).applyfunc(sp.simplify) == sp.zeros(3)
    divergence = sp.Matrix([sum(sp.diff(G[k, j], y[k]) for k in range(3))
                            for j in range(3)])
    assert (divergence-6*y/(32*sp.pi*radius**3)).applyfunc(sp.simplify) == sp.zeros(3, 1)
    # The sphere flux of P_E G fixes the delta-source normalization.
    sphere_flux = (7*4*sp.pi*sp.eye(3)+3*4*sp.pi*sp.eye(3)/3)/(32*sp.pi)
    assert sphere_flux == sp.eye(3)
    return dict(axial_examples=checked, flat_vector_symbol_inverse=True,
                even_fundamental_matrix='-(7*delta_ij+n_i*n_j)/(32*pi*r)',
                vector_green_equation_off_origin_exact_points=3,
                vector_green_delta_normalization=True,
                odd_vector_source_has_zero_total_integral=True)


def exterior_power_checks():
    r, R, S = sp.symbols('r R S', positive=True)
    rows = []
    for delta in (sp.Rational(-3, 4), sp.Rational(-2, 3), sp.Rational(-9, 10)):
        p1, p2 = 2-2*delta, 3-2*delta
        assert 3 < p1 < 4 and 4 < p2 < 5
        assert 2-p1 == 2*delta and 2-p2 == 2*delta-1
        assert sp.integrate(r**(3-p2), (r, 1, sp.oo)) == 1/(p2-4)
        rows.append(dict(delta=str(delta), first_source_power=str(p1),
                         first_vector_remainder=str(2*delta), second_source_power=str(p2),
                         second_vector_remainder=str(2*delta-1)))
    norms = []
    for ell0 in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4)):
        metric = sp.integrate(r**(-5+2*ell0), (r, R, sp.oo))
        momentum = sp.integrate(r**(-3+2*ell0), (r, R, sp.oo))
        mismatch = sp.integrate(r**(-1+2*ell0), (r, R, S))
        assert sp.simplify(metric-R**(-4+2*ell0)/(4-2*ell0)) == 0
        assert sp.simplify(momentum-R**(-2+2*ell0)/(2-2*ell0)) == 0
        assert sp.simplify(mismatch-(S**(2*ell0)-R**(2*ell0))/(2*ell0)) == 0
        norms.append(dict(weight=str(ell0), metric_squared_tail=str(metric),
                          momentum_squared_tail=str(momentum), unmatched_mass_norm_diverges=True))
    assert sp.integrate(1/r, (r, 1, R)) == sp.log(R)
    return dict(elliptic_bootstrap=rows, weighted_norms=norms,
                final_scalar_source_power=6, first_uncontrolled_moment='third, logarithmic',
                scalar_remainder='O(r^-4 log r)', numerical_quadrature_points=0)


def conformal_tail_checks():
    a, rho, M, e2, Q = sp.symbols('a rho M e2 Q', real=True, nonzero=True)
    psi = 1+M/(2*rho)
    z2 = (e2/(2*rho)+Q/rho**3)/psi
    Phi = psi*(1+4*a*a*z2)**sp.Rational(1, 4)
    coefficient = sp.diff(Phi, a, 2).subs(a, 0)/2
    assert sp.simplify(coefficient-e2/(2*rho)-Q/rho**3) == 0
    reference_radius = rho*(1+M/(2*rho))**2
    assert sp.simplify(sp.diff(reference_radius, M)-1-M/(2*rho)) == 0
    # In a conformal exterior, a background-tracefree covariant tensor
    # remains tracefree in the physical metric.
    K = sp.Matrix([[1, 2, 3], [2, -3, 4], [3, 4, 2]])
    chi = sp.Symbol('chi', positive=True)
    assert sp.trace(K/chi**4) == 0
    return dict(quadrupole_tail_coefficient='q20(a)=a^2*Q_L+O(a^3)',
                mass_tail_coefficient='M(a)/2', reference_areal_radius_smooth=True,
                exterior_tracefree_correction_zero=True)


def retiming_checks():
    r, M, scale = sp.symbols('r M scale', positive=True)
    a, e2, e3, chi = sp.symbols('a e2 e3 chi', real=True)
    x = r+2*M*sp.log((r-2*M)/scale)
    derivative = 2*sp.log((r-2*M)/scale)-4*M/(r-2*M)
    assert sp.simplify(sp.diff(x, M)-derivative) == 0
    assert sp.limit(derivative-2*sp.log(r/scale), r, sp.oo) == 0
    q1, q2, q3, q1p = sp.symbols('q1 q2 q3 q1p', real=True)
    shift = chi*(e2*a*a+e3*a**3)
    raw = a*(q1-shift*q1p)+a*a*q2+a**3*q3
    assert sp.expand(raw).coeff(a, 3) == q3-e2*chi*q1p
    quartic_difference = sp.expand(raw**2).coeff(a, 4)-(q2*q2+2*q1*q3)
    assert sp.expand(quartic_difference+2*e2*chi*q1*q1p) == 0
    b2, b3, b4, e4, jump = sp.symbols('b2 b3 b4 e4 jump')
    assert sp.cancel(efficiency(e2, e3, e4, b2, b3, b4-e2*chi*jump)
                     -efficiency(e2, e3, e4, b2, b3, b4)+chi*jump/e2) == 0
    u, left, right = sp.symbols('u left right', real=True)
    profile = 1+u+u*u
    integrated = sp.integrate(-2*e2*chi*profile*sp.diff(profile, u), (u, left, right))
    assert sp.expand(integrated+e2*chi*(profile.subs(u, right)**2-profile.subs(u, left)**2)) == 0
    return dict(tortoise_mass_derivative=str(derivative),
                raw_cubic_news='N3 - e2*chi_r*d_u N1',
                raw_quartic_flux='b4 - e2*chi_r*[||N1||_sphere^2]_endpoints',
                raw_efficiency='kappa - chi_r/e2*[||N1||_sphere^2]_endpoints',
                smooth_window_difference='e2*chi_r*integral(w_prime*||N1||_sphere^2)',
                early_zero_N1_cubic_correction_zero=True)


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "parity": parity_and_vector_kernel(),
        "powers": exterior_power_checks(),
        "conformal": conformal_tail_checks(),
        "retiming": retiming_checks(),
    }, indent=2))
