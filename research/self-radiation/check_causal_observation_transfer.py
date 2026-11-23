"""Algebraic checks: causal observation transfer."""
import json
import sympy as sp


def zero(value):
    if isinstance(value, sp.MatrixBase):
        assert value.applyfunc(sp.simplify) == sp.zeros(*value.shape)
    else:
        assert sp.simplify(value) == 0


def causal_initial_polynomial():
    s, omega = sp.symbols('s omega', positive=True)
    J = sp.Matrix([[0, -1], [1, 0]])
    D = 2*omega*J
    K = sp.Matrix([[3, sp.Rational(1, 5)], [sp.Rational(1, 5), 7]])
    u0 = sp.Matrix(sp.symbols('u00 u01'))
    u1 = sp.Matrix(sp.symbols('u10 u11'))
    u2 = sp.Matrix(sp.symbols('u20 u21'))
    uhat = u0/s+u1/s**2+2*u2/s**3
    fhat = 2*u2/s+D*(u1/s+2*u2/s**2)+K*uhat
    polynomial = s**2*sp.eye(2)+s*D+K
    zero(polynomial*uhat-fhat-(s*sp.eye(2)+D)*u0-u1)
    zero(polynomial*uhat-fhat-s*u0-u1-D*u0)
    assert K*D != D*K
    assert D*sp.Matrix([1, 2]) != sp.zeros(2, 1)
    t, tau = sp.symbols('t tau', real=True)
    eta = sp.Function('eta')
    zero(-sp.diff(eta(t-tau), tau).subs(tau, 0)-sp.diff(eta(t), t))
    return dict(noncommuting_gyroscopic_system=True,
                laplace_initial_term='u1+(s*I+D_t)*u0',
                fourier_convention='s=-i*zeta',
                omitted_D_t_u0_is_nonzero=True,
                delta_prime_convolution='eta_prime(t)*u0',
                scope='Polynomial vector test of the exact distributional initial identity')


def null_cutoff_initial_terms():
    c0, c1 = sp.symbols('c0 c1')
    D = sp.Matrix([[0, -2], [2, 0]])
    u0 = sp.Matrix(sp.symbols('a0 a1'))
    u1 = sp.Matrix(sp.symbols('b0 b1'))
    cutoff_first = c0*u1+c1*u0+D*(c0*u0)
    causal_first = c0*(u1+D*u0)-c1*u0+2*c1*u0
    zero(cutoff_first-causal_first)
    zero(cutoff_first-c0*(u1+D*u0)-c1*u0)
    # First-order gauge product rule has only a delta, not a delta-prime.
    t = sp.symbols('t', real=True)
    c, h = sp.Function('c')(t), sp.Function('h')(t)
    zero(sp.diff(c*h, t)-sp.diff(c, t)*h-c*sp.diff(h, t))
    return dict(delta_prime_coefficient='c0*u0',
                delta_coefficient='c0*(u1+D_t*u0)+c1*u0',
                cutoff_delta_prime_product=-1, principal_commutator_delta=2,
                net_c1_coefficient=1,
                gauge_initial_term='delta(t)*c0*grad(t)^a*bar_h0_ab',
                scope='Distributional product coefficients; stellar gauge propagation is a separate input')


def coupled_causal_residue():
    z, y = sp.symbols('z y')
    J = sp.Matrix([[0, -1], [1, 0]])
    K = sp.Matrix([[-1, sp.Rational(1, 5)], [sp.Rational(1, 5), 4]])
    u0, u1, force = sp.Matrix([2, -3]), sp.Matrix([5, 7]), sp.Matrix([1, 2])
    records = []
    for omega, nu in ((sp.Rational(1, 5), sp.Rational(1, 7)),
                      (sp.Rational(1, 3), sp.Rational(2, 9)),
                      (sp.Rational(2, 5), sp.Rational(1, 4))):
        D = 2*omega*J
        P = K-z*z*sp.eye(2)-sp.I*z*D
        assert P.subs(z, nu).det() != 0
        source = sp.I*z*force/(z*z-nu*nu)
        field = P.inv()*(source+u1+(D-sp.I*z*sp.eye(2))*u0)
        news = -u0-sp.I*z*field
        residue = ((z-nu)*news).applyfunc(lambda entry: sp.cancel(entry).subs(z, nu))
        expected = nu*P.subs(z, nu).inv()*force/2
        zero(residue-expected)
        delta0, delta1 = sp.Matrix([11, -13]), sp.Matrix([-17, 19])
        change = -delta0-sp.I*z*P.inv()*(delta1+(D-sp.I*z*sp.eye(2))*delta0)
        zero(((z-nu)*change).subs(z, nu))
        assert residue != sp.zeros(2, 1)
        # det(P(i*gamma)) is quadratic in y=gamma^2 with negative constant.
        growth_polynomial = y*y+(sp.trace(K)+4*omega**2)*y+K.det()
        gamma = sp.symbols('gamma', real=True)
        zero(P.subs(z, sp.I*gamma).det()-growth_polynomial.subs(y, gamma**2))
        assert sp.Poly(growth_polynomial, y).LC() == 1
        assert growth_polynomial.subs(y, 0) < 0
        records.append(dict(omega=str(omega), frequency=str(nu),
                            news_residue=[str(entry) for entry in residue],
                            independent_of_initial_data=True,
                            unstable_positive_growth_root_exists=True))
    return dict(cases=records,
                residue_formula='nu*P(nu)^(-1)*force/2',
                convergence_half_plane_requires_growth_bound=True,
                scope='Coupled gyroscopic ODE with unstable poles, not a stellar radiation calculation')


def observation_margin():
    records = []
    for L in (8, 16, 32, 64, 128, 256):
        c = 1-sp.Rational(1, L+2)
        compact, exterior = sp.Rational(1, 4*L), sp.Rational(1, L**2)
        mean = c-compact+exterior
        error = abs(c-1)+compact+exterior
        observations = [mean+L, mean-L]
        zero(sum(observations)/2-mean)
        assert abs(mean-1) <= error < sp.Rational(1, 2)
        assert max(abs(v) for v in observations) >= abs(mean) >= sp.Rational(1, 2)
        records.append(dict(carrier=L, margin=str(error),
                            phase_observations=[str(v) for v in observations]))
    zero(sp.Rational(3, 10)*sp.Rational(1, 2)**2-sp.Rational(3, 40))
    return dict(cases=records, quadrupole_lower_bound_factor='3/40',
                physical_numerical_threshold_certified=False,
                scope='Exact triangle inequality and two-phase selection models; no physical constants assigned')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "initial": causal_initial_polynomial(),
        "cutoff": null_cutoff_initial_terms(),
        "residue": coupled_causal_residue(),
        "margin": observation_margin(),
    }, indent=2))
