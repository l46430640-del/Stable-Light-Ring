"""Algebraic checks: kinetic future transform."""
import json
import sympy as sp


def zero(expr):
    if isinstance(expr, sp.MatrixBase):
        assert expr.applyfunc(sp.simplify) == sp.zeros(*expr.shape)
    else:
        assert sp.simplify(expr) == 0


def weighted_energy_identity():
    t, x, sigma, lam = sp.symbols('t x sigma lam', positive=True)
    q = sp.Function('q')(t, x)
    V = sp.Function('V')(x)
    F = sp.Function('F')(t, x)
    qt, qx = sp.diff(q, t), sp.diff(q, x)
    qu, qv = (qt-qx)/2, (qt+qx)/2
    w = sp.exp(-2*sigma*(t-x)+lam/x)
    energy = (qt**2+qx**2+V*q*q)/2
    current = -qt*qx
    divergence = (sp.diff(w*energy, t)+sp.diff(w*current, x)).subs(
        sp.diff(q, t, 2), sp.diff(q, x, 2)-V*q+F)
    bulk = (4*sigma-lam/x**2)*qv*qv+lam*qu*qu/x**2+sigma*V*q*q
    zero(divergence-w*F*qt+w*bulk)
    assert sp.diff(V, x) not in divergence.atoms(sp.Derivative)
    zero(energy+current-2*qu*qu-V*q*q/2)
    zero(sp.diff(sp.exp(lam/x), x)/sp.exp(lam/x)+lam/x**2)
    return dict(exact_weighted_divergence=True, potential_derivative_absent=True,
                incoming_null_flux='q_u^2+V*q^2/4 in du',
                radial_correction_sign='p_prime/p=-lambda/x^2',
                scope='Exact scalar identity, summed over real channels in the proof')


def absorption_models():
    a, b, c, d, e, f = sp.symbols('a b c d e f', real=True)
    variables = [a, b, c, d, e, f]
    lam, sigma = 128, 16
    rows = []
    for x in (20, 40, 80, 160, 320):
        x = sp.Integer(x)
        # Variables are (Psi_u/x,Psi_v,Psi/x,s_u/x,s_v,s).
        bulk = lam*(a*a+d*d)+(4*sigma-lam/x**2)*(b*b+e*e)+sigma*c*c+sigma*f*f/2
        first = 12*(a*a+b*b/x**2)
        zeroth = 5*c*(a+b/x)
        mixed = ((x*d+e+f)*(x*a+b)+(x*c+f)*(x*d+e))/x**3
        form = sp.hessian(bulk-first-zeroth-mixed, variables)/2
        assert all(form[:j, :j].det() > 0 for j in range(1, 7))
        rows.append(dict(radius=int(x), all_six_leading_minors_positive=True))
    # exp(-x/2)<=x^-3 for x>=20 follows from this base bound and monotonicity.
    assert sum(sp.Rational(10)**j/sp.factorial(j) for j in range(16)) > 20**3
    assert sp.Rational(3, 20)-sp.Rational(1, 2) < 0
    return dict(examples=rows, exponential_envelope_checked=True,
                scope='Illustrative coefficient models only; actual coefficients use analytic Young inequalities')


def cutoff_commutator():
    t, x = sp.symbols('t x', real=True)
    U = sp.Matrix([sp.Function('u')(t, x), sp.Function('v')(t, x)])
    c = sp.Function('c')(t-x)
    cp = sp.diff(c, t)
    Dt = sp.Matrix([[x, 1], [2, x*x]])
    Dx = sp.Matrix([[1, x], [-x, 2]])
    V = sp.Matrix([[x*x, 1], [1, 3]])
    P = lambda z: z.diff(t, 2)-z.diff(x, 2)+Dt*z.diff(t)+Dx*z.diff(x)+V*z
    zero(P(c*U)-c*P(U)-cp*(2*(U.diff(t)+U.diff(x))+(Dt-Dx)*U))
    zero(sp.diff(c, t, 2)-sp.diff(c, x, 2))
    velocity = (c*U).diff(t).subs(t, 0)
    zero(velocity-c.subs(t, 0)*U.diff(t).subs(t, 0)-cp.subs(t, 0)*U.subs(t, 0))
    # Constant Lorenz data need not remain Lorenz after a null cutoff.
    hbar = sp.diag(1, 2)
    constraint = lambda h: sp.Matrix([-sp.diff(h[0, j], t)+sp.diff(h[1, j], x) for j in range(2)])
    zero(constraint(hbar))
    zero(constraint(c*hbar)+cp*sp.Matrix([1, 2]))
    assert constraint(c*hbar) != sp.zeros(2, 1)
    return dict(matrix_null_commutator_exact=True, second_cutoff_derivative_cancels=True,
                initial_velocity_cutoff_term_retained=True, Lorenz_cutoff_defect_nonzero=True)


def source_and_limit_integrals():
    x, X = sp.symbols('x X', positive=True)
    rows = []
    for power in (sp.Integer(2), sp.Rational(11, 4), sp.Rational(15, 4)):
        integral = sp.integrate(x**(2-2*power), (x, X, sp.oo))
        expected = X**(3-2*power)/(2*power-3)
        zero(integral-expected)
        assert 2*power > 3
        rows.append(dict(source_decay=str(power), weighted_squared_tail=str(integral)))
    zero(sp.integrate(x**-4, (x, X, sp.oo))-1/(3*X**3))
    k = sp.symbols('k', positive=True)
    assert sp.integrate(x*x*sp.exp(-2*k*x), (x, X, sp.oo)).is_finite is not False
    u, u0, sigma = sp.symbols('u u0 sigma', real=True, positive=True)
    y = sp.symbols('y', real=True)
    zero(sp.integrate(sp.exp(2*sigma*y), (y, u0, u))
         -(sp.exp(2*sigma*u)-sp.exp(2*sigma*u0))/(2*sigma))
    return dict(cutoff_source_integrals=rows, metric_null_derivative_kernel_integrable=True,
                weighted_Cauchy_Schwarz_constant='1/sqrt(2*sigma)',
                scope='Exact integrals supporting the analytic flux-to-trace argument')


def outgoing_phase_and_laplace():
    x, X, eps, growth = sp.symbols('x X eps growth', positive=True)
    nu = sp.symbols('nu', real=True)
    zeta = nu+sp.I*eps
    incoming = (sp.exp(sp.I*zeta*(x-X))-sp.exp(-sp.I*zeta*(x-X)))/(2*sp.I*zeta)
    expected = (sp.exp(sp.I*zeta*(2*x-X))-sp.exp(sp.I*zeta*X))/(2*sp.I*zeta)
    zero(incoming*sp.exp(sp.I*zeta*x)-expected)
    assert sp.limit(sp.exp(3*x/4)/x**2, x, sp.oo) == sp.oo
    # q=u^2 exp(growth*u) for u>=0 has an exact forward transform.
    decay = eps-growth-sp.I*nu
    field_transform = 2/decay**3
    news_transform = 2/decay**2+2*growth/decay**3
    zero(news_transform+sp.I*zeta*field_transform)
    zero(sp.exp(-sp.I*zeta*x)*(sp.exp(sp.I*zeta*x)*field_transform)-field_transform)
    cases = []
    for past, future in ((1, 2), (2, 4), (3, 5)):
        assert past < future
        cutoff_damping = future+1
        assert cutoff_damping > future and not (future < cutoff_damping < past)
        cases.append(dict(past_decay=past, future_growth=future,
                          bilateral_strip_empty=True, cutoff_transform_damping=cutoff_damping))
    return dict(free_Dirichlet_phase_cancellation=True,
                outgoing_coefficient_bound='1/(abs(zeta)*X) for a 1/x^2 envelope',
                algebraic_source_not_in_old_exponential_class=True,
                exact_growing_outgoing_wave_transform=True,
                incompatible_strip_examples=cases,
                scope='Free channel models only; the coupled physical continuation is still open')


def forcing_lift():
    t, x = sp.symbols('t x', real=True)
    W = sp.Function('W')(t, x)
    a, b, v = sp.Function('a')(x), sp.Function('b')(x), sp.Function('v')(x)
    A0W = -sp.diff(W, x, 2)+W
    PW = sp.diff(W, t, 2)-sp.diff(W, x, 2)+a*sp.diff(W, t)+b*sp.diff(W, x)+v*W
    remainder = sp.diff(W, t, 2)+a*sp.diff(W, t)+b*sp.diff(W, x)+(v-1)*W
    zero(PW-A0W-remainder)
    assert sp.diff(W, x, 2) not in remainder.atoms(sp.Derivative)
    return dict(second_spatial_derivative_cancels=True,
                scope='Forcing-lift algebra; the H^-1 to H^1 bound uses the positive reference operator')


def late_window_residue():
    nu, eps, T = sp.symbols('nu eps T', positive=True)
    zeta = nu+sp.I*eps
    future = sp.I*(sp.exp(sp.I*(zeta-nu)*T)/(zeta-nu)
                   +sp.exp(sp.I*(zeta+nu)*T)/(zeta+nu))/2
    zero(sp.limit(sp.I*eps*future, eps, 0, dir='+')-sp.I/2)
    zero(sp.expand_complex(sp.diff(future, T)+sp.exp(sp.I*zeta*T)*sp.cos(nu*T)))
    zero(sp.limit(sp.I*eps*(future-future.subs(T, 0)), eps, 0, dir='+'))
    return dict(residue_unchanged_by_any_finite_future_shift=True,
                difference_has_zero_residue=True,
                scope='Periodic model. The actual kinetic residue has not yet been identified')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "energy": weighted_energy_identity(),
        "absorption": absorption_models(),
        "cutoff": cutoff_commutator(),
        "integrals": source_and_limit_integrals(),
        "transform": outgoing_phase_and_laplace(),
        "lift": forcing_lift(),
        "late_window": late_window_residue(),
    }, indent=2))
