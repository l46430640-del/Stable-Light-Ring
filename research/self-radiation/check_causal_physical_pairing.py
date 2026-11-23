"""Algebraic checks: causal physical pairing."""
import json
import sympy as sp


def zero(value):
    if isinstance(value, sp.MatrixBase):
        assert value.applyfunc(sp.simplify) == sp.zeros(*value.shape)
    else:
        assert sp.simplify(value) == 0


def gauge_integration_identity():
    coordinates = sp.symbols('t x y z', real=True)
    metric = sp.diag(-1, 1, 1, 1)
    k = sp.Matrix(4, 4, lambda i, j:
                  sp.Function(f'k{min(i,j)}{max(i,j)}')(*coordinates))
    C = sp.Matrix([sp.Function(f'C{i}')(*coordinates) for i in range(4)])
    psi = sp.Matrix([sp.Function(f'psi{i}')(*coordinates) for i in range(2)])
    q = [sp.Function(f'q{i}')(*coordinates) for i in range(2)]
    trace = sp.trace(metric*k)
    bar_upper = metric*(k-metric*trace/2)*metric
    upper = metric*k*metric
    dC = sp.Matrix(4, 4, lambda i, j:
                   (sp.diff(C[j], coordinates[i])+sp.diff(C[i], coordinates[j]))/2)
    tr_dC = sp.trace(metric*dC)
    Gg = -(dC-metric*tr_dC/2)/(16*sp.pi)
    scalar_defect = -sum(psi[I]*metric[a, a]*sp.diff(q[I], coordinates[a])*C[a]
                         for I in range(2) for a in range(4))
    full = sum(upper[i, j]*Gg[i, j] for i in range(4) for j in range(4))+scalar_defect
    boundary = sp.Matrix([sum(bar_upper[a, b]*C[b] for b in range(4))/(16*sp.pi)
                          for a in range(4)])
    div_boundary = sum(sp.diff(boundary[a], coordinates[a]) for a in range(4))
    Ck_upper = sp.Matrix([sum(sp.diff(bar_upper[a, b], coordinates[a])
                             for a in range(4)) for b in range(4)])
    volume = sum(Ck_upper[b]*C[b] for b in range(4))/(16*sp.pi)+scalar_defect
    zero(full+div_boundary-volume)
    # A Lorenz test still has the scalar term: chi=1, q=t, C_t=1.
    missing_scalar_example = -metric[0, 0]
    assert missing_scalar_example == 1
    return dict(flat_covariant_product_rule_exact=True, metric_boundary_sign='plus',
                remaining_scalar_volume_sign='minus chi*dq*C',
                omitted_scalar_term_counterexample=int(missing_scalar_example),
                scope='Exact tensor product rule in a flat chart; covariance gives the curved density identity')


def radiative_gauge_quotient():
    metric = sp.diag(-1, 1, 1, 1)
    p = sp.Matrix([-1, 0, 0, 1])
    p_up = metric*p
    b00, b01, b02, b11, b12, b22 = sp.symbols('b00 b01 b02 b11 b12 b22')
    hbar = sp.Matrix([[b00, b01, b02, -b00],
                      [b01, b11, b12, -b01],
                      [b02, b12, b22, -b02],
                      [-b00, -b01, -b02, b00]])
    zero(p_up.T*hbar)
    h = hbar-metric*sp.trace(metric*hbar)/2
    xi = sp.Matrix(sp.symbols('xi0:4'))
    pure_gauge = p*xi.T+xi*p.T
    bilinear = lambda a, b: sp.trace(metric*a*metric*b)-sp.trace(metric*a)*sp.trace(metric*b)/2
    zero(bilinear(pure_gauge, h))
    components = lambda tensor: sp.Matrix([tensor[i, j] for i in range(4) for j in range(i, 4)])
    parameters = [b00, b01, b02, b11, b12, b22]
    assert components(h).jacobian(parameters).rank() == 6
    assert components(pure_gauge).jacobian(xi).rank() == 4
    basis = [h.diff(parameter) for parameter in parameters]
    gram = sp.Matrix(6, 6, lambda i, j: bilinear(basis[i], basis[j]))
    assert gram.rank() == 2
    zero(p_up.T*(pure_gauge-metric*sp.trace(metric*pure_gauge)/2))
    plus = sp.diag(0, 1, -1, 0)
    cross = sp.zeros(4)
    cross[1, 2] = cross[2, 1] = 1
    zero(bilinear(plus, plus)-2)
    zero(bilinear(cross, cross)-2)
    zero(bilinear(plus, cross))
    r = sp.symbols('r', positive=True)
    G = sp.diag(1, 2, 3, 4)
    zero(p_up.T*(hbar/r+G/r**2)-(p_up.T*G)/r**2)
    assert p_up.T*G != sp.zeros(1, 4)
    return dict(transverse_polarizations=6, residual_gauge_directions=4,
                transverse_radiative_directions=2, pure_gauge_pairs_to_zero=True,
                polarization_gram_matrix=[[2, 0], [0, 2]],
                cutoff_defect_order='r^-2 after the past longitudinal constant vanishes',
                scope='Leading asymptotically flat algebra; past decay and radial remainder are analytical inputs')


def constraint_minor():
    r, f, A, Ap, omega, nu, s, n, L, C = sp.symbols('r f A Ap omega nu s n L C')
    H, Hp, K, T, U, Up, W, Wp, Vs = sp.symbols('H Hp K T U Up W Wp Vs')
    z = 2*nu**2*r**2/s
    B = 1/r-n
    RK = (2*r*r*f*n*Hp+(4*r*r*f*n*n+L+2-2*f
          -8*sp.pi*r*r*(omega**2*A*A/s-f*Ap*Ap))*H
          -32*sp.pi*r*r*f*(n+1/r)*Ap*U
          -16*sp.pi*r*r*(omega**2/s-Vs)*A*U
          -16*sp.pi*r*r*f*Ap*Up-16*sp.pi*sp.I*nu*r*r*omega*A*W/s)
    RT = -16*sp.pi*omega*r*r*(A*Wp-Ap*W)-2*sp.I*nu*r*r*(Hp+(2*n-1/r)*H-8*sp.pi*Ap*U)
    E = sp.Matrix([(L-z)*K+2*sp.I*nu*r*r*f*B*T/s-RK,
                   2*sp.I*nu*r*r*B*K+(C-z)*T-RT])
    minor = E.jacobian([Up, Wp]).det()
    expected = (16*sp.pi*r*r*f*Ap)*(16*sp.pi*omega*r*r*A)
    zero(minor-expected)
    zero(sp.diff(minor, nu))
    return dict(minor=str(sp.factor(minor)), frequency_independent=True,
                condition='A*A_prime*omega*f nonzero on one fixed matching sphere',
                scope='Exact reconstruction-constraint minor; no numerical division by the stellar tail')


def massive_exponent_gap():
    a, eps, y = sp.symbols('a eps y', real=True)
    polynomial = y*y-(1-a*a+eps*eps)*y-a*a*eps*eps
    zero(polynomial.subs(y, eps*eps)+eps*eps)
    return dict(real_part_squared_polynomial=str(polynomial),
                value_at_epsilon_squared='-epsilon^2',
                conclusion='Re sqrt(1-(a+i epsilon)^2)>epsilon for epsilon>0',
                reason='The quadratic has one positive root; its value at epsilon^2 is strictly negative')


def symplectic_matching_model():
    z, kp, km = sp.symbols('z kp km', nonzero=True)
    eye, nil = sp.eye(3), sp.zeros(3)
    J = nil.row_join(eye).col_join((-eye).row_join(nil))
    B = sp.Matrix([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    S = eye.row_join(nil).col_join(B.row_join(eye))
    zero(S.T*J*S-J)
    center = S*eye.col_join(nil)
    outgoing = S*eye.col_join(sp.diag(sp.I*z, -kp, -km))
    matching = center.row_join(outgoing)
    row = matching.inv()[3, :]
    zero(row*center)
    zero(row*outgoing-sp.Matrix([[1, 0, 0]]))
    test = J*row.T
    zero(test.T*J-row)
    zero(center.T*J*center)
    zero(sp.Matrix.hstack(center, test).rank()-3)
    return dict(coupled_canonical_change=True, central_space_lagrangian=True,
                outgoing_dual_normalizations=[1, 0, 0], test_is_regular=True,
                scope='Six-dimensional canonical model only; nondegeneracy of the physical pairing is proved in the document')


def tail_pairing_integrals():
    x, X, eps = sp.symbols('x X eps', positive=True)
    nu = sp.symbols('nu', real=True)
    zeta = nu+sp.I*eps
    rows = []
    for exponent in (sp.Integer(3), sp.Rational(15, 4)):
        integrand = x*x*(sp.exp(-sp.I*zeta*x)/x)*(sp.exp(sp.I*zeta*x)/x**exponent)
        zero(integrand-x**(1-exponent))
        value = sp.integrate(sp.simplify(integrand), (x, X, sp.oo))
        zero(value-X**(2-exponent)/(exponent-2))
        rows.append(dict(physical_source_decay=str(exponent), paired_integral=str(value)))
    boundary = x*x*(sp.exp(-sp.I*zeta*x)/x)*(sp.exp(sp.I*zeta*x)/x**2)
    zero(boundary-1/x)
    assert sp.limit(boundary, x, sp.oo) == 0
    return dict(paired_tails=rows, gauge_boundary='1/x -> 0',
                exponential_cancellation_exact=True,
                scope='Exact majorant integrals, not an evaluation of the actual stellar amplitude')


def causal_residue_with_growth():
    zeta = sp.symbols('zeta')
    nu, gamma, eps, T = sp.symbols('nu gamma eps T', positive=True)
    field = sp.I/(zeta-sp.I*gamma)+sp.I*zeta/(zeta*zeta-nu*nu)
    news = sp.I*gamma/(zeta-sp.I*gamma)+nu*nu/(zeta*zeta-nu*nu)
    zero(-sp.I*zeta*field-2-news)
    zero(sp.limit(sp.I*eps*news.subs(zeta, nu+sp.I*eps), eps, 0, dir='+')-nu/2)
    shifted_periodic = (sp.I*nu/2)*(sp.exp(sp.I*(zeta+nu)*T)*sp.I/(zeta+nu)
                                      -sp.exp(sp.I*(zeta-nu)*T)*sp.I/(zeta-nu))
    zero(sp.limit((zeta-nu)*shifted_periodic, zeta, nu)-nu/2)
    zero(sp.limit((zeta-nu)*(shifted_periodic-shifted_periodic.subs(T, 0)), zeta, nu))
    return dict(model='F(u)=exp(gamma*u)+cos(nu*u), u>=0',
                original_transform_domain='Im zeta>gamma',
                allowed_upper_pole='zeta=i*gamma',
                boundary_news_residue='nu/2',
                all_finite_future_shifts_preserve_residue=True,
                initial_boundary_field_value=2,
                scope='Exact growing causal model; no late-time decay or final-value theorem is inferred')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "gauge_identity": gauge_integration_identity(),
        "polarizations": radiative_gauge_quotient(),
        "constraints": constraint_minor(),
        "gap": massive_exponent_gap(),
        "matching": symplectic_matching_model(),
        "tails": tail_pairing_integrals(),
        "residue": causal_residue_with_growth(),
    }, indent=2))
