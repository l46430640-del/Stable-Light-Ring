"""Algebraic checks: late carrier transfer."""
import json
import sympy as sp


def zero(value):
    if isinstance(value, sp.MatrixBase):
        assert value.applyfunc(sp.simplify) == sp.zeros(*value.shape)
    else:
        assert sp.simplify(value) == 0


def outgoing_gauge_and_flux():
    r, p, Q, Qt, Qx, Qv, lam = sp.symbols('r p Q Qt Qx Qv Lambda', positive=True)
    ht_rw = p*Q+r*Qx
    hr_rw = r*Qt/p
    ht = ht_rw+r*Qt
    hr = hr_rw+r*Qx/p-Q
    null_rule = {Qx: 2*Qv-Qt}
    zero(ht.subs(null_rule)-(p*Q+2*r*Qv))
    zero(hr.subs(null_rule)-(-Q+2*r*Qv/p))
    zero((hr+ht/p).subs(null_rule)-4*r*Qv/p)
    tensor_norm = lam*(lam-2)/2
    news_factor = 4*tensor_norm/(32*sp.pi)
    zero(news_factor-lam*(lam-2)/(16*sp.pi))
    e_factor = lam*(lam-2)/(32*sp.pi)
    zero(news_factor-2*e_factor)
    # If t_source>T+X_s and x_source<=X_s then u_source>T.
    T, Xs, extra, offset = sp.symbols('T Xs extra offset', positive=True)
    us = T+Xs+extra-(Xs-offset)
    zero(us-T-extra-offset)
    return dict(retarded_radial_component='4*r*Q_v/p',
                shear='2*Q_rad*X_AB', news_factor=str(news_factor),
                energy_flux_normalization_exact=True,
                causal_source_truncation_identity=True,
                scope='Gauge and flux algebra. Exponential defect size is inherited from the quasimode construction')


def quadratic_plane_ricci():
    u = sp.symbols('u', real=True)
    names = ['a', 'b', 'c', 'd', 'e']
    a, b, c, d, e = [sp.Function(name)(u) for name in names]
    g0 = sp.Matrix([[0, -sp.Rational(1, 2), 0, 0],
                    [-sp.Rational(1, 2), 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    h = sp.Matrix([[e, 0, c, d], [0, 0, 0, 0], [c, 0, a, b], [d, 0, b, -a]])
    inv0 = g0.inv()
    inv1 = -inv0*h*inv0
    derivative = lambda expr, index: sp.diff(expr, u) if index == 0 else sp.Integer(0)
    gamma1, gamma2 = {}, {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                terms = [derivative(h[s, k], j)+derivative(h[s, j], k)
                         -derivative(h[j, k], s) for s in range(4)]
                gamma1[i, j, k] = sp.expand(sum(inv0[i, s]*terms[s] for s in range(4))/2)
                gamma2[i, j, k] = sp.expand(sum(inv1[i, s]*terms[s] for s in range(4))/2)
    R1, R2 = sp.zeros(4), sp.zeros(4)
    for i in range(4):
        for j in range(4):
            R1[i, j] = sum(derivative(gamma1[k, i, j], k)
                            -derivative(gamma1[k, i, k], j) for k in range(4))
            R2[i, j] = sum(derivative(gamma2[k, i, j], k)
                            -derivative(gamma2[k, i, k], j) for k in range(4))
            R2[i, j] += sum(gamma1[k, k, s]*gamma1[s, i, j]
                             -gamma1[k, j, s]*gamma1[s, i, k]
                             for k in range(4) for s in range(4))
    zero(R1)
    H = sp.Matrix([[a, b], [b, -a]])
    expected = sp.trace(H*H.diff(u, 2))/2+sp.trace(H.diff(u)*H.diff(u))/4
    target = sp.zeros(4)
    target[0, 0] = expected
    zero(R2-target)
    scalar2 = sp.trace(inv0*R2)+sp.trace(inv1*R1)
    zero(scalar2)
    zero(sp.diff(R2[0, 0], c))
    zero(sp.diff(R2[0, 0], d))
    zero(sp.diff(R2[0, 0], e))
    return dict(all_linear_components_zero=True, only_quadratic_component='uu',
                quadratic_uu=str(sp.expand(expected)), scalar_curvature_quadratic_zero=True,
                null_shift_functions_cancel=True,
                scope='Exact quadratic Einstein tensor for arbitrary one-null-coordinate profiles; angular and background remainders are estimated analytically')


def logarithmic_lift():
    t, x, R = sp.symbols('t x R', positive=True)
    H = sp.Function('H')(t-x)
    q = H*sp.log(x/R)
    remainder = sp.diff(q, t, 2)-sp.diff(q, x, 2)-2*sp.diff(H, t)/x
    zero(remainder-H/x**2)
    primitive = -(sp.log(x/R)**2+4*sp.log(x/R)+5)/x
    zero(sp.diff(primitive, x)-(1+sp.log(x/R))**2/x**2)
    integral = sp.limit(primitive, x, sp.oo)-primitive.subs(x, R)
    zero(integral-5/R)
    # In the frozen null frame, the logarithmic lift is longitudinal.
    p = sp.Matrix([1, -1, 0, 0])
    metric = sp.diag(-1, 1, 1, 1)
    h = q*p*p.T/x
    zero(sp.trace(metric*h))
    zero(h[2:4, 2:4])
    defect = -h.diff(t)[0, :].T+h.diff(x)[1, :].T
    zero(defect+H*(1-sp.log(x/R))*p/x**2)
    return dict(principal_lift_exact=True, weighted_remainder_integral=str(integral),
                transverse_lift_zero=True, gauge_defect='H*(log(x/R)-1)*du/x^2',
                scope='Principal identity and exact tail integral; stellar coefficient corrections have an extra radial power')


def ordered_lifts():
    y = sp.symbols('y')
    A = sp.Matrix([[3, 1], [-1, 4]])
    B = sp.Matrix([[y, 1], [2, -y]])
    F = sp.Matrix([1+y, 2-y])
    assert A*B != B*A
    current = A.inv()*F
    total = sp.zeros(2, 1)
    for _ in range(5):
        total += current
        zero((A+B)*total-F-B*current)
        current = -A.inv()*B*current
    budgets = []
    for m in range(2, 10):
        k = (m+2)//2+1
        assert 2*k > m+2
        budgets.append(dict(input_negative_order=m, lift_count=k, gained_order=2*k-m))
    return dict(noncommuting_matrix_identity=True, checked_lift_counts=list(range(1, 6)),
                spatial_budgets=budgets,
                scope='Ordered operator algebra and derivative counts; elliptic coercivity and radiation continuity are analytical')


def future_extension_checks():
    u = sp.symbols('u', real=True)
    orders = [3, 5, 8]
    for k in orders:
        bump = sp.factorial(2*k+1)*u**k*(1-u)**k/sp.factorial(k)**2
        zero(sp.integrate(bump, (u, 0, 1))-1)
        primitive = sp.integrate(bump, (u, 0, u))
        first_lift = primitive/2
        last_lift = (1-primitive.subs(u, u-1))/2
        zero(first_lift.subs(u, 0))
        zero(last_lift.subs(u, 2))
        zero(first_lift.subs(u, 1)-last_lift.subs(u, 1))
        zero(2*sp.diff(first_lift, u)-bump)
        zero(2*sp.diff(last_lift, u)+bump.subs(u, u-1))
        zero(sp.integrate(bump, (u, 0, 1))-sp.integrate(bump.subs(u, u-1), (u, 1, 2)))
    return dict(polynomial_bump_orders=orders, total_integrals_zero=True,
                lift_compact_at_both_ends=True, leading_source_cancels_on_both_intervals=True,
                scope='Finite-smoothness examples of the future correction. The proof uses a smooth unit-integral bump')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "gauge": outgoing_gauge_and_flux(),
        "null_source": quadratic_plane_ricci(),
        "logarithm": logarithmic_lift(),
        "lifts": ordered_lifts(),
        "future_extension": future_extension_checks(),
    }, indent=2))
