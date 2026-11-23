"""Algebraic checks: bondi clock calibration."""
import json
import sympy as sp
from check_radiation_quartic import efficiency


def zero(expr):
    assert sp.simplify(expr) == 0


def sphere_translation_kernel():
    z = sp.symbols('z', real=True)
    cases = []
    for ell in range(11):
        f = sp.legendre(ell, z)
        eigenvalue = ell*(ell+1)
        norm = sp.integrate(f*f, (z, -1, 1))
        hessian = sp.integrate((1-z*z)**2*sp.diff(f, z, 2)**2/2,
                               (z, -1, 1))
        zero(hessian-eigenvalue*(eigenvalue-2)*norm/2)
        zero(f.subs(z, -z)-(-1)**ell*f)
        assert (hessian == 0) == (ell in (0, 1))
        cases.append(dict(ell=ell, norm_without_2pi=str(hessian)))
    k = sp.symbols('k', nonnegative=True)
    ell = k+2
    assert (ell*(ell+1)*(ell*(ell+1)-2)/2).is_positive
    return dict(axisymmetric_checks=cases, kernel_degrees=[0, 1],
                inversion_even_kernel=[0],
                scope='Exact checks of the sphere identity and parity; the all-m statement follows from the Bochner identity in the proof')


def news_and_endpoint_jets():
    a, u, t1, t2 = sp.symbols('a u t1 t2', real=True)
    n1, n2, n3 = [sp.Function(name)(u) for name in ('n1', 'n2', 'n3')]
    shift = t1*a+t2*a*a
    shifted = (a*(n1-shift*sp.diff(n1, u)+shift**2*sp.diff(n1, u, 2)/2)
               +a*a*(n2-shift*sp.diff(n2, u))+a**3*n3)
    expected = [n1, n2-t1*sp.diff(n1, u),
                n3-t1*sp.diff(n2, u)-t2*sp.diff(n1, u)
                +t1*t1*sp.diff(n1, u, 2)/2]
    for order, value in enumerate(expected, 1):
        zero(sp.expand(shifted).coeff(a, order)-value)
    db3 = 2*expected[0]*expected[1]-2*n1*n2
    db4 = expected[1]**2+2*expected[0]*expected[2]-n2*n2-2*n1*n3
    primitive3 = -t1*n1*n1
    primitive4 = -2*t1*n1*n2-t2*n1*n1+t1*t1*sp.diff(n1*n1, u)/2
    zero(db3-sp.diff(primitive3, u))
    zero(db4-sp.diff(primitive4, u))
    e2 = sp.symbols('e2', positive=True)
    e3, e4, b2, b3, b4, d3, d4 = sp.symbols('e3 e4 b2 b3 b4 d3 d4')
    change = efficiency(e2, e3, e4, b2, b3+d3, b4+d4)-efficiency(e2, e3, e4, b2, b3, b4)
    zero(change-d4/e2**2+3*e3*d3/(2*e2**3))
    # A polynomial waveform keeps the endpoint and energy corrections nonzero.
    example = {n1: u, n2: u*u, n3: 1+u, t1: 2, t2: 3}
    exact_d3 = sp.integrate(db3.subs(example).doit(), (u, 0, 1))
    exact_d4 = sp.integrate(db4.subs(example).doit(), (u, 0, 1))
    assert (exact_d3, exact_d4) == (-2, -3)
    exact_dk = change.subs({d3: exact_d3, d4: exact_d4, e2: 2, e3: 3})
    zero(exact_dk-sp.Rational(3, 8))
    return dict(news_jets_through_cubic=True, endpoint_total_derivatives=True,
                normalized_efficiency_change='delta_b4/e2^2-3*e3*delta_b3/(2*e2^3)',
                finite_example=dict(delta_b3=str(exact_d3), delta_b4=str(exact_d4),
                                    delta_kappa=str(sp.simplify(exact_dk))))


def mixed_composition_budget():
    b, u = sp.symbols('b u', real=True)
    t = sp.symbols('t1:5')
    tau = sum(t[j]*b**(j+1)/sp.factorial(j+1) for j in range(4))
    jets = {(p, q): sp.Symbol(f'n{p}{q}') for p in range(5) for q in range(5-p)}
    n = sum(value*b**p*u**q/(sp.factorial(p)*sp.factorial(q))
            for (p, q), value in jets.items())
    composed = sp.Poly(sp.expand(n.subs(u, -tau)), b)
    terms = {(0, 0): sp.S.One}
    rows = []
    for order in range(5):
        actual = sp.factorial(order)*composed.nth(order)
        expected = 0
        for (p, q), coefficient in terms.items():
            expected += coefficient*jets[p, q]
            assert p+q <= order
        zero(actual-expected)
        rows.append(dict(order=order, mixed_terms=len(terms),
                         coefficients={f'{p},{q}': str(c) for (p, q), c in sorted(terms.items())}))
        if order == 4:
            break
        following = {}
        for (p, q), coefficient in terms.items():
            for key, value in (((p+1, q), coefficient), ((p, q+1), -t[0]*coefficient)):
                following[key] = following.get(key, 0)+value
            derivative = sum(sp.diff(coefficient, t[j])*t[j+1] for j in range(3))
            following[p, q] = following.get((p, q), 0)+derivative
        terms = {key: sp.expand(value) for key, value in following.items() if value != 0}
    # Five parameter orders for C_b^4, four time orders of news, one for shear.
    assert 5+4+1 == 10 < 48
    return dict(chain_rule=rows, sufficient_trace_order=10, available_order=48,
                scope='The chain rule preserves four parameter derivatives with mixed time regularity; the common PDE bounds are separate')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "sphere": sphere_translation_kernel(),
        "jets": news_and_endpoint_jets(),
        "composition": mixed_composition_budget(),
    }, indent=2))
