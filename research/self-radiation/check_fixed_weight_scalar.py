"""Algebraic checks: fixed weight scalar."""
import json
from math import comb
import sympy as sp


def zero(expr):
    assert sp.simplify(expr) == 0


def mass_commutators():
    a, b, theta = sp.symbols('a b theta', real=True)
    mu = sp.symbols('mu', positive=True)
    M = mu**2*sp.exp(-2*a-b)
    D = lambda u: sp.diff(u, b)
    V = lambda u: sp.diff(u, a)-2*sp.diff(u, b)
    angular = [lambda u: sp.diff(u, theta), lambda u: theta*sp.diff(u, theta)]
    zero(V(M))
    zero(D(M)+M)
    polynomial = (1+a+a*b+b**4)*(1+theta+theta**3)
    count = 0
    for power in range(1, 7):
        for word in ((), (0,), (1,), (0, 1), (1, 0)):
            u = polynomial
            for index in word:
                u = angular[index](u)
            expected = M*sum((-1)**(j+1)*comb(power, j)*sp.diff(u, b, power-j)
                             for j in range(1, power+1))
            zero(M*sp.diff(u, b, power)-sp.diff(M*u, b, power)-expected)
            zero(V(expected)-M*sum((-1)**(j+1)*comb(power, j)
                                   *sp.diff(V(u), b, power-j)
                                   for j in range(1, power+1)))
            count += 1
    zero(V(-b)-2)
    zero(V(-2*a-b))
    zero(D(-b)+1)
    zero(D(-2*a-b)+1)
    return dict(commutator_cases=count, mass_preserved_by_V=True,
                coordinate_change='tau=-log(y), s=log(q*r), V=2*d_tau, D1=-d_tau-d_s',
                angular_example='Two noncommuting scalar angular derivations; no spherical wave estimate inferred')


def duality():
    a, b, alpha, A, mu = sp.symbols('a b alpha A mu', real=True)
    w2 = sp.exp(-2*alpha*a+2*A*b)
    M = mu**2*sp.exp(-2*a-b)
    J, z, u = (sp.Function(name)(a, b) for name in ('J', 'z', 'u'))
    V = lambda f: sp.diff(f, a)-2*sp.diff(f, b)
    density = w2*M*J
    d = 2*alpha+4*A-V(J)/J
    zero(V(density)+d*density)
    zero(density*z*V(u)-V(density*z*u)+density*(V(z)-d*z)*u)

    # An exact finite rectangle retains the entire divergence boundary term.
    aa, bb = sp.symbols('aa bb', real=True)
    z0, u0 = 1+aa+bb, 2+aa*bb
    vector = lambda f: sp.diff(f, aa)-2*sp.diff(f, bb)
    dens = 1+aa/100+bb*bb/100
    integrand = sp.expand(vector(dens*z0*u0))
    volume = sp.integrate(integrand, (aa, 0, 1), (bb, 0, 1))
    boundary = sp.integrate((dens*z0*u0).subs(aa, 1)-(dens*z0*u0).subs(aa, 0), (bb, 0, 1))
    boundary -= 2*sp.integrate((dens*z0*u0).subs(bb, 1)-(dens*z0*u0).subs(bb, 0), (aa, 0, 1))
    zero(volume-boundary)
    assert boundary != 0
    return dict(variable_density_identity=True, mass_bulk_coefficient='2*alpha0+4*A-V(log J)',
                exact_rectangle_boundary_checked=True, boundary_not_discarded=True)


def derivative_hierarchy():
    nodes, edges = 0, 0
    for p in range(1, 13):
        for a in range(p+1):
            for b in range(p-a+1):
                angular = p-a-b
                nodes += 1
                for j in range(1, b+1):
                    # Vz is the most differentiated field after integration.
                    target = (a+1, b-j, angular)
                    assert sum(target) <= p
                    assert target[1] < b
                    edges += 1
    models = []
    for p in (1, 2, 3, 5, 8):
        T = sp.zeros(p+1)
        for b in range(1, p+1):
            for lower in range(b):
                T[b, lower] = comb(b, b-lower)**2+1
        assert T**(p+1) == sp.zeros(p+1)
        inverse = sum((T**j for j in range(p+1)), sp.zeros(p+1))
        assert (sp.eye(p+1)-T)*inverse == sp.eye(p+1)
        assert all(value >= 0 for value in inverse)
        models.append(dict(max_D1_count=p, finite_induction_length=p+1,
                           largest_integer_coefficient=str(max(inverse))))
    return dict(nodes=nodes, mass_dependency_edges=edges, every_edge_lowers_D1_count=True,
                finite_triangular_models=models, weight_change_required_by_mass_induction=False,
                scope='Derivative and induction algebra; not a numerical estimate of PDE constants')


def reference_wave():
    q, r, m, mu = sp.symbols('q r m mu', positive=True)
    f = 1-2*m/r
    u = sp.Function('u')(q, r)
    D0 = lambda a: -q*sp.diff(a, q)-r*sp.diff(a, r)
    D1 = lambda a: -r*sp.diff(a, r)
    scalar = u/r
    box = (sp.diff(r*r*sp.diff(scalar, r), q)
           +sp.diff(r*r*(sp.diff(scalar, q)+f*sp.diff(scalar, r)), r))/(r*r)
    actual = q*r*r*(-box+mu*mu*scalar)
    expected = -2*D1(D0(u)-D1(u))-(q/r)*f*(D1(D1(u))+D1(u))
    expected += q*sp.diff(f, r)*(D1(u)+u)+mu*mu*q*r*u
    zero(actual-expected)
    a, b, theta = sp.symbols('a b theta', real=True)
    field = sp.Function('field')(a, b, theta)
    V = lambda expr: sp.diff(expr, a)-2*sp.diff(expr, b)
    D = lambda expr: sp.diff(expr, b)
    angular = lambda expr: -sp.exp(b)*sp.diff(expr, theta, 2)
    zero(angular(V(field))-V(angular(field))+2*sp.exp(b)*sp.diff(field, theta, 2))
    zero(angular(D(field))-D(angular(field))-sp.exp(b)*sp.diff(field, theta, 2))
    return dict(exact_rescaled_Schwarzschild_operator=True,
                reference_angular_commutators_checked=True,
                principal_radial_part='-2 D1 (D0-D1)',
                scope='Exact radial operator and a local angular commutator model')


def nonlinear_weights():
    rho0, y, mu, sigma, z = sp.symbols('rho0 y mu sigma z', positive=True)
    rho = rho0*y
    scale = y*rho**(-3)
    zero(scale*(8*mu**2*(rho*z)**3/sigma**2)-8*mu**2*y*z**3/sigma**2)
    zero(scale*(12*mu**2*(rho*z)**5/sigma**4)-12*mu**2*rho0**2*y**3*z**5/sigma**4)
    cases = []
    for A in (sp.Rational(1, 16), sp.Rational(1, 4), sp.Rational(3, 8)):
        for a in (sp.Rational(1, 4), sp.Rational(1, 2), 1):
            assert 1-2*A > 0 and 3-4*A > 0
            weights = []
            for n in range(7):
                aa, bb = a+2*n, -A+n
                linear = (aa+2, bb+1)
                cubic = (3*aa+2, 3*bb+2)
                quintic = (5*aa+4, 5*bb+4)
                assert cubic[0] >= linear[0] and cubic[1] > linear[1]
                assert quintic[0] > linear[0] and quintic[1] > linear[1]
                weights.append([str(aa), str(bb)])
            cases.append(dict(A=str(A), alpha0=str(a), recovered_weights=weights,
                              three_step_phi_weights=[str(a+7), str(4-A)]))
    # Low-norm contraction after incorporating the fixed inverse norm into C.
    contraction = []
    for C in (1, 2, 10, 100):
        radius = sp.Rational(1, 100*C)
        lipschitz = C*(3*radius**2+5*radius**4)
        assert lipschitz < sp.Rational(1, 2)
        contraction.append(dict(C=C, radius=str(radius), bound=str(lipschitz)))
    return dict(exact_cubic_and_quintic_rescaling=True, recovery_cases=cases,
                abstract_contraction_examples=contraction,
                three_step_recovery_derivative_cost=6,
                critical_cubic_boundary='A=1/2 has no positive y gain; it is excluded',
                scope='Weights, derivative subbudget and abstract norm inequalities; no evolution is performed')


def coupling_margins():
    cases = []
    A = sp.Rational(1, 4)
    for ell0, ellI in ((sp.Rational(1, 4), sp.Rational(1, 8)),
                       (sp.Rational(1, 2), sp.Rational(1, 4)),
                       (1, sp.Rational(2, 5))):
        assert 2*ell0+11 > ell0 and 6-2*A > ellI
        assert 2*ell0+5 > ell0 and 3-2*A > ellI
        cases.append(dict(ell0=str(ell0), ellI=str(ellI),
                          nonlinear_source_null_margin=str(6-2*A-ellI),
                          mixed_source_null_margin=str(3-2*A-ellI),
                          mixed_source_spatial_margin=str(ell0+5)))
    return cases


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "commutators": mass_commutators(),
        "pairing": duality(),
        "hierarchy": derivative_hierarchy(),
        "wave": reference_wave(),
        "nonlinearity": nonlinear_weights(),
        "coupling": coupling_margins(),
    }, indent=2))
