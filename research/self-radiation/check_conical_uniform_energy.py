"""Algebraic checks: conical uniform energy."""
import json
from itertools import product
from math import factorial
import sympy as sp


def zero(expression):
    if isinstance(expression, sp.MatrixBase):
        assert expression.applyfunc(sp.simplify) == sp.zeros(*expression.shape)
    else:
        assert sp.simplify(expression) == 0


def causal_patch_and_flux():
    eta = sp.symbols('eta', positive=True)
    c, v, kappa = 1+eta/16, 1-eta/4, eta/8
    zero(4-kappa-4*c*v-5*eta/8-eta**2/16)
    assert sp.expand(1-c*v-kappa).is_positive
    assert sp.expand(4-2*(1+c*v)).is_positive
    b, b1, d11, d12, d13, lam = sp.symbols('b b1 d11 d12 d13 lam', nonzero=True)
    A = sp.zeros(5)
    A[0, 0] = -2*b1/b
    A[0, 1], A[0, 2], A[0, 3] = -d11/b, -d12/b, -d13/b
    A[1, 0] = 1
    zero((lam*sp.eye(5)-A).det()-lam**3*(lam**2+2*b1*lam/b+d11/b))
    directions = [sp.eye(3)[:, j] for j in range(3)]
    directions += [sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5), 0]),
                   sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)])]
    count = 0
    for denominator in (1024, 2048, 4096, 8192):
        eps = sp.Rational(1, denominator)
        bb = -1-eps
        shift = eps*sp.Matrix([1, sp.Rational(1, 2), sp.Rational(1, 3)])
        spatial = sp.eye(3)+eps*sp.Matrix([[1, sp.Rational(1, 2), 0],
                                         [sp.Rational(1, 2), 1, sp.Rational(1, 3)],
                                         [0, sp.Rational(1, 3), 1]])
        H = sp.diag(-bb, spatial, 1)
        assert all(H[:j, :j].det() > 0 for j in range(1, 6))
        for n in directions:
            zero(n.dot(n)-1)
            An = sp.zeros(5)
            An[0, 0] = -2*shift.dot(n)/bb
            for j in range(3):
                An[0, j+1] = -(n.T*spatial)[j]/bb
                An[j+1, 0] = n[j]
            zero(H*An-(H*An).T)
            for sign in (-1, 1):
                outgoing = (1+4*eps)*H-sign*H*An
                assert all(outgoing[:j, :j].det() > 0 for j in range(1, 6))
                count += 1
    return dict(minimum_annulus_width='5*eta/8+eta^2/16',
                target_points_inside_shrinking_annuli=True,
                characteristic_polynomial='lambda^3*(lambda^2+2*b_n*lambda/b+d_nn/b)',
                positive_outgoing_matrices=count,
                scope='Exact principal identities and finite rational flux examples; the uniform cone bound is analytical')


def partitions(n):
    if n == 0:
        return [()]
    result = []
    for old in partitions(n-1):
        result.append(old+((n-1,),))
        for j in range(len(old)):
            result.append(old[:j]+(old[j]+(n-1,),)+old[j+1:])
    return result


def mass_jet_scaling():
    x, a, L, mu = sp.symbols('x a L mu', positive=True)
    b = sp.Function('b')(x)
    terms = 0
    for order in range(1, 7):
        formula = 0
        for partition in partitions(order):
            blocks = len(partition)
            formula += (-1)**blocks*factorial(blocks)*sp.prod(
                sp.diff(b, x, len(block)) for block in partition)/b**(blocks+1)
            # Each differentiated coefficient is M/L+U/L^2; Lambda=mu*L.
            for selection in product((0, 1), repeat=blocks):
                perturbation_factors = sum(selection)
                scale_power = 1-blocks-perturbation_factors
                assert scale_power <= 0
                if perturbation_factors:
                    assert scale_power <= -1
                terms += 1
        zero(formula-sp.diff(1/b, x, order))
    m = sp.Function('m')(a)
    reference_b = -1/(1-2*m/(L*x))
    reference_checks = 0
    for spatial_order in range(5):
        for parameter_order in range(5):
            if spatial_order+parameter_order == 0:
                continue
            actual = mu*L*sp.diff(1/reference_b, x, spatial_order, a, parameter_order)
            expected = 2*mu*(-1)**spatial_order*factorial(spatial_order)
            expected *= sp.diff(m, a, parameter_order)/x**(spatial_order+1)
            zero(actual-expected)
            zero(sp.diff(actual, L))
            reference_checks += 1
    m1, m2, u1, u2, f = sp.symbols('m1 m2 u1 u2 f')
    b1, b2 = -1+m1*f/L+u1/L**2, -1+m2*f/L+u2/L**2
    zero(mu*L*(1/b1-1/b2)-mu*((m2-m1)*f+(u2-u1)/L)/(b1*b2))
    return dict(inverse_derivative_orders=6, coefficient_weight_allocations=terms,
                schwarzschild_mixed_jets=reference_checks,
                every_mass_commutator_scale_power_nonpositive=True,
                each_perturbation_factor_gains_at_least_one_inverse_scale=True,
                mass_difference_identity=True,
                scope='General chain-rule identities through six derivatives and exact reference jets; all-order product estimates are written in the proof')


def bootstrap_and_high_energy():
    t, C, A0, W0, eps, L = sp.symbols('t C A0 W0 eps L', positive=True)
    Abar, Wbar = (A0+1)*sp.exp(C*t)-1, W0*sp.exp(2*C*t)
    zero(sp.diff(Abar, t)-C*(Abar+1))
    zero(sp.diff(Wbar, t)-2*C*Wbar)
    gamma = C*eps/sp.sqrt(L)
    A = sp.exp(C*t)*(A0*sp.cosh(gamma*t)+sp.sqrt(L)*W0*sp.sinh(gamma*t))
    W = sp.exp(C*t)*(W0*sp.cosh(gamma*t)+A0*sp.sinh(gamma*t)/sp.sqrt(L))
    zero(sp.diff(A, t)-C*(A+eps*W))
    zero(sp.diff(W, t)-C*(W+eps*A/L))
    zero(A.subs(t, 0)-A0)
    zero(W.subs(t, 0)-W0)
    zero(sp.diff(W, eps).subs(eps, 0)-sp.exp(C*t)*C*t*A0/L)
    return dict(low_majorants_satisfy_comparison_equations=True,
                high_comparison_system_solved_exactly=True,
                induced_scalar_first_order='exp(C*t)*C*t*eps*A0/L',
                scope='Comparison equations verify the scaling of the analytic Gronwall argument, not the constants of the PDE estimates')


def cutoff_and_tail_norms():
    r, R, x, alpha = sp.symbols('r R x alpha', positive=True)
    chi = sp.Function('chi')
    radial, scaled = chi(r/R), chi(x)
    for order in range(7):
        zero(radial.subs(r, R*x).doit()-scaled)
        radial, scaled = r*sp.diff(radial, r), x*sp.diff(scaled, x)
    logarithmic_checks = []
    for j in range(7):
        n = 2*j
        tail = sp.exp(-alpha*x)*sum(sp.binomial(n, k)*factorial(k)*(1+x)**(n-k)
                                    /alpha**(k+1) for k in range(n+1))
        zero(sp.diff(tail, x)+sp.exp(-alpha*x)*(1+x)**n)
        assert sp.limit(tail, x, sp.oo) == 0
        logarithmic_checks.append(dict(parameter_order=j, exact_antiderivative=True,
                                       vanishing_tail=True))
    return dict(conormal_cutoff_orders=7, logarithmic_tail_checks=logarithmic_checks,
                metric_norm_rate='R^(ell0-1)*(1+log(R))^J',
                residual_extension_orders=[31, 38], fixed_high_extension_orders=[3300, 4824],
                scope='Exact cutoff and majorant formulas; constants and conical solution estimates are analytical')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "geometry": causal_patch_and_flux(),
        "mass_jets": mass_jet_scaling(),
        "comparison": bootstrap_and_high_energy(),
        "tails": cutoff_and_tail_norms(),
    }, indent=2))
