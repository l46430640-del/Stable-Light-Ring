"""Algebraic checks: joint gauge matter."""
import json
import sympy as sp
from check_coupled_exterior_cauchy import matter_trace_reversal


def matrix(seed):
    return sp.Matrix(4, 4, lambda i, j:
                     sp.Rational(((min(i, j)+2)*(max(i, j)+3)+seed) % 11-5, 37))


def connection(inverse, derivatives):
    return {(k, i, j): sum(inverse[k, l]*(derivatives[i][j, l]
               +derivatives[j][i, l]-derivatives[l][i, j])/2 for l in range(4))
            for k in range(4) for i in range(4) for j in range(4)}


def contracted(inverse, gamma):
    return sp.Matrix([sum(inverse[i, j]*gamma[k, i, j]
                         for i in range(4) for j in range(4)) for k in range(4)])


def algebraic_gauge_and_scalar_variation():
    eta = sp.diag(-1, 1, 1, 1)
    gamma = sp.Rational(-1, 3)
    examples = []
    for seed in (1, 3, 7):
        basis = sp.eye(4)+matrix(seed)/4
        g = basis.T*eta*basis
        bar_basis = sp.eye(4)+matrix(seed+5)/5
        bar = bar_basis.T*eta*bar_basis
        gi, bi = g.inv(), bar.inv()
        h = matrix(seed+1)
        c = sp.Matrix([sp.Rational(1, 7), 0, 0, 0])
        k = g-bar
        trace_reversed = k-bar*sp.trace(bi*k)/2
        J = gamma*(2*trace_reversed*bi*c-c*sp.trace(bi*trace_reversed))
        F = 2*gamma*(bi-gi)*c
        assert gi*J == F
        assert J == 2*gamma*k*bi*c
        dgi = -gi*h*gi
        dJ = 2*gamma*h*bi*c
        dF = dgi*J+gi*dJ
        assert dF == 2*gamma*gi*h*gi*c
        dg = [matrix(seed+11+a) for a in range(4)]
        dbar = [matrix(seed+21+a) for a in range(4)]
        G = connection(gi, dg)
        Gbar = connection(bi, dbar)
        gradient = sp.Matrix([sp.Rational(seed+i, 13) for i in range(4)])
        Hessian = matrix(seed+31)
        contracted_g = contracted(gi, G)
        contracted_bar = contracted(gi, Gbar)
        C = contracted_g-contracted_bar-F
        physical = sp.trace(gi*Hessian)-(contracted_g.T*gradient)[0]
        reduced = physical+(C.T*gradient)[0]
        expected = sp.trace(gi*Hessian)-((contracted_bar+F).T*gradient)[0]
        assert reduced == expected
        variations = []
        physical_variations = []
        for dh_seed in (41, 57):
            dh = [matrix(seed+dh_seed+a) for a in range(4)]
            varied_inverse_connection = connection(dgi, dg)
            varied_derivative_connection = connection(gi, dh)
            dG = {key: varied_inverse_connection[key]+varied_derivative_connection[key]
                  for key in G}
            dcontracted_g = contracted(dgi, G)+contracted(gi, dG)
            dcontracted_bar = contracted(dgi, Gbar)
            dC = dcontracted_g-dcontracted_bar-dF
            varied_physical = sp.trace(dgi*Hessian)-(dcontracted_g.T*gradient)[0]
            varied_reduced = varied_physical+(dC.T*gradient)[0]
            target = sp.trace(dgi*Hessian)-((dcontracted_bar+dF).T*gradient)[0]
            assert varied_reduced == target
            variations.append(varied_reduced)
            physical_variations.append(varied_physical)
        assert variations[0] == variations[1]
        assert physical_variations[0] != physical_variations[1]
        examples.append(dict(seed=seed, metric_derivative_variants=2,
                             reduced_scalar_variation=str(variations[0])))
    return dict(exact_rational_metric_jets=examples,
                four_dimensional_trace_cancellation=True,
                gauge_source='2*gamma*(bar_g_inverse-g_inverse)*c',
                scalar_metric_block_has_no_derivative_of_h=True,
                cancellation_does_not_require_C_zero=True)


def principal_and_initial_constraints():
    eta = sp.diag(-1, 1, 1, 1)
    k = sp.Matrix(sp.symbols('k0:4'))
    C = sp.Matrix(sp.symbols('C0:4'))
    h = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f'h{min(i,j)}{max(i,j)}'))
    k2 = (k.T*eta*k)[0]
    trace_h = sp.trace(eta*h)
    dU = h*eta*k-k*trace_h/2
    Ric = (k*(h*eta*k).T+(h*eta*k)*k.T-k2*h-k*k.T*trace_h)/2
    reduced = Ric-(k*dU.T+dU*k.T)/2
    assert (reduced+k2*h/2).applyfunc(sp.expand) == sp.zeros(4)
    DC = (k*C.T+C*k.T)/2
    trace_reversed = DC-eta*sp.trace(eta*DC)/2
    assert (trace_reversed*eta*k-k2*C/2).applyfunc(sp.expand) == sp.zeros(4, 1)
    n = sp.Matrix([1, 0, 0, 0])
    ncov = eta*n
    A = sp.Matrix(sp.symbols('A0:4'))
    initial_DC = -(ncov*A.T+A*ncov.T)/2
    initial_reversed = initial_DC-eta*sp.trace(eta*initial_DC)/2
    assert initial_reversed*n == A/2
    # Direct connection jets isolate the four lapse/shift time derivatives.
    a = sp.symbols('a0:4')
    derivatives = [sp.zeros(4) for _ in range(4)]
    derivatives[0][0, 0] = a[0]
    for i in range(1, 4):
        derivatives[0][0, i] = derivatives[0][i, 0] = a[i]
    initial_C = eta*contracted(eta, connection(eta, derivatives))
    assert initial_C.jacobian(a) == sp.diag(sp.Rational(-1, 2), -1, -1, -1)
    return dict(metric_principal='-Box_g*I_10/2', scalar_principal='Box_g*I_2',
                common_characteristic_fields=12, subsidiary_principal='Box_g*C',
                normal_constraint='n^mu*G(DC)_mu_nu=A_nu/2',
                initial_gauge_jacobian=['-1/2', '-1', '-1', '-1'])


def stress_divergence_and_mass():
    eta = sp.diag(-1, 1, 1, 1)
    fields = sp.Matrix(sp.symbols('p0:2', real=True))
    gradients = [sp.Matrix(sp.symbols(f'd{j}_0:4', real=True)) for j in range(2)]
    Hessians = [sp.Matrix(4, 4, lambda a, b:
                    sp.Symbol(f'H{j}_{min(a,b)}{max(a,b)}', real=True)) for j in range(2)]
    Vp = sp.Symbol('Vp', real=True)
    box = [sp.trace(eta*H) for H in Hessians]
    for nu in range(4):
        divergence = sum(eta[a, a]*sum(Hessians[j][a, a]*gradients[j][nu]
                  +gradients[j][a]*Hessians[j][a, nu] for j in range(2))
                         for a in range(4))
        divergence -= sum(sum(eta[a, a]*gradients[j][a]*Hessians[j][nu, a]
                          for a in range(4))+Vp*fields[j]*gradients[j][nu] for j in range(2))
        expected = sum((box[j]-Vp*fields[j])*gradients[j][nu] for j in range(2))
        assert sp.expand(divergence-expected) == 0
    C = sp.Matrix(sp.symbols('C0:4'))
    kinetic = sum((d*d.T for d in gradients), sp.zeros(4))
    gauge_divergence = sum((-(C.T*d)[0]*d for d in gradients), sp.zeros(4, 1))
    assert (gauge_divergence+kinetic*C).applyfunc(sp.expand) == sp.zeros(4, 1)
    s, mu, sigma = sp.symbols('s mu sigma', positive=True)
    V = mu**2*s*(1-2*s/sigma**2)**2
    Vp_expr, Vpp = sp.diff(V, s), sp.diff(V, s, 2)
    M = Vp_expr*sp.eye(2)+2*Vpp*fields*fields.T
    z = sp.Symbol('z')
    characteristic = (M-z*sp.eye(2)).det()
    target = (Vp_expr-z)*(Vp_expr+2*Vpp*(fields.T*fields)[0]-z)
    assert sp.expand(characteristic-target) == 0
    x = sp.Symbol('x', nonnegative=True)
    phase = sp.simplify((Vp_expr/mu**2).subs(s, x*sigma**2))
    radial = sp.simplify(((Vp_expr+2*s*Vpp)/mu**2).subs(s, x*sigma**2))
    assert sp.expand(phase-(1-8*x+12*x*x)) == 0
    assert sp.expand(radial-(1-24*x+60*x*x)) == 0
    endpoint = sp.Rational(1, 64)
    assert phase.subs(x, endpoint) == sp.Rational(899, 1024)
    assert radial.subs(x, endpoint) == sp.Rational(655, 1024)
    assert sp.diff(phase, x).subs(x, endpoint) < 0
    assert sp.diff(radial, x).subs(x, endpoint) < 0
    assert sp.diff(phase, x, 2) > 0 and sp.diff(radial, x, 2) > 0
    assert sp.Rational(655, 1024) > sp.Rational(1, 2)
    return dict(stress_divergence_identity=True,
                off_gauge_divergence='-Re(dphi_bar*dphi)_nu_lambda*C^lambda',
                mass_eigenvalues=[str(phase), str(radial)],
                scalar_small_field_condition='|phi|^2 <= sigma0^2/64',
                scalar_mass_lower_bound='655*mu^2/1024',
                inherited_stress_variation=matter_trace_reversal())


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "gauge": algebraic_gauge_and_scalar_variation(),
        "principal": principal_and_initial_constraints(),
        "matter": stress_divergence_and_mass(),
    }, indent=2))
