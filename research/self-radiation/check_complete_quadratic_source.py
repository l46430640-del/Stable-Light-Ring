"""Algebraic checks: complete quadratic source."""
import json
import sympy as sp
SIGNS = (-1, 1, 1, 1)


def zero(value):
    assert sp.expand(value) == 0


def connection_products():
    jets = {}
    variables = []
    for d in range(4):
        for a in range(4):
            for b in range(a, 4):
                if (a, b) != (0, 0):
                    value = sp.Symbol(f'h{d}_{a}{b}')
                    jets[d, a, b] = jets[d, b, a] = value
                    variables.append(value)
        jets[d, 0, 0] = sum(jets[d, j, j] for j in range(1, 4))
    gamma = {(c, a, b): SIGNS[c]*(jets[a, b, c]+jets[b, a, c]-jets[c, a, b])/2
             for c in range(4) for a in range(4) for b in range(4)}
    for d in range(4):
        zero(sum(gamma[c, c, d] for c in range(4)))
    lam = sp.Symbol('lambda')
    mixed = {value: sp.Integer((7*j+3) % 13-6)+lam*((5*j+2) % 11-5)
             for j, value in enumerate(variables)}
    nonzero_cross = 0
    for a in range(4):
        for b in range(4):
            product = sum(gamma[c, b, d]*gamma[d, a, c]
                          for c in range(4) for d in range(4))
            T = sum(SIGNS[c]*SIGNS[d]*jets[a, c, d]*jets[b, c, d]
                    for c in range(4) for d in range(4))
            U = sum(SIGNS[c]*SIGNS[d]*jets[d, b, c]*jets[c, a, d]
                    for c in range(4) for d in range(4))
            V = sum(SIGNS[c]*SIGNS[d]*jets[d, b, c]*jets[d, a, c]
                    for c in range(4) for d in range(4))
            zero(4*product-T-2*U+2*V)
            cross = sp.expand(product.subs(mixed)).coeff(lam, 1)
            nonzero_cross += int(cross != 0)
    assert nonzero_cross > 0
    return dict(free_first_derivative_symbols=len(variables), checked_components=16,
                trace_connection_zero=True,
                product_identity='Gamma1^c_bd Gamma1^d_ac=(T_ab+2*U_ab-2*V_ab)/4',
                nonzero_mixed_components=nonzero_cross,
                scope='Exact normal-frame identity for arbitrary trace-free symmetric tensor jets, including mixed terms')


def integration_by_parts():
    coordinates = sp.symbols('t x y z')
    h = sp.Matrix(4, 4, lambda a, b:
                  sp.Function(f'h{min(a,b)}{max(a,b)}')(*coordinates))
    divergence = [sum(SIGNS[c]*sp.diff(h[b, c], coordinates[c]) for c in range(4))
                  for b in range(4)]
    box = lambda value: sum(SIGNS[d]*sp.diff(value, coordinates[d], 2) for d in range(4))
    for a in range(4):
        for b in range(4):
            U = sum(SIGNS[c]*SIGNS[d]*sp.diff(h[b, c], coordinates[d])
                    *sp.diff(h[a, d], coordinates[c]) for c in range(4) for d in range(4))
            U_flux = sum(SIGNS[c]*SIGNS[d]*sp.diff(h[a, d]*sp.diff(h[b, c], coordinates[d]), coordinates[c])
                         for c in range(4) for d in range(4))
            U_flux -= sum(SIGNS[d]*sp.diff(h[a, d]*divergence[b], coordinates[d]) for d in range(4))
            zero(U-U_flux-divergence[a]*divergence[b])
            V = sum(SIGNS[c]*SIGNS[d]*sp.diff(h[b, c], coordinates[d])
                    *sp.diff(h[a, c], coordinates[d]) for c in range(4) for d in range(4))
            V_flux = sum(SIGNS[c]*SIGNS[d]*sp.diff(h[b, c]*sp.diff(h[a, c], coordinates[d]), coordinates[d])
                         for c in range(4) for d in range(4))
            zero(V-V_flux+sum(SIGNS[c]*h[b, c]*box(h[a, c]) for c in range(4)))
    return dict(U_identities=16, V_identities=16,
                U_remainder='D_a*D_b in flat space; curvature*h*h in a curved background',
                V_remainder='-h_bc*box(h_a^c)',
                scope='Exact flat product rules; the covariant commutator is a zeroth-order curvature term')


def scalar_noether_identity():
    coordinates = sp.symbols('t x y z')
    mu = sp.Symbol('mu')
    sigma = sp.Symbol('sigma', nonzero=True)
    q = [sp.Function(f'q{j}')(*coordinates) for j in range(2)]
    s = sum(value*value for value in q)
    potential = mu*mu*s*(1-2*s/sigma**2)**2
    potential_prime = mu*mu*(1-2*s/sigma**2)*(1-6*s/sigma**2)
    kinetic = sum(SIGNS[a]*sp.diff(value, coordinates[a])**2 for value in q for a in range(4))
    T = sp.Matrix(4, 4, lambda a, b: sum(sp.diff(value, coordinates[a])*sp.diff(value, coordinates[b])
                                        for value in q)
                  -(SIGNS[a] if a == b else 0)*(kinetic+potential)/2)
    residuals = [sum(SIGNS[a]*sp.diff(value, coordinates[a], 2) for a in range(4))
                 -potential_prime*value for value in q]
    for b in range(4):
        div = sum(SIGNS[a]*sp.diff(T[a, b], coordinates[a]) for a in range(4))
        zero(div-sum(residuals[j]*sp.diff(q[j], coordinates[b]) for j in range(2)))
    return dict(real_scalar_components=2, checked_divergences=4,
                potential='V(s)=mu^2*s*(1-2*s/sigma^2)^2',
                stress_identity='div(T)_b=sum_I (box(qI)-V_s*qI)*d_b(qI)',
                quadratic_source_identity='div(tau)_b=sum_I S2_I*d_b(q0I), when E0=E1=S0=S1=0',
                scope='Exact off-shell matter identity; its second-order consequence uses the exact background and linear equations')


def distribution_budget():
    L = sp.Symbol('L', positive=True)
    h, dh, divh = 1/L, sp.Integer(1), 1/L
    terms = dict(connection_divergence=h*dh, U_divergence=h*dh,
                 U_second_divergence=h*divh, divergence_square=divh*divh,
                 V_remainder=divh*dh, curvature=h*h, scalar_direct=h*h+h*dh)
    assert all(sp.limit(value, L, sp.oo) == 0 for value in terms.values())
    assert all(sp.limit(L*value, L, sp.oo).is_finite for value in terms.values())
    m, order = 4, 2
    assert m > order+sp.Rational(1, 2)
    lift_count = (m+2)//2+1
    assert 2*lift_count > m+2
    return dict(terms={name: str(value) for name, value in terms.items()},
                complete_residual_order='O(L^-1) against fixed C^2 tests',
                annular_radial_sobolev_example=dict(distribution_order=order, negative_order=m,
                                                  lift_count=lift_count, time_derivatives=2*lift_count+2),
                scope='Bookkeeping only; no convergence rate is assigned to the separate WKB orbital limit')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "connections": connection_products(),
        "products": integration_by_parts(),
        "noether": scalar_noether_identity(),
        "bounds": distribution_budget(),
    }, indent=2))
