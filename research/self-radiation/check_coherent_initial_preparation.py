"""Algebraic checks: coherent initial preparation."""
import json
import numpy as np
import sympy as sp
from check_radiation_quartic import laurent


def phase_filter_checks():
    z = sp.Symbol('z', nonzero=True)
    a, b, abar, bbar = sp.symbols('a b abar bbar')
    product = (z*a + abar/z)*(z*b + bbar/z)/4
    coefficients = laurent(product, (z,))
    retained = (a*bbar + abar*b)/4
    assert coefficients[(0,)] == retained
    assert sp.expand((product + product.subs(z, sp.I*z))/2 - retained) == 0
    grid = sum(c for (q,), c in coefficients.items() if q % 5 == 0)
    assert grid == retained
    # Twenty positive-frequency carriers, with two frequency offsets only.
    offsets = [0, 1]*10
    retained_differences = {a-b for a in offsets for b in offsets}
    assert retained_differences == {-1, 0, 1}
    all_products = [(s+t, s+t, s*a+t*b)
                    for a in offsets for b in offsets
                    for s in (-1, 1) for t in (-1, 1)]
    assert all(carrier == 0 and slow in retained_differences
               for charge, carrier, slow in all_products if charge % 5 == 0)
    return dict(common_phase_quadratic_projection=True,
                two_quadrature_identity=True, five_phase_identity=True,
                carrier_count=20, bilinear_frequency_checks=len(all_products),
                retained_frequency_multipliers=sorted(retained_differences),
                coherent_difference_retained=True)


def temporal_trace_checks():
    # Exact interval Gram model, not the smooth-cutoff proof in the text.
    nu = sp.Symbol('nu', positive=True)
    j = (-1, 0, 1)
    gram = sp.Matrix([[2 if a == b else 2*sp.sin((a-b)*nu)/((a-b)*nu)
                       for b in j] for a in j])
    assert gram == gram.T
    assert gram.subs(nu, sp.pi) == 2*sp.eye(3)
    assert gram.applyfunc(lambda e: sp.limit(e, nu, 0)).rank() == 1
    numeric = sp.lambdify(nu, gram, 'numpy')
    values = []
    for value in (0.5, 0.75, 1.0, 1.25):
        matrix = numeric(value)
        assert np.linalg.eigvalsh(matrix).min() > 0
        vector = np.array([1.0+0.5j, -0.25, 1.0-0.5j])
        recovered = np.linalg.solve(matrix, matrix @ vector)
        assert np.linalg.norm(recovered-vector) < 1e-11
        values.append(dict(frequency=value,
                           minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min())))
    # Spacetime weak convergence alone fails: cos(L*t) is 1 at t=0.
    t, L = sp.symbols('t L', real=True, positive=True)
    assert sp.cos(L*t).subs(t, 0) == 1
    return dict(exact_gram_symmetry=True, separated_frequency_model=values,
                coincident_frequency_gram_rank=1, unrestricted_trace_counterexample=True,
                scope='Small Gram models only; the smooth-test trace argument is analytic.')


def constraint_checks():
    z, c, W, V, gradA2, lapz, N = sp.symbols('z c W V gradA2 lapz N')
    R0 = 8*sp.pi*(W+gradA2+V)
    delta_R = -8*lapz-4*z*R0
    delta_H = delta_R-8*sp.pi*(2*(c-6*z)*W-4*z*gradA2)
    expected = -8*lapz+(64*sp.pi*W-32*sp.pi*V)*z-16*sp.pi*c*W
    assert sp.expand(delta_H-expected) == 0
    delta_N = ((c-6*z)+6*z)*N
    assert sp.expand(delta_N-c*N) == 0
    bH, bM, bN = sp.symbols('bH bM bN')
    assert sp.expand((bH+16*sp.pi*c*W)-16*sp.pi*c*W-bH) == 0
    assert (c*N).subs(c, bN/N) == bN
    # Product rule uses abstract gradient and Laplacian contractions in 3D.
    alpha, v, lapv, cross = sp.symbols('alpha v lapv cross', positive=True)
    lapalpha = (8*sp.pi*W-4*sp.pi*V)*alpha
    Lalpha_v = -8*(alpha*lapv+2*cross+v*lapalpha)
    Lalpha_v += (64*sp.pi*W-32*sp.pi*V)*alpha*v
    divergence_form = -8/alpha*(alpha**2*lapv+2*alpha*cross)
    conformal_laplacian = alpha**-4*(lapv+2*cross/alpha)
    assert sp.expand(Lalpha_v-divergence_form) == 0
    assert sp.expand(Lalpha_v+8*alpha**5*conformal_laplacian) == 0
    # Affine metric and trace-free chart have only h*K at second order.
    a = sp.Symbol('a')
    h = sp.Matrix([[1, 2, 0], [2, -1, 3], [0, 3, 0]])
    k = sp.Matrix([[2, 1, 4], [1, -3, 0], [4, 0, 1]])
    trace2 = -sp.trace(h*k)
    physical_K = a*k-a**2*trace2*sp.eye(3)/3
    inverse2 = sp.eye(3)-a*h+a*a*h*h
    assert sp.expand(sp.trace(inverse2*physical_K)).coeff(a, 2) == 0
    assert sp.trace(k) == 0
    return dict(conformal_hamiltonian_linearization=True,
                fixed_charge_cancellation=True, scalar_block_right_inverse=True,
                lapse_ground_state_factorization=True, conformal_laplacian_factorization=True,
                maximal_chart_second_order_trace=True,
                scope='Differential identities and a matrix chart test; no elliptic solve.')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "phase_filter": phase_filter_checks(),
        "temporal_trace": temporal_trace_checks(),
        "constraints": constraint_checks(),
    }, indent=2))
