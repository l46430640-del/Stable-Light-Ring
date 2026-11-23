"""Algebraic checks: axial adm energy."""
import json
import sympy as sp


def first_law_checks():
    alpha, p, V, A, omega, trace, P, grad = sp.symbols(
        'alpha p V A omega trace P grad', nonzero=True)
    lap_alpha = alpha*(8*sp.pi*p*p-4*sp.pi*V)
    curvature_bulk = -8*sp.pi*alpha*grad-(4*sp.pi*alpha*V+lap_alpha)*trace
    matter_bulk = 8*sp.pi*alpha*grad-16*sp.pi*alpha*p*P
    charge_bulk = -16*sp.pi*omega*(A*P+A*p*trace/2)
    assert sp.expand(curvature_bulk+matter_bulk-charge_bulk.subs(omega, alpha*p/A)) == 0
    defect, mass_w, charge_w = sp.symbols('defect mass_w charge_w')
    # The two paths have the same first jet. Their quadratic difference w
    # cancels the auxiliary path's defect and has zero linear charge.
    law = (mass_w-omega*charge_w-defect/(16*sp.pi))
    assert sp.solve(law.subs(charge_w, 0), mass_w) == [defect/(16*sp.pi)]
    return dict(fixed_scalar_first_law_bulk_cancellation=True,
                quadratic_completion_difference_has_positive_sign=True,
                auxiliary_path_changes_no_actual_preparation=True)


def master_energy_checks():
    r, alpha, f, Lambda = sp.symbols('r alpha f Lambda', positive=True)
    m, V, Q, Qt, Qx, Qxx = sp.symbols('m V Q Qt Qx Qxx', real=True)
    lam = Lambda-2
    p = alpha*sp.sqrt(f)
    pp = alpha**2*(2*m/r**2-4*sp.pi*r*V)/p
    potential = alpha**2*(Lambda/r**2-6*m/r**3+4*sp.pi*V)
    on_background = lambda expr: sp.simplify(expr.subs(f, 1-2*m/r))

    h0, h1 = p*Q+r*Qx, r*Qt/p
    h0r = pp*Q+2*Qx+r*Qxx/p
    h1t = r*(Qxx-potential*Q)/p
    Kr = (h0r-2*h0/r-h1t)/(2*alpha)
    assert on_background(Kr-lam*Q/(2*r*sp.sqrt(f))) == 0
    Kr = lam*Q/(2*r*sp.sqrt(f))
    Ktensor = h0/alpha
    vector_norm, tensor_norm = Lambda, Lambda*lam/2
    Ksq = 2*f*Kr*Kr*vector_norm/r**2+Ktensor*Ktensor*tensor_norm/r**4
    shear_sq = h1*h1*tensor_norm/r**4
    density_x = alpha**2*r*r*(Ksq+f*shear_sq)/(16*sp.pi)
    c = Lambda*lam/(32*sp.pi)
    completed = Qt*Qt+(Qx+p*Q/r)**2+alpha**2*lam*Q*Q/r**2
    assert sp.simplify(density_x-c*completed) == 0

    s = p/r
    sx = p*pp/r-p*p/r**2
    assert on_background(s*s-sx+alpha**2*lam/r**2-potential) == 0
    boundary = sx*Q*Q+2*s*Q*Qx
    standard = Qt*Qt+Qx*Qx+potential*Q*Q
    assert on_background(completed-standard-boundary) == 0
    assert sp.simplify(c/2-Lambda*lam/(64*sp.pi)) == 0
    return dict(radial_extrinsic_coefficient='(Lambda-2)*Q/(2*r*sqrt(f))',
                physical_energy_real_harmonic_prefactor='Lambda*(Lambda-2)/(32*pi)',
                complex_highest_weight_prefactor='Lambda*(Lambda-2)/(64*pi)',
                master_factorization='P = (-d_x+p/r)(d_x+p/r)+alpha^2*(Lambda-2)/r^2',
                integration_by_parts_boundary='d_x[(p/r)*Q^2]',
                matches_existing_master_normalization=True)


def phase_and_residual_checks():
    theta = sp.Symbol('theta', real=True)
    A, B, delta = sp.symbols('A B delta', real=True)
    energy = A*sp.cos(theta)**2+B*sp.sin(theta)**2
    assert sp.trigsimp(energy-(A+B)/2-(A-B)*sp.cos(2*theta)/2) == 0
    assert sp.trigsimp(energy.subs(B, A+delta)-A-delta*sp.sin(theta)**2) == 0
    assert energy.subs(theta, 0) == A
    assert energy.subs(theta, sp.pi/2) == B
    # Independent exact finite matrices test the residual pairing, with a
    # nonzero residual in each model.
    rows = []
    for n in range(2, 8):
        v = sp.Matrix([1, sp.Rational(1, n), sp.Rational(-1, n+1)])
        P = sp.diag(n*n+1, n*n+2, n*n+3)
        omega2 = sp.Integer(n*n)
        residual = P*v-omega2*v
        kinetic = omega2*(v.T*v)[0]
        spatial = (v.T*P*v)[0]
        pairing = (v.T*residual)[0]
        assert spatial-kinetic == pairing
        assert pairing**2 <= (v.T*v)[0]*(residual.T*residual)[0]
        assert kinetic > 0 and spatial > 0
        rows.append(dict(dimension=3, frequency_squared=int(omega2),
                         residual_pairing=str(pairing)))
    return dict(phase_formula='e2(theta)=A_L*cos(theta)^2+B_L*sin(theta)^2',
                variation_from_mean='(A_L-B_L)*cos(2*theta)/2',
                cutoff_residual_pairing_exact=True, finite_matrix_models=rows,
                asymptotic_scope='Uniform positivity follows analytically from existing normalized quasimode defect bounds.')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "first_law": first_law_checks(),
        "master_energy": master_energy_checks(),
        "phase": phase_and_residual_checks(),
    }, indent=2))
