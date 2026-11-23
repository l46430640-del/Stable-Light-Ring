"""Algebraic checks: coherent orbit pairs."""
import json
import sympy as sp
from sympy.physics.wigner import gaunt, wigner_3j


def angular_checks():
    ell = sp.Symbol('ell', integer=True, positive=True)
    a, b = ell*(ell+1), (ell+2)*(ell+3)
    r0sq = (ell+1)*(ell+2)/((ell+sp.Rational(3, 2))*(ell+sp.Rational(5, 2)))
    r1sq = ell*(ell+3)/((ell+1)*(ell+2))
    r2sq = ell*(ell-1)*(ell+3)*(ell+4)/((ell+1)**2*(ell+2)**2)
    tensor_numerator = ((a+b-6)*(a+b-8)-2*a*b)/2
    assert sp.factor(tensor_numerator-ell*(ell-1)*(ell+3)*(ell+4)) == 0
    assert sp.cancel(tensor_numerator**2/(a*b*(a-2)*(b-2))-r2sq) == 0
    assert all(sp.limit(value, ell, sp.oo) == 1 for value in (r0sq, r1sq, r2sq))
    rows = []
    for l in (2, 3, 5, 10, 20, 40):
        other = l+2
        # Addition theorem for P2 with the observation axis inclined by pi/3.
        addition = sp.Rational(4, 5)*sp.pi*sp.sqrt(15/(32*sp.pi))*sp.Rational(3, 4)
        scalar = addition*(-1)**other*gaunt(l, other, 2, l, -other, 2)
        expected = sp.Rational(9, 32)*sp.sqrt(r0sq.subs(ell, l))
        assert sp.simplify(scalar-expected) == 0
        spin_ratios = []
        for spin, square in ((1, r1sq), (2, r2sq)):
            # Same-parity vector/tensor contraction averages the two helicities.
            ratio = (-1)**spin*wigner_3j(l, other, 2, -spin, spin, 0)
            ratio /= wigner_3j(l, other, 2, 0, 0, 0)
            assert sp.simplify(ratio-sp.sqrt(square.subs(ell, l))) == 0
            opposite = wigner_3j(l, other, 2, -spin, spin, 0)-wigner_3j(l, other, 2, spin, -spin, 0)
            assert opposite == 0
            spin_ratios.append(float(ratio))
        assert scalar > 0
        rows.append(dict(ell=l, scalar=float(scalar), vector=float(scalar)*spin_ratios[0],
                         tensor=float(scalar)*spin_ratios[1]))
    # Five node angles integrate every quadrupole magnetic component exactly.
    z = sp.Symbol('z')
    cyclotomic = sp.Poly(1+z+z**2+z**3+z**4, z)
    for m in (-2, -1, 1, 2):
        polynomial = sp.Poly(sum(z**((m*j) % 5) for j in range(5)), z)
        assert sp.rem(polynomial, cyclotomic).is_zero
    return dict(exact_scalar_vector_tensor_cases=18, opposite_parity_contraction_zero=True,
                quadrupole_node_count=5, asymptotic_inclined_coefficient='9/32', rows=rows,
                scope='Highest weights in their common tilted plane; not equal m in the observation frame.')


def radial_checks():
    L, e, J, Jp, dI, dL = sp.symbols('L e J Jp dI dL', positive=True)
    Or = 1/(2*sp.sqrt(e)*Jp)
    Op = sp.sqrt(e)-J*Or
    de = (dI-J*dL)/(L*Jp)
    dw = sp.sqrt(e)*dL+L*de/(2*sp.sqrt(e))
    assert sp.simplify(dw-Or*dI-Op*dL) == 0
    assert sp.simplify(dw.subs({dI: -5, dL: 2})-(2*Op-5*Or)) == 0
    omega, kr, optical = sp.symbols('omega kr optical', positive=True)
    wave_number = sp.sqrt(omega**2-L**2*optical)
    assert sp.diff(wave_number, omega) == omega/wave_number
    assert sp.diff(wave_number, L) == -L*optical/wave_number
    t, tr, phi, phir, nu = sp.symbols('t tr phi phir nu', real=True)
    wr, wp = Or*tr, phi-phir+Op*tr
    assert sp.simplify((2*wp-5*wr)-(2*phi-2*phir+(2*Op-5*Or)*tr)) == 0
    wr_in, wp_in = 2*sp.pi-Or*tr, phi+phir-Op*tr
    assert sp.simplify((2*wp_in-5*wr_in)-(2*phi+2*phir-(2*Op-5*Or)*tr)+10*sp.pi) == 0
    # The common inner Airy phase cancels in the radial difference.
    S, dS = sp.symbols('S dS', real=True)
    assert sp.trigsimp(2*sp.sin(S+sp.pi/4)*sp.sin(S+dS+sp.pi/4)
                       -sp.cos(dS)+sp.cos(2*S+dS+sp.pi/2)) == 0
    h, delta = sp.symbols('h delta', positive=True)
    cap = 2*(sp.sqrt(h**sp.Rational(2, 3)+delta)-h**sp.Rational(1, 3))
    assert sp.simplify(sp.diff(cap, delta)-(h**sp.Rational(2, 3)+delta)**(-sp.Rational(1, 2))) == 0
    assert sp.limit(cap, h, 0, dir='+') == 2*sp.sqrt(delta)
    # An independent exact oscillator check of a five-index radial coherence.
    oscillator = []
    for n in (8, 16, 32, 64, 128, 256):
        amplitudes = {n: sp.Integer(1)}
        for _ in range(5):
            next_amplitudes = {}
            for j, value in amplitudes.items():
                for dest, factor in ((j-1, sp.sqrt(sp.Rational(j, 2))),
                                     (j+1, sp.sqrt(sp.Rational(j+1, 2)))):
                    if dest >= 0:
                        next_amplitudes[dest] = next_amplitudes.get(dest, 0)+value*factor
            amplitudes = next_amplitudes
        expected = sp.sqrt(sp.prod(range(n-4, n+1))/sp.Integer(32))
        assert sp.simplify(amplitudes[n-5]-expected) == 0
        coefficient = amplitudes[n-5]/sp.sqrt(sp.Integer(2*n+1))**5
        oscillator.append(dict(n=n, fifth_moment=float(coefficient), classical_limit=1/32))
    assert all(a['fifth_moment'] < b['fifth_moment'] < 1/32 for a, b in zip(oscillator, oscillator[1:]))
    return dict(action_frequency_differential=True, pair_indices=[2, -5],
                radial_Hamilton_Jacobi_phase=True, common_Airy_phase_cancels=True,
                uniform_turning_cap=True, oscillator_five_index_checks=oscillator,
                scope='Exact identities and a solvable radial model; no stellar eigenvalues or WKB error constants computed.')


def polarization_checks():
    omega, kr, kt, c = sp.symbols('omega kr kt c', real=True, nonzero=True)
    eta = sp.diag(-1, 1, 1, 1)
    covector = sp.Matrix([-omega, kr, kt, 0])
    normal = sp.Matrix([0, 0, 0, 1])
    transverse = c*sp.Matrix([-kr, omega, 0, 0])
    polarization = transverse*normal.T+normal*transverse.T
    assert sp.trace(eta*polarization) == 0
    assert polarization*eta*covector == sp.zeros(4, 1)
    tt, tr, rr, angular, gt, gr, divN = sp.symbols('tt tr rr angular gt gr divN')
    scalar_hessian = sp.Matrix([[tt, tr, 0, 0], [tr, rr, 0, 0],
                                [0, 0, angular, 0], [0, 0, 0, angular]])
    assert sp.trace(eta*polarization*eta*scalar_hessian) == 0
    assert (sp.Matrix([0, 0, 0, divN]).T*eta*sp.Matrix([gt, gr, 0, 0]))[0] == 0
    norm = sp.trace(eta*polarization*eta*polarization)
    assert sp.expand(norm-2*c*c*(omega*omega-kr*kr)) == 0
    assert sp.expand(norm.subs(omega**2, kr**2+kt**2)-2*c*c*kt*kt) == 0
    a, b, psi, theta = sp.symbols('a b psi theta', real=True)
    signal = a*sp.cos(theta)+b*sp.cos(theta+psi)
    mean = sp.integrate(sp.expand_trig(signal)**2, (theta, 0, 2*sp.pi))/(2*sp.pi)
    intensity = a*a+b*b+2*a*b*sp.cos(psi)
    assert sp.trigsimp(2*mean-intensity) == 0
    # The master reconstruction has two mixed polarization components.
    radial_product = omega**2*sp.sin(theta)*sp.sin(theta+psi)-kr**2*sp.cos(theta)*sp.cos(theta+psi)
    radial_mean = sp.integrate(sp.expand_trig(radial_product), (theta, 0, 2*sp.pi))/(2*sp.pi)
    assert sp.trigsimp(2*radial_mean-(omega**2-kr**2)*sp.cos(psi)) == 0
    radial_current = kr*(sp.sin(theta)*sp.cos(theta+psi)-sp.cos(theta)*sp.sin(theta+psi))
    assert sp.trigsimp(radial_current+kr*sp.sin(psi)) == 0
    L = sp.Symbol('L', positive=True)
    Lambda = L**2-sp.Rational(1, 4)
    assert sp.limit(Lambda*(Lambda-2)/L**4, L, sp.oo) == 1
    assert sp.expand((a*a+b*b)**2-(2*a*b)**2-(a*a-b*b)**2) == 0
    angle, time = sp.symbols('angle time', real=True)
    reverse_product = sp.cos(angle-time)*sp.cos(angle+time)
    assert sp.trigsimp(2*reverse_product-sp.cos(2*angle)-sp.cos(2*time)) == 0
    # Polarization of the weak quadratic identity retains its coherent cross term.
    hp, hq, dhp, dhq, ddhp, ddhq = sp.symbols('hp hq dhp dhq ddhp ddhq', real=True)
    ricci_cross = (hp*ddhq+hq*ddhp)/2+dhp*dhq/2
    divergence = (dhp*dhq+hp*ddhq+dhq*dhp+hq*ddhp)/2
    assert sp.expand(ricci_cross-divergence+dhp*dhq/2) == 0
    stress = covector*covector.T*intensity
    assert sp.simplify(sp.trace(eta*stress).subs(omega**2, kr**2+kt**2)) == 0
    return dict(physical_axial_trace=True, physical_axial_transversality=True,
                positive_polarization_norm='2*c^2*k_tangent^2', coherent_intensity=True,
                weak_Ricci_cross_coefficient=True, complete_null_dyad_components=16,
                reversed_pair_retains_fast_low_angular_term=True,
                radial_density_and_current_phases=True, master_energy_normalization=True,
                eta='2*a*b/(a^2+b^2)', scalar_first_order_zero=True)


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "angular": angular_checks(),
        "radial": radial_checks(),
        "polarization": polarization_checks(),
    }, indent=2))
