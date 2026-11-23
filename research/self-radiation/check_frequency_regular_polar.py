"""Independent Fourier-domain Einstein--scalar identities in physical variables.

This checks local equations, not the outgoing stellar boundary-value problem.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from mpmath import iv
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
REPO = ROOT.parent.parent
sys.path.insert(0, str(OLD))
from check_barrier_certificate import enclosure, exact
from check_asymptotic_exterior import lower
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs, physical_cell
from check_regular_center import leading
from check_stationary_phase import center_bound
from trapped_axial import Background

if not __debug__:
    raise SystemExit('Run checks without python -O.')


def curvature():
    r, theta = sp.symbols('r theta', positive=True)
    t, phi, nu, omega, eigenvalue = sp.symbols('t phi nu omega Lambda', real=True)
    coordinates = (t, r, theta, phi)
    N, f, H, K, T = [sp.Function(name)(r) for name in ('N', 'f', 'H', 'K', 'T')]
    Y = sp.Function('Y')(theta)
    alpha2 = sp.exp(2*N)
    diagonal = [-alpha2, 1/f, r*r, r*r*sp.sin(theta)**2]
    h = sp.diag(alpha2*H*Y, H*Y/f, r*r*K*Y, r*r*sp.sin(theta)**2*K*Y)
    h[0, 1] = h[1, 0] = T*Y
    harmonic = {sp.diff(Y, theta, 2): -sp.cot(theta)*sp.diff(Y, theta)-eigenvalue*Y}

    def derivative(expr, index):
        return -sp.I*nu*expr if index == 0 else sp.diff(expr, coordinates[index])

    gamma, delta = {}, {}
    for a in range(4):
        for b in range(4):
            for c in range(4):
                gamma[a, b, c] = (int(a == c)*sp.diff(diagonal[a], coordinates[b])
                                  +int(a == b)*sp.diff(diagonal[a], coordinates[c])
                                  -int(b == c)*sp.diff(diagonal[b], coordinates[a]))/(2*diagonal[a])
    for a in range(4):
        for b in range(4):
            for c in range(4):
                delta[a, b, c] = sp.expand((derivative(h[a, c], b)+derivative(h[a, b], c)
                                           -derivative(h[b, c], a))/(2*diagonal[a])
                                          -sum(h[a, j]*gamma[j, b, c]/diagonal[a] for j in range(4)))

    def ricci(a, b):
        value = sum(derivative(delta[j, a, b], j)-derivative(delta[j, a, j], b)
                    +sum(delta[j, j, k]*gamma[k, a, b]+gamma[j, j, k]*delta[k, a, b]
                         -delta[j, b, k]*gamma[k, a, j]-gamma[j, b, k]*delta[k, a, j]
                         for k in range(4)) for j in range(4))
        return sp.expand(value).subs(harmonic)

    n, fp = sp.diff(N, r), sp.diff(f, r)
    F = f*(sp.diff(N, r, 2)+n*n+(2/r+fp/(2*f))*n)
    R = 1-f-r*f*n-r*fp/2
    background = [alpha2*F, -sp.diff(N, r, 2)-n*n-(n+2/r)*fp/(2*f), R, R*sp.sin(theta)**2]
    diagonal_ricci = [ricci(j, j) for j in range(4)]
    delta_scalar = sum(diagonal_ricci[j]/diagonal[j]-h[j, j]*background[j]/diagonal[j]**2
                       for j in range(4))
    rr = (f*diagonal_ricci[1]-H*Y*f*background[1]-delta_scalar/2)/Y
    rt = ricci(1, 2)/sp.diff(Y, theta)
    ta = ricci(0, 2)/sp.diff(Y, theta)
    tt = sp.expand(diagonal_ricci[0]/(alpha2*Y)).subs(sp.cot(theta), sp.cos(theta)/sp.sin(theta))
    Hp, Kp, Tp = [sp.diff(q, r) for q in (H, K, T)]
    expected_rr = (-f*Hp/r+f*(n+1/r)*Kp+(eigenvalue/(2*r*r)-2*f*n/r-f/r**2)*H
                   +(nu*nu/alpha2-(eigenvalue-2)/(2*r*r))*K-2*sp.I*nu*f*T/(r*alpha2))
    expected_tt = (-f*sp.diff(H, r, 2)/2-f*(3*n/2+1/r+fp/(4*f))*Hp+f*n*Kp
                   -2*F*H+eigenvalue*H/(2*r*r)+nu*nu/alpha2*(H/2+K)
                   -sp.I*nu*f/alpha2*(Tp+(2/r+fp/(2*f))*T))
    pairs = dict(mixed_radial_constraint=(rr, expected_rr),
                 radial_angular=(rt, (Hp-Kp)/2+n*H+sp.I*nu*T/(2*alpha2)),
                 time_angular=(ta, f*Tp/2+(f*n/2+fp/4)*T+sp.I*nu*(H+K)/2),
                 time_radial=(ricci(0, 1)/Y, (eigenvalue/(2*r*r)-F)*T
                              +sp.I*nu*(Kp+(1/r-n)*K-H/r)),
                 time_time=(tt, expected_tt),
                 angular_tracefree=(diagonal_ricci[2]-diagonal_ricci[3]/sp.sin(theta)**2, 0))
    for name, (value, expected) in pairs.items():
        defect = sp.expand(value-expected).subs(sp.cot(theta), sp.cos(theta)/sp.sin(theta))
        assert sp.trigsimp(sp.expand(defect)) == 0, name
    Et, Pa = sp.symbols('Et Pa')
    remaining_defect = [Et, 0, Pa, Pa]
    angular_divergence = Pa*sp.diff(Y, theta)+sum(
        gamma[j, j, 2]*(Pa-remaining_defect[j])*Y for j in range(4))
    assert sp.expand(angular_divergence-Pa*sp.diff(Y, theta)) == 0

    # Vary the divergence form of Box(Phi), retaining both scalar sidebands.
    A, U, W = [sp.Function(name)(r) for name in ('A', 'U', 'W')]
    Vs, Vss = sp.symbols('Vs Vss')
    Ap, d, O = sp.diff(A, r), 2/r+n+fp/(2*f), omega**2/alpha2
    volume = r*r*sp.exp(N)/sp.sqrt(f)
    raw = []
    for sign in (1, -1):
        q = U+sign*sp.I*W
        metric_time = ((nu+sign*omega)*sign*omega*(H+K)*A
                       -sp.I*(nu+sign*omega)*f*T*Ap)/alpha2
        metric_radial = sp.diff(volume*f*((K-H)*Ap-sign*sp.I*omega*T*A/alpha2), r)/volume
        raw.append(f*(sp.diff(q, r, 2)+d*sp.diff(q, r))
                   +((nu+sign*omega)**2/alpha2-eigenvalue/r**2-Vs)*q-2*A*A*Vss*U
                   +metric_time+metric_radial-K*Vs*A)
    kgU = (f*(sp.diff(U, r, 2)+d*sp.diff(U, r))
           +(O+nu*nu/alpha2-Vs-2*A*A*Vss-eigenvalue/r**2)*U+2*sp.I*nu*omega*W/alpha2
           +f*(Kp-Hp)*Ap+(2*O-Vs)*A*H-sp.I*nu*f*T*Ap/alpha2)
    kgW = (f*(sp.diff(W, r, 2)+d*sp.diff(W, r))
           +(O+nu*nu/alpha2-Vs-eigenvalue/r**2)*W-2*sp.I*nu*omega*U/alpha2
           -sp.I*nu*omega*A*(H+K)/alpha2
           -f*omega/alpha2*(2*T*Ap+A*(Tp+(d-2*n)*T)))
    for calculated, expected in (((raw[0]+raw[1])/2, kgU), ((raw[0]-raw[1])/(2*sp.I), kgW)):
        defect = sp.expand(calculated-expected).subs(sp.diff(A, r, 2), -d*Ap+(Vs-O)*A/f)
        assert sp.expand(defect) == 0
    return {**{key: True for key in pairs}, 'angular_Bianchi_closure': True,
            'both_scalar_sidebands_from_divergence_form': True}


def matter():
    s, f, A, Ap, U, Up, W, Wp, H, T, nu, omega, V, Vs = sp.symbols(
        's f A Ap U Up W Wp H T nu omega V Vs', nonzero=True)
    g = sp.diag(-s, 1/f)
    inv = sp.diag(-1/s, f)
    h = sp.Matrix([[s*H, T], [T, H/f]])
    delta_inv = -inv*h*inv
    base = sp.Matrix([-sp.I*omega*A, Ap])
    bar = sp.Matrix([sp.I*omega*A, Ap])
    pert = sp.Matrix([-sp.I*(nu+omega)*(U+sp.I*W), Up+sp.I*Wp])
    barpert = sp.Matrix([-sp.I*(nu-omega)*(U-sp.I*W), Up-sp.I*Wp])
    norm = (bar.T*inv*base)[0]
    delta_norm = (barpert.T*inv*base+bar.T*inv*pert+bar.T*delta_inv*base)[0]
    stress = (bar*base.T+base*bar.T-g*(norm+V))/2
    kinetic = (barpert*base.T+bar*pert.T+pert*bar.T+base*barpert.T)/2
    variation = kinetic-h*(norm+V)/2-g*(delta_norm+2*Vs*A*U)/2
    pressure = (inv*variation+delta_inv*stress)[1, 1]
    expected = f*Ap*Up+(omega**2*A*A/s-f*Ap*Ap)*H/2+(omega**2/s-Vs)*A*U+sp.I*nu*omega*A*W/s
    assert sp.expand(pressure-expected) == 0
    trace_reversed = kinetic+V*h/2+g*Vs*A*U
    assert sp.expand(trace_reversed[0, 0]/s-(2*omega**2/s-Vs)*A*U
                     -2*sp.I*nu*omega*A*W/s-V*H/2) == 0
    assert sp.expand(trace_reversed[0, 1]+sp.I*nu*Ap*U-omega*(Ap*W-A*Wp)-V*T/2) == 0
    return dict(radial_pressure=True, time_time_stress=True, radial_current=True)


def reduction():
    r, s, f, n, A, Ap, Vs, Vss, omega, nu, L, k = sp.symbols(
        'r s f n A Ap Vs Vss omega nu Lambda kappa', nonzero=True)
    H, Hp, K, T, U, Up, W, Wp = state = sp.symbols('H Hp K T U Up W Wp')
    O = omega**2/s
    V = O*A*A+f*Ap*Ap-2*(2*f*r*n-1+f)/(k*r*r)
    fp = 2*f*n-k*r*(O*A*A+f*Ap*Ap)
    d = 2/r+n+fp/(2*f)
    np_ = k*(2*O*A*A-V)/(2*f)-n*n-(2/r+fp/(2*f))*n
    App = -d*Ap+(Vs-O)*A/f
    B, C = 1/r-n, L-2*k*r*r*O*A*A
    D, E = L-2-2*nu*nu*r*r/s, C-2*nu*nu*r*r/s
    a, b = 2*sp.I*nu*r*r*f*B/s, 2*sp.I*nu*r*r*B
    Rk = (2*r*r*f*n*Hp+(4*r*r*f*n*n+L-2*f-k*r*r*(O*A*A-f*Ap*Ap))*H
          -4*k*r*r*f*(n+1/r)*Ap*U-2*k*r*r*(O-Vs)*A*U-2*k*r*r*f*Ap*Up
          -2*k*sp.I*nu*r*r*omega*A*W/s)
    Rt = -2*k*omega*r*r*(A*Wp-Ap*W)-2*sp.I*nu*r*r*(Hp+(2*n-1/r)*H-k*Ap*U)
    constraints = sp.Matrix([D*K+a*T-Rk, b*K+E*T-Rt])
    determinant = sp.expand(D*E-a*b)
    cross = 4*n*Ap+2*A*(2*O-Vs)/f
    qH = 4*n*n+L/(r*r*f)+k*(V-4*O*A*A)/f
    qU = (L/r**2+Vs+2*A*A*Vss-O)/f+2*k*Ap*Ap
    evolution = [Hp, -d*Hp+(qH-nu*nu/(f*s))*H-k*cross*U-4*sp.I*nu*B*T/s,
                 Hp+2*n*H-2*k*Ap*U+sp.I*nu*T/s,
                 -(d-2/r)*T-sp.I*nu*(H+K)/f-2*k*omega*A*W/f,
                 Up, -d*Up+(qU-nu*nu/(f*s))*U-cross*H/2-2*sp.I*nu*omega*W/(f*s),
                 Wp, -d*Wp+(L/r**2+Vs-O-nu*nu/s-2*k*O*A*A)*W/f
                 +2*sp.I*nu*omega*U/(f*s)+2*omega*(Ap+B*A)*T/s]
    rates = dict(zip(state, evolution))
    rates.update({r: 1, s: 2*n*s, f: fp, n: np_, A: Ap, Ap: App, Vs: 2*Vss*A*Ap})
    prime = constraints.applyfunc(lambda expr: sp.expand(sum(sp.diff(expr, x)*dx for x, dx in rates.items())))
    propagation = sp.Matrix([[0, sp.I*nu/s], [-sp.I*nu/f, -(d-2/r)]])
    defect = (prime-propagation*constraints).jacobian(state)
    for row in range(2):
        for col in range(8):
            value = sp.cancel(defect[row, col])
            assert value == 0, (row, col, value)
    assert sp.expand(determinant-(D*E+4*nu*nu*r**4*f*B*B/s)) == 0
    assert sp.expand(determinant.subs(nu, 0)-(L-2)*C) == 0
    Hpp, Kp, Tp, Upp, Wpp = [evolution[j] for j in (1, 2, 3, 5, 7)]
    F = k*(2*O*A*A-V)/2
    rr = (-f*Hp/r+f*(n+1/r)*Kp+(L/(2*r*r)-2*f*n/r-f/r**2)*H
          +(nu*nu/s-(L-2)/(2*r*r))*K-2*sp.I*nu*f*T/(r*s)
          -k*(f*Ap*Up+(O*A*A-f*Ap*Ap)*H/2+(O-Vs)*A*U+sp.I*nu*omega*A*W/s))
    tr = C*T/(2*r*r)+sp.I*nu*(Kp+B*K-H/r+k*Ap*U)+k*omega*(A*Wp-Ap*W)
    tt = (-f*Hpp/2-f*(3*n/2+1/r+fp/(4*f))*Hp+f*n*Kp-2*F*H+L*H/(2*r*r)
          +nu*nu/s*(H/2+K)-sp.I*nu*f/s*(Tp+(2/r+fp/(2*f))*T)
          -k*((2*O-Vs)*A*U+2*sp.I*nu*omega*A*W/s+V*H/2))
    kgU = (f*(Upp+d*Up)+(O+nu*nu/s-Vs-2*A*A*Vss-L/r**2)*U+2*sp.I*nu*omega*W/s
           +f*(Kp-Hp)*Ap+(2*O-Vs)*A*H-sp.I*nu*f*T*Ap/s)
    kgW = (f*(Wpp+d*Wp)+(O+nu*nu/s-Vs-L/r**2)*W-2*sp.I*nu*omega*U/s
           -sp.I*nu*omega*A*(H+K)/s-f*omega/s*(2*T*Ap+A*(Tp+(d-2*n)*T)))
    for value in (rr+constraints[0]/(2*r*r), tr-constraints[1]/(2*r*r), tt, kgU, kgW):
        assert sp.expand(value) == 0

    # The six-dimensional chart retains all three regular central amplitudes.
    reconstructed_K = (E*Rk-a*Rt)/determinant
    reconstructed_T = (D*Rt-b*Rk)/determinant
    reflection = {r: -r, n: -n, Ap: -Ap, Hp: -Hp, Up: -Up, Wp: -Wp}
    assert sp.cancel(reconstructed_K.subs(reflection, simultaneous=True)-reconstructed_K) == 0
    assert sp.cancel(reconstructed_T.subs(reflection, simultaneous=True)+reconstructed_T) == 0
    p, h0, u0, w0, s0, A0, a2, f2, n1 = sp.symbols('p h0 u0 w0 s0 A0 a2 f2 n1', nonzero=True)
    central = {H: h0, Hp: p*h0/r, U: u0, Up: p*u0/r, W: w0, Wp: p*w0/r,
               s: s0, A: A0+a2*r*r, Ap: 2*a2*r, f: 1+f2*r*r, n: n1*r}
    Kcentral = reconstructed_K.subs(central, simultaneous=True)
    Tcentral = reconstructed_T.subs(central, simultaneous=True)
    assert sp.simplify(sp.limit(Kcentral, r, 0)-h0) == 0
    assert sp.limit(Tcentral, r, 0) == 0
    for value, amplitude in ((Hpp, h0), (Upp, u0), (Wpp, w0)):
        rhs = value.subs(central, simultaneous=True).subs(T, Tcentral)
        assert sp.simplify(sp.limit(r*r*rhs, r, 0)-(L-2*p)*amplitude) == 0
    ell = sp.Symbol('ell', integer=True, positive=True)
    assert sp.factor((p*p+p-L).subs(L, ell*(ell+1))) == (p-ell)*(p+ell+1)
    return dict(algebraic_constraints=2, independent_second_order_equations=3,
                constraint_propagation_entries=16, reconstructed_field_equations=5,
                determinant_at_zero='(Lambda-2)*(Lambda-Xi)', regular_central_amplitudes=3,
                central_indicial_roots=['ell', '-ell-1'], central_reconstruction_parity=True,
                no_division_by_frequency=True,
                no_division_by_scalar_amplitude=True)


def core_bound(digits, max_seconds):
    iv.dps = digits
    hashes = check_inputs()
    phase_path = ROOT/'data/stationary-phase-50.json'
    phase = json.loads(phase_path.read_text())
    assert phase['status'] == 'stationary_phase_form_positive_for_all_l_at_least_2'
    assert phase['input_sha256'] == hashes
    assert all(hashlib.sha256((REPO/key).read_bytes()).hexdigest() == value
               for key, value in phase['dependency_sha256'].items())
    background = json.loads((OLD/'data/s08a06/global-background.json').read_text())
    omega = enclosure(*background['endpoint_graphs']['frequency_bound_on_the_graph'])
    bg = Background()
    center = center_bound(bg)
    nominal = [iv.mpf(1), exact(float(bg.sol(0)[3]))]
    center_leading = leading(nominal, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
    r0 = exact(center['radius'])
    zeta = nominal[1]+enclosure(-iv.mpf('2.1e-8'), iv.mpf('2.1e-8')) \
           +enclosure(0, r0*r0)*(center_leading[3]+enclosure(-iv.mpf('1e-6'), iv.mpf('1e-6')))
    maximum = upper(2*r0*r0*iv.exp(2*zeta)/(omega*omega))
    center_maximum = maximum
    knots = [Fraction(float(x)) for x in bg.sol.x]
    start = perf_counter()
    cells = 0
    for left, right in zip(knots[1:-1], knots[2:]):
        assert perf_counter()-start < max_seconds, 'Core interval time budget exhausted.'
        piece = bisect_right(knots, (left+right)/2)-1
        y = physical_cell(bg, piece, left, right)
        radius = enclosure(left, right)
        maximum = max(maximum, upper(2*radius*radius*iv.exp(2*y[3])/(omega*omega)))
        cells += 1
    assert cells+1 <= 4000
    z = iv.mpf('.01')**2*exact(maximum)
    gap = 6-exact(phase['global_Xi_upper'])
    determinant_lower = (4-z)*(gap-z)
    assert exact(maximum) < iv.mpf('4461.207')
    assert z.b < iv.mpf('.446121') and gap.a > iv.mpf('3.200219')
    assert determinant_lower.a > iv.mpf('9.78')
    return dict(radius=40, frequency_absolute_upper='.01', cells=cells+1,
                center_two_r_squared_over_lapse_squared_upper=center_maximum,
                two_r_squared_over_lapse_squared_upper=maximum, z_upper=upper(z),
                determinant_lower=lower(determinant_lower), determinant_strict_lower_display='9.78',
                phase_certificate_sha256=hashlib.sha256(phase_path.read_bytes()).hexdigest(),
                dependency_sha256={**phase['dependency_sha256'],
                    str((ROOT/'check_stationary_phase.py').relative_to(REPO)):
                    hashlib.sha256((ROOT/'check_stationary_phase.py').read_bytes()).hexdigest()},
                input_sha256=hashes, radial_solves=0, quadrature_points=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--max-seconds', type=int, default=60)
    parser.add_argument('--output', type=Path, default=ROOT/'data/frequency-regular-polar.json')
    args = parser.parse_args()
    start = perf_counter()
    checks = dict(curvature=curvature(), matter=matter(), reduction=reduction())
    if args.checks_only:
        print(json.dumps(checks, indent=2))
    else:
        result = dict(status='frequency_regular_six_dimensional_core_verified', digits=args.digits,
                      scope='Homogeneous polar core and regular central subspace. Coupled outgoing exterior and forced response still open.',
                      checks=checks, core=core_bound(args.digits, args.max_seconds))
        result['elapsed_seconds'] = perf_counter()-start
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
        print(json.dumps(result, indent=2))
