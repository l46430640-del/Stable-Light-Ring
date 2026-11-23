"""Check the frequency-regular polar equations for a conserved null stress.

Shifted derivative variables remove derivatives of the anisotropic stress.
The kinetic-source radiation result does not realize the source by waves.
"""

import argparse
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
from check_asymptotic_exterior import lower
from check_barrier_certificate import enclosure
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs

if not __debug__:
    raise SystemExit('Run checks without python -O.')


def anisotropic_curvature():
    r, theta = sp.symbols('r theta', positive=True)
    t, phi, nu, omega, L = sp.symbols('t phi nu omega Lambda', real=True)
    coordinates = (t, r, theta, phi)
    N, f, D, A = [sp.Function(name)(r) for name in ('N', 'f', 'D', 'A')]
    Y = sp.Function('Y')(theta)
    s = sp.exp(2*N)
    diagonal = [-s, 1/f, r*r, r*r*sp.sin(theta)**2]
    h = sp.diag(0, 2*D*Y/f, 0, 0)
    harmonic = {sp.diff(Y, theta, 2): -sp.cot(theta)*sp.diff(Y, theta)-L*Y}
    derivative = lambda value, j: -sp.I*nu*value if j == 0 else sp.diff(value, coordinates[j])
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
    background = [s*F, -sp.diff(N, r, 2)-n*n-(n+2/r)*fp/(2*f), R, R*sp.sin(theta)**2]
    diag_ricci = [ricci(j, j) for j in range(4)]
    delta_R = sum(diag_ricci[j]/diagonal[j]-h[j, j]*background[j]/diagonal[j]**2 for j in range(4))
    pairs = dict(
        time_time=(diag_ricci[0]/(s*Y), -f*n*sp.diff(D, r)-2*F*D+nu*nu*D/s),
        mixed_radial=(f*diag_ricci[1]/Y-2*D*f*background[1]-delta_R/(2*Y),
                      -2*f*(1/r**2+2*n/r)*D),
        radial_angular=(ricci(1, 2)/sp.diff(Y, theta), (n+1/r)*D),
        time_angular=(ricci(0, 2)/sp.diff(Y, theta), sp.I*nu*D),
        time_radial=(ricci(0, 1)/Y, -2*sp.I*nu*D/r),
        angular_tracefree=(diag_ricci[2]-diag_ricci[3]/sp.sin(theta)**2,
                           -D*(sp.diff(Y, theta, 2)-sp.cot(theta)*sp.diff(Y, theta))))
    for name, (value, expected) in pairs.items():
        difference = sp.expand(value-expected).subs(harmonic).subs(sp.cot(theta), sp.cos(theta)/sp.sin(theta))
        assert sp.trigsimp(sp.expand(difference)) == 0, name
    Vs = sp.Symbol('Vs')
    Ap, damping = sp.diff(A, r), 2/r+n+fp/(2*f)
    volume = r*r*sp.exp(N)/sp.sqrt(f)
    metric_KG = [(nu+sign*omega)*sign*omega*D*A/s
                 -sp.diff(volume*f*D*Ap, r)/volume-D*Vs*A for sign in (1, -1)]
    expected_U = -f*Ap*sp.diff(D, r)-2*(Vs-omega*omega/s)*A*D
    expected_W = -sp.I*nu*omega*A*D/s
    for actual, expected in (((metric_KG[0]+metric_KG[1])/2, expected_U),
                             ((metric_KG[0]-metric_KG[1])/(2*sp.I), expected_W)):
        difference = sp.expand(actual-expected).subs(sp.diff(A, r, 2), -damping*Ap+(Vs-omega*omega/s)*A/f)
        assert sp.expand(difference) == 0
    # The full source is projected before its angular divergence is reduced.
    rho, pressure, jt, j0, j1, tau = [sp.Function(name)(r) for name in ('rho', 'p', 'jt', 'j0', 'j1', 'tau')]
    q = (rho-pressure)/2
    stress = sp.diag(s*rho*Y, pressure*Y/f,
                     r*r*(q*Y+tau*(sp.diff(Y, theta, 2)+L*Y/2)),
                     r*r*sp.sin(theta)**2*(q*Y+tau*(sp.cot(theta)*sp.diff(Y, theta)+L*Y/2)))
    stress[0, 1] = stress[1, 0] = jt*Y
    stress[0, 2] = stress[2, 0] = j0*sp.diff(Y, theta)
    stress[1, 2] = stress[2, 1] = j1*sp.diff(Y, theta)
    mixed = sp.diag(*[1/value for value in diagonal])*stress
    harmonic3 = {sp.diff(Y, theta, 3): (1/sp.sin(theta)**2+sp.cot(theta)**2-L)*sp.diff(Y, theta)+L*sp.cot(theta)*Y}
    expected = [sp.I*nu*rho+f*(sp.diff(jt, r)+damping*jt)-L*j0/r**2,
                sp.diff(pressure, r)+n*(rho+pressure)+(3*pressure-rho)/r-L*j1/r**2+sp.I*nu*jt/s,
                f*(sp.diff(j1, r)+damping*j1)+q-(L-2)*tau/2+sp.I*nu*j0/s]
    for b in range(3):
        divergence = sum(derivative(mixed[a, b], a)+sum(gamma[a, a, c]*mixed[c, b]
                         -gamma[c, a, b]*mixed[a, c] for c in range(4)) for a in range(4))
        difference = divergence-expected[b]*(sp.diff(Y, theta) if b == 2 else Y)
        difference = sp.expand(difference).subs(harmonic3).subs(harmonic).subs(sp.cot(theta), sp.cos(theta)/sp.sin(theta))
        assert sp.trigsimp(sp.expand(difference)) == 0, b
    assert sp.expand(sp.trace(mixed)).subs(harmonic).expand() == 0
    # Direct inverse-metric contribution to the mixed scalar pressure.
    pr_background = (omega*omega*A*A/s+f*Ap*Ap-sp.Symbol('V'))/2
    delta_pr_D = D*(omega*omega*A*A/s-sp.Symbol('V'))-2*D*pr_background
    assert sp.expand(delta_pr_D+f*Ap*Ap*D) == 0
    return {**{name: True for name in pairs}, 'both_scalar_anisotropy_terms': True,
            'independent_source_conservation_components': 3, 'null_source_trace': True,
            'scalar_pressure_anisotropy': True}


def forced_reduction():
    r, s, f, n, A, Ap, Vs, Vss, omega, nu, L, k = sp.symbols(
        'r s f n A Ap Vs Vss omega nu Lambda kappa', nonzero=True)
    H, PH, K, T, U, PU, W, PW = state = sp.symbols('H PH K T U PU W PW')
    rho, pressure, jt, j0, j1, D = source = sp.symbols('rho pressure jt j0 j1 D')
    O = omega*omega/s
    V = O*A*A+f*Ap*Ap-2*(2*f*r*n-1+f)/(k*r*r)
    fp = 2*f*n-k*r*(O*A*A+f*Ap*Ap)
    d = 2/r+n+fp/(2*f)
    F = k*(2*O*A*A-V)/2
    np_, App = F/f-d*n, -d*Ap+(Vs-O)*A/f
    B, C = 1/r-n, L-2*k*r*r*O*A*A
    z = 2*nu*nu*r*r/s
    matrix = sp.Matrix([[L-2-z, 2*sp.I*nu*r*r*f*B/s], [2*sp.I*nu*r*r*B, C-z]])
    Rk = (2*r*r*f*n*PH+(4*r*r*f*n*n+L-2*f-k*r*r*(O*A*A-f*Ap*Ap))*H
          -4*k*r*r*f*(n+1/r)*Ap*U-2*k*r*r*(O-Vs)*A*U-2*k*r*r*f*Ap*PU
          -2*k*sp.I*nu*r*r*omega*A*W/s-4*k*r*r*f*(n+1/r)*j1-2*k*r*r*pressure)
    Rt = (-2*k*omega*r*r*(A*PW-Ap*W)-2*sp.I*nu*r*r*(PH+(2*n-1/r)*H-k*Ap*U)
          +4*k*sp.I*nu*r*r*j1+2*k*r*r*jt)
    constraints = matrix*sp.Matrix([K, T])-sp.Matrix([Rk, Rt])
    cross = 4*n*Ap+2*A*(2*O-Vs)/f
    qH = 4*n*n+L/(r*r*f)+k*(V-4*O*A*A)/f
    qU = (L/r**2+Vs+2*A*A*Vss-O)/f+2*k*Ap*Ap
    FH = ((4*n*(n+1/r)-2*F/f-2*nu*nu/(f*s))*D
          -4*k*n*j1-4*k*sp.I*nu*j0/(f*s)-2*k*rho/f)
    FU = (A*(Vs-O)/f-2*(n+1/r)*Ap)*D+2*k*Ap*j1
    FW = -sp.I*nu*omega*A*D/(f*s)+2*k*omega*A*j0/(f*s)
    evolution = [PH-2*n*D,
                 -d*PH+(qH-nu*nu/(f*s))*H-k*cross*U-4*sp.I*nu*B*T/s+FH,
                 PH+2*n*H-2*k*Ap*U+sp.I*nu*T/s+2*D/r-2*k*j1,
                 -(d-2/r)*T-sp.I*nu*(H+K+2*D)/f-2*k*omega*A*W/f+2*k*j0/f,
                 PU+Ap*D,
                 -d*PU+(qU-nu*nu/(f*s))*U-cross*H/2-2*sp.I*nu*omega*W/(f*s)+FU,
                 PW,
                 -d*PW+(L/r**2+Vs-O-nu*nu/s-2*k*O*A*A)*W/f
                 +2*sp.I*nu*omega*U/(f*s)+2*omega*(Ap+B*A)*T/s+FW]
    rates = dict(zip(state, evolution))
    rates.update({r: 1, s: 2*n*s, f: fp, n: np_, A: Ap, Ap: App, Vs: 2*Vss*A*Ap,
                  pressure: -n*(rho+pressure)-(3*pressure-rho)/r+L*j1/r**2-sp.I*nu*jt/s,
                  jt: -d*jt-sp.I*nu*rho/f+L*j0/(f*r*r),
                  j1: -d*j1-(rho-pressure)/(2*f)-(L-2)*D/(2*k*f*r*r)-sp.I*nu*j0/(f*s)})
    prime = constraints.applyfunc(lambda expr: sp.expand(sum(sp.diff(expr, v)*dv for v, dv in rates.items())))
    propagation = sp.Matrix([[0, sp.I*nu/s], [-sp.I*nu/f, -(d-2/r)]])
    defect = (prime-propagation*constraints).jacobian([*state, *source])
    for row in range(2):
        for col in range(len(state)+len(source)):
            value = sp.cancel(defect[row, col])
            assert value == 0, (row, col, value)
    assert constraints.diff(D) == sp.zeros(2, 1)
    Dp = sp.Symbol('Dp')
    Hp, Up, Wp = evolution[0], evolution[4], evolution[6]
    Hpp, Upp, Wpp = evolution[1]-2*np_*D-2*n*Dp, evolution[5]+App*D+Ap*Dp, evolution[7]
    Kp, Tp = evolution[2], evolution[3]
    rr = (-f*Hp/r+f*(n+1/r)*Kp+(L/(2*r*r)-2*f*n/r-f/r**2)*H
          +(nu*nu/s-(L-2)/(2*r*r))*K-2*sp.I*nu*f*T/(r*s)
          -2*f*(1/r**2+2*n/r)*D
          -k*(f*Ap*Up+(O*A*A-f*Ap*Ap)*H/2+(O-Vs)*A*U+sp.I*nu*omega*A*W/s-f*Ap*Ap*D+pressure))
    tr = (C*T/(2*r*r)+sp.I*nu*(Kp+B*K-H/r+k*Ap*U)+k*omega*(A*Wp-Ap*W)
          -2*sp.I*nu*D/r-k*jt)
    tt = (-f*Hpp/2-f*(3*n/2+1/r+fp/(4*f))*Hp+f*n*Kp-2*F*H+L*H/(2*r*r)
          +nu*nu/s*(H/2+K)-sp.I*nu*f/s*(Tp+(2/r+fp/(2*f))*T)
          -f*n*Dp-2*F*D+nu*nu*D/s
          -k*((2*O-Vs)*A*U+2*sp.I*nu*omega*A*W/s+V*H/2+rho))
    kgU = (f*(Upp+d*Up)+(O+nu*nu/s-Vs-2*A*A*Vss-L/r**2)*U+2*sp.I*nu*omega*W/s
           +f*(Kp-Hp)*Ap+(2*O-Vs)*A*H-sp.I*nu*f*T*Ap/s-f*Ap*Dp-2*(Vs-O)*A*D)
    kgW = (f*(Wpp+d*Wp)+(O+nu*nu/s-Vs-L/r**2)*W-2*sp.I*nu*omega*U/s
           -sp.I*nu*omega*A*(H+K)/s-f*omega/s*(2*T*Ap+A*(Tp+(d-2*n)*T))
           -sp.I*nu*omega*A*D/s)
    for value in (rr+constraints[0]/(2*r*r), tr-constraints[1]/(2*r*r), tt, kgU, kgW):
        assert sp.expand(value) == 0
    for value in [constraints[0], *[evolution[j] for j in (0, 1, 4, 5)]]:
        for odd in (W, PW, T, jt, j0):
            assert sp.diff(value, odd).subs(nu, 0).expand() == 0
    assert sp.diff(FW, j0).subs(nu, 0) == 2*k*omega*A/(f*s)
    return dict(constraint_propagation_entries=28, reconstructed_forced_field_equations=5,
                shifted_constraints_independent_of_anisotropy=True,
                source_derivatives_removed=True, zero_frequency_even_block_decouples=True,
                zero_frequency_current_block_retained=True)


def kinetic_checks():
    t, wr, wp, Or, Op, psi, nu = sp.symbols('t wr wp Or Op psi nu', real=True)
    phase = 2*wp-5*wr
    frequency = 2*Op-5*Or
    plus, reverse = sp.cos(phase-frequency*t), sp.cos(phase+frequency*t)
    assert sp.expand(sp.diff(plus, t)+Or*sp.diff(plus, wr)+Op*sp.diff(plus, wp)) == 0
    assert sp.expand(sp.diff(reverse, t)-Or*sp.diff(reverse, wr)-Op*sp.diff(reverse, wp)) == 0
    even = (sp.cos(psi-nu*t)+sp.cos(psi+nu*t))/2
    odd = (sp.cos(psi-nu*t)-sp.cos(psi+nu*t))/2
    assert sp.trigsimp(even-sp.cos(psi)*sp.cos(nu*t)) == 0
    assert sp.trigsimp(odd-sp.sin(psi)*sp.sin(nu*t)) == 0
    assert sp.simplify(sp.re(sp.I*sp.exp(-sp.I*nu*t))-sp.sin(nu*t)) == 0
    assert sp.integrate(sp.cos(2*wp-5*wr-nu*t), (wp, 0, 2*sp.pi)) == 0
    theta = sp.Symbol('theta', real=True)
    y = (1-sp.cos(theta))/2
    assert sp.trigsimp(sp.diff(y, theta)**2-y*(1-y)) == 0
    delta = sp.Symbol('delta', positive=True)
    cap = 2*sp.asin(sp.sqrt(delta))
    assert sp.simplify(sp.diff(cap, delta)-1/sp.sqrt(delta*(1-delta))) == 0
    assert sp.limit(cap, delta, 0, dir='+') == 0
    # For a translated one-sided fold, the three absolute-difference pieces.
    left = 2*sp.sqrt(delta)
    middle = 2*sp.sqrt(1-delta)-2+2*sp.sqrt(delta)
    right = 2-2*sp.sqrt(1-delta)
    assert sp.expand(left+middle+right-4*sp.sqrt(delta)) == 0
    return dict(two_time_reversed_transport_equations=True, standing_source_peak_normalization=True,
                time_current_Fourier_phase=True, modulation_total_energy_zero=True,
                turning_point_regularization=True, integrable_endpoint_caps=True,
                translated_fold_L1_distance='4*sqrt(delta) for 0<delta<1')


def inherited_power(digits):
    iv.dps = digits
    hashes = check_inputs()
    names = ['frequency-regular-polar-50', 'coupled-outgoing-space-50',
             'azimuth-orbit-50', 'resonant-tidal-overlap-50', 'low-frequency-quadrupole-50']
    paths = [ROOT/'data'/(name+'.json') for name in names]
    core, outgoing, orbit, overlap, threshold = [json.loads(path.read_text()) for path in paths]
    assert core['status'] == 'frequency_regular_six_dimensional_core_verified'
    assert outgoing['status'] == 'coupled_outgoing_space_continuation_checked_qualitatively'
    assert orbit['status'] == 'azimuth_orbit_ball_verified'
    assert overlap['status'] == 'positive_static_tidal_overlap_verified'
    assert core['core']['input_sha256'] == outgoing['input_sha256'] == orbit['input_sha256'] == hashes
    assert threshold['conditional_prediction']['inherited_background_sha256'] == hashes
    assert all(overlap['input_sha256'][key] == value for key, value in hashes.items())
    for name, value in overlap['input_sha256'].items():
        candidates = [directory/name for directory in
                      (ROOT/'data', OLD/'data/s08a06', OLD/'data/solitonic-screen')]
        paths_found = [path for path in candidates if path.exists()]
        assert len(paths_found) == 1, name
        path = paths_found[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value, name
    for report in (core['core'], outgoing, orbit, overlap):
        assert all(hashlib.sha256((REPO/key).read_bytes()).hexdigest() == value
                   for key, value in report['dependency_sha256'].items())
    extent = enclosure(*orbit['branch']['radial_extent'])
    assert extent.a > 15 and extent.b < 17
    assert orbit['branch']['first_cell_acceleration_lower'] > 0
    assert orbit['branch']['last_cell_acceleration_upper'] < 0
    Bd = enclosure(*overlap['static_multipole_change_per_energy_eta_at_pi_over_3'])
    assert Bd.b < 0
    amplitude, power = 2*Bd/3, Bd*Bd/15
    assert power.a > iv.mpf('1.402e-11') and power.b < iv.mpf('5.775e-11')
    assert [lower(power), upper(power)] == threshold['conditional_prediction']['conditional_mean_power_limit']
    paths += [ROOT/(name+'.py') for name in ('check_frequency_regular_polar', 'check_coupled_outgoing_space',
                                             'check_resonant_tidal_overlap', 'check_low_frequency_quadrupole')]
    return dict(kinetic_amplitude_limit=[lower(amplitude), upper(amplitude)],
                kinetic_mean_power_limit=[lower(power), upper(power)],
                amplitude_normalization='Aout/(E_rad*eta*nu^2)',
                power_normalization='mean_power/((E_rad*eta)^2*nu^6)',
                fixed_support_annulus=[15, 17], frequency_cutoff=None,
                current_Fourier_limit_may_be_nonzero=True,
                scope='Positive small-frequency periodic power for the prescribed conserved kinetic source; no gravitational-wave realization or nonlinear efficiency theorem.',
                input_sha256=hashes, dependency_sha256={str(path.relative_to(REPO)):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                new_radial_solves=0, quadrature_points=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/forced-polar-continuation.json')
    args = parser.parse_args()
    start = perf_counter()
    checks = dict(curvature_and_conservation=anisotropic_curvature(),
                  forced_reduction=forced_reduction(), kinetic_source=kinetic_checks())
    if args.checks_only:
        print(json.dumps(checks, indent=2))
    else:
        result = dict(status='positive_periodic_kinetic_quadrupole_power_limit_verified', digits=args.digits,
                      checks=checks, kinetic_response=inherited_power(args.digits))
        result['elapsed_seconds'] = perf_counter()-start
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
        print(json.dumps(result, indent=2))
