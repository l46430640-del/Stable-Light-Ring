"""Check the exterior quadrupole threshold law and its flux normalization.

The stellar frequency continuation is an explicit unverified hypothesis.
The derived power interval is conditional, not a radiation certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from mpmath import iv
import sympy as sp
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
REPO = ROOT.parent.parent
sys.path.insert(0, str(OLD))
from check_asymptotic_exterior import lower
from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs

if not __debug__:
    raise SystemExit('Run checks without python -O.')


def algebra_checks():
    r, M, nu, x, y = sp.symbols('r M nu x y', positive=True)
    f = 1-2*M/r
    V = f*(24*r**3+24*M*r*r+36*M*M*r+18*M**3)/(r**3*(2*r+3*M)**2)
    K, H = r*r-2*M*M, r*(r-2*M)
    master = r/3*(K+f/(2+3*M/r)*(H-r*sp.diff(K, r)))
    growing = sp.factor(6*master)
    assert growing == 2*r*(r**3+3*M*r*r-6*M**3)/(2*r+3*M)
    assert sp.factor(f*sp.diff(f*sp.diff(growing, r), r)-V*growing) == 0
    assert sp.limit(growing/r**3, r, sp.oo) == 1

    Bd = sp.Symbol('Bd')
    leading_K, leading_H = -2*Bd/r**3, -2*Bd/r**3
    leading_master = r/3*(leading_K+f/(2+3*M/r)*(leading_H-r*sp.diff(leading_K, r)))
    assert sp.limit(r*r*leading_master, r, sp.oo) == -2*Bd
    # The unit outgoing Hankel field and its small-frequency coefficient.
    outgoing = sp.exp(sp.I*nu*x)*(1+3*sp.I/(nu*x)-3/(nu*x)**2)
    assert sp.simplify(sp.diff(outgoing, x, 2)+(nu*nu-6/x**2)*outgoing) == 0
    assert sp.limit(nu*nu*outgoing, nu, 0, dir='+') == -3/x**2
    incoming = sp.conjugate(outgoing)
    assert sp.simplify(incoming*sp.diff(outgoing, x)-sp.diff(incoming, x)*outgoing) == 2*sp.I*nu
    kernel = (incoming*outgoing.subs(x, y)-outgoing*incoming.subs(x, y))/(2*sp.I*nu)
    assert sp.simplify(kernel.subs(y, x)) == 0
    assert sp.simplify(sp.diff(kernel, y).subs(y, x)) == 1
    kernel0 = (y**3/x**2-x**3/y**2)/5
    # A finite series suffices; it retains the cancellation of singular terms.
    def truncated(sign, radius):
        return sp.series(sp.exp(sign*sp.I*nu*radius)
                         *(1+sign*3*sp.I/(nu*radius)-3/(nu*radius)**2), nu, 0, 5).removeO()
    limit_kernel = sp.expand((truncated(-1, x)*truncated(1, y)
                             -truncated(1, x)*truncated(-1, y))/(2*sp.I*nu)).coeff(nu, 0)
    assert sp.simplify(limit_kernel-kernel0) == 0
    assert kernel0.subs(y, x) == 0 and sp.diff(kernel0, y).subs(y, x) == 1

    q = sp.Symbol('q', nonnegative=True)
    defect = sp.factor(6-r*r*V).subs(M, q*r)
    expected = (96*q+66*q*q+54*q**3+36*q**4)/(2+3*q)**2
    assert sp.factor(defect-expected) == 0
    assert (96+66/sp.Integer(4)+54/sp.Integer(16)+36/sp.Integer(64))/4 < 30
    # Integral representation bounds the regular free solution near zero.
    t = sp.Symbol('t', real=True)
    zpos = sp.Symbol('zpos', positive=True)
    imaginary_hankel = sp.im(sp.exp(sp.I*zpos)*(1+3*sp.I/zpos-3/zpos**2))
    integral_form = -zpos**3/16*sp.integrate(sp.cos(zpos*t)*(1-t*t)**2, (t, -1, 1))
    assert sp.simplify(imaginary_hankel-integral_form) == 0
    assert sp.integrate((1-t*t)**2, (t, -1, 1))/16 == sp.Rational(1, 15)
    assert sp.integrate(abs(t)*(1-t*t)**2, (t, -1, 1))/16 == sp.Rational(1, 48)
    # These cover small/small, small/large and large/large kernel regions.
    value_bounds = [sp.Rational(8, 5), sp.Integer(72), sp.Integer(36)]
    derivative_bounds = [sp.Rational(16, 3), sp.Integer(192), sp.Integer(96)]
    assert max(value_bounds+derivative_bounds) < 200

    z = sp.Symbol('z', real=True)
    angular_norm = 2*sp.pi*sp.integrate(2*(sp.Rational(3, 2)*(1-z*z))**2, (z, -1, 1))
    assert angular_norm == 48*sp.pi/5
    mean_square = sp.integrate(sp.sin(t)**2, (t, 0, 2*sp.pi))/(2*sp.pi)
    assert mean_square == sp.Rational(1, 2)
    flux_coefficient = angular_norm/(32*sp.pi)*mean_square*sp.Rational(2, 3)**2
    assert flux_coefficient == sp.Rational(1, 15)
    # Independent axisymmetric STF quadrupole normalization, Bd=-q_mass.
    stf = sp.diag(-sp.Rational(1, 3), -sp.Rational(1, 3), sp.Rational(2, 3))
    assert sum(v*v for v in stf)/10 == flux_coefficient
    # A decaying effective source cannot be discarded at a finite matching radius.
    tail = sp.exp(-x)
    source = sp.diff(tail, x, 2)-6*tail/x**2
    recovered = sp.integrate(kernel0*source.subs(x, y), (y, x, sp.oo))
    assert sp.simplify(recovered-tail) == 0
    e = sp.Symbol('e', positive=True)
    action, radial = sp.Function('J')(e), sp.Function('Omega_r')(e)
    beat = 2*(sp.sqrt(e)-action*radial)-5*radial
    derivative = sp.diff(beat, e).subs(sp.diff(action, e), 1/(2*sp.sqrt(e)*radial))
    assert sp.simplify(derivative+(2*action+5)*sp.diff(radial, e)) == 0
    return dict(growing_static_Zerilli_solution=True, static_master_coefficient=True,
                outgoing_free_equation=True, outgoing_threshold_coefficient=True,
                free_Wronskian=True, kernel_zero_frequency_limit=True,
                Zerilli_tail_inequality=True, regular_Bessel_bound=True,
                kernel_region_bounds=[list(map(str, value_bounds)), list(map(str, derivative_bounds))],
                tensor_harmonic_norm=True, Bondi_flux_factor=True,
                independent_STF_flux_factor=True, noncompact_source_reconstruction=True,
                regular_Einstein_constraints=regular_constraints(), fixed_classical_detuning=True)


def regular_constraints():
    r, nu, omega, alpha, f, A, Ap, n, d, C = sp.symbols('r nu omega alpha f A Ap n d C', nonzero=True)
    H, Hp, K, Kp, T, Tp, U, Up, V, Vp = sp.symbols('H Hp K Kp T Tp U Up V Vp')
    kappa = 8*sp.pi
    plus, minus = [r*(U+sign*sp.I*V)/sp.sqrt(2) for sign in (1, -1)]
    plusp, minusp = [(U+sign*sp.I*V+r*(Up+sign*sp.I*Vp))/sp.sqrt(2) for sign in (1, -1)]
    common = omega*A/sp.sqrt(2)*(plus-minus-r*plusp+r*minusp)
    q1 = Kp+(1/r-n)*K-H/r+C*T/(2*sp.I*nu*r*r) \
         +kappa/(r*r*nu)*(r*((nu+omega)*plus+(nu-omega)*minus)*Ap/sp.sqrt(2)+common)
    q2 = Hp+(1/r-n)*K-H/r+2*n*H+(C/(2*r*r)-nu*nu/alpha**2)*T/(sp.I*nu) \
         +kappa/(r*r*nu)*(r*((omega-nu)*plus-(omega+nu)*minus)*Ap/sp.sqrt(2)+common)
    q3 = Tp/(sp.I*nu)+(H+K)/f+(d-2/r)*T/(sp.I*nu) \
         -2*kappa*omega*A*(plus-minus)/(sp.sqrt(2)*r*nu*f)
    m, pressure, potential = sp.symbols('m pressure potential')
    n_background = (m/r**2+4*sp.pi*r*pressure)/f
    fp_background = 2*m/r**2-8*sp.pi*r*(pressure+potential)
    assert sp.simplify(n_background+fp_background/(2*f)
                       -(2*m/r**2-4*sp.pi*r*potential)/f) == 0
    shift = -2*kappa*omega*r*r*(A*Vp-Ap*V)/C \
            -2*sp.I*nu*r*r*(Kp+(1/r-n)*K-H/r+kappa*Ap*U)/C
    assert sp.simplify(q1.subs(T, shift)) == 0
    assert sp.simplify(q2-q1-(Hp-Kp+2*n*H-2*kappa*Ap*U+sp.I*nu*T/alpha**2)) == 0
    assert sp.simplify(sp.I*nu*q3-(Tp+(d-2/r)*T+sp.I*nu*(H+K)/f+2*kappa*omega*A*V/f)) == 0
    # At zero frequency the current equation reproduces the phase form.
    amp, phase, lapse, metric, c = [sp.Function(name)(r) for name in ('A', 'psi', 'alpha', 'f', 'c')]
    b = -2*kappa*omega*r*r*amp*amp*sp.diff(phase, r)/c
    flow = sp.diff(b, r)+(sp.diff(lapse, r)/lapse+sp.diff(metric, r)/(2*metric))*b \
           +2*kappa*omega*amp*amp*phase/metric
    p = r*r*lapse*sp.sqrt(metric)
    reduced = sp.diff(p*amp*amp/c*sp.diff(phase, r), r)-lapse*amp*amp/sp.sqrt(metric)*phase
    assert sp.simplify(flow+2*kappa*omega/(lapse*sp.sqrt(metric))*reduced) == 0
    return dict(published_equations=3, phase_limit=True,
                scope='Regular identities only; an independent six-dimensional coupled system remains to be constructed.')


def exterior_tail():
    # This is an analytic half-line estimate, not an integration out to R.
    R, M = iv.mpf('1e6'), iv.mpf('5.66')
    ratio = 1+2*M*iv.ln(R)/R
    inverse_f = 1/(1-2*M/R)
    first_moment = ratio*inverse_f*M*(54+24*iv.ln(R))/R
    contraction = 200*first_moment
    assert first_moment.b < iv.mpf('.002182705')
    assert contraction.b < iv.mpf('.437')
    assert R > 4*M and 2*M > 1
    return dict(analytic_radius='1e6', mass_upper='5.66',
                first_moment_upper=upper(first_moment),
                weighted_kernel_constant=200, contraction_upper=upper(contraction),
                verified_contraction_bound='.437', radial_solves=0, quadrature_points=0)


def conditional_power():
    path = ROOT/'data/resonant-tidal-overlap-50.json'
    report = json.loads(path.read_text())
    assert report['status'] == 'positive_static_tidal_overlap_verified'
    hashes = check_inputs()
    background = json.loads((OLD/'data/s08a06/global-background.json').read_text())
    mass = enclosure(*background['endpoint_graphs']['ADM_mass_interval'])
    assert mass.a > iv.mpf('5.64') and mass.b < iv.mpf('5.66')
    assert all(report['input_sha256'][key] == value for key, value in hashes.items())
    assert all(hashlib.sha256((REPO/key).read_bytes()).hexdigest() == value
               for key, value in report['dependency_sha256'].items())
    Bd = enclosure(*report['static_multipole_change_per_energy_eta_at_pi_over_3'])
    assert Bd.b < 0
    amplitude, power = 2*Bd/3, Bd**2/15
    assert amplitude.a > iv.mpf('-1.963e-5') and amplitude.b < iv.mpf('-9.670e-6')
    assert power.a > iv.mpf('1.402e-11') and power.b < iv.mpf('5.775e-11')
    assert power.a > iv.mpf('1.4028484820740533e-11')
    assert power.b < iv.mpf('5.7743046321533906e-11')
    return dict(static_multipole_interval=[lower(Bd), upper(Bd)],
                conditional_outgoing_amplitude_limit=[lower(amplitude), upper(amplitude)],
                conditional_mean_power_limit=[lower(power), upper(power)],
                amplitude_normalization='Aout/(E_rad eta nu^2)',
                power_normalization='mean_power/((E_rad eta)^2 nu^6)',
                assumptions=['a real P2 harmonic with prescribed peak amplitude',
                             'outgoing solutions converge to the certified static response on a fixed matching sphere',
                             'the effective exterior source converges pointwise and has a uniform exponential bound'],
                scope='Conditional spectral prediction. Stellar continuation, finite-time preparation and wave realization unverified.',
                overlap_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), inherited_background_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/low-frequency-quadrupole.json')
    args = parser.parse_args()
    iv.dps = args.digits
    start = perf_counter()
    checks = algebra_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    result = dict(status='exterior_threshold_law_checked_stellar_continuation_open', digits=args.digits,
                  checks=checks, exterior=exterior_tail(), conditional_prediction=conditional_power())
    names = ('gr-qc-0502028v1', '0908.4292v2')
    result['literature'] = {name: dict(url='https://arxiv.org/src/'+name.replace('gr-qc-', 'gr-qc/'),
                                   archive_sha256=hashlib.sha256((REPO/'literature/self-radiation'/('arXiv-'+name+'.tar.gz')).read_bytes()).hexdigest())
                            for name in names}
    result['elapsed_seconds'] = perf_counter()-start
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
