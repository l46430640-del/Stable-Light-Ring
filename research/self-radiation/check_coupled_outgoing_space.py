"""Check the exterior chart and estimates for the coupled outgoing space.

The continuation theorem is qualitative. This does not compute radiation
from a physical source or establish a nonzero-frequency numerical margin.
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
from check_asymptotic_exterior import certify as certify_exterior, lower
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs

if not __debug__:
    raise SystemExit('Run checks without python -O.')


def vacuum_chart():
    r, M, nu, omega = sp.symbols('r M nu omega', positive=True)
    Z, P = sp.symbols('Z P')
    f = 1-2*M/r
    n = M/(r*r*f)
    B, d = 1/r-n, 2/r+sp.diff(f, r)/f
    Vz = f*(24*r**3+24*M*r*r+36*M*M*r+18*M**3)/(r**3*(2*r+3*M)**2)
    aK = 6*(r*r+M*r+M*M)/(r*r*(2*r+3*M))
    b = sp.factor((r*aK-2-3*M/r)/f)
    a = sp.factor(r*sp.diff(aK, r)+r*(Vz-nu*nu)/f
                  +(2+3*M/r)/f*(3/r-aK))
    prime = lambda value: sp.diff(value, r)+sp.diff(value, Z)*P/f+sp.diff(value, P)*(Vz-nu*nu)*Z/f
    K, H = aK*Z+P, a*Z+b*P
    Kp, Hp = sp.factor(prime(K)), sp.factor(prime(H))
    T = sp.factor(-sp.I*nu*r*r/3*(Kp+B*K-H/r))
    assert sp.factor(r/3*(K+f/(2+3*M/r)*(H-r*Kp))-Z) == 0
    assert sp.factor(Hp-Kp+2*n*H+sp.I*nu*T/f) == 0
    defect = sp.expand(sp.factor(prime(T)+(d-2/r)*T+sp.I*nu*(H+K)/f))
    assert defect == 0, defect
    Hpp = prime(Hp)
    assert sp.factor(Hpp+d*Hp-(4*n*n+6/(r*r*f)-nu*nu/(f*f))*H+4*sp.I*nu*B*T/f) == 0
    rr = (-f*Hp/r+f*(n+1/r)*Kp+(3/r**2-2*f*n/r-f/r**2)*H
          +(nu*nu/f-2/r**2)*K-2*sp.I*nu*T/r)
    assert sp.factor(rr) == 0
    H1 = sp.factor(-r*r/3*(Kp+B*K-H/r))
    published_H = ((2*r*(r-2*M)-nu*nu*r**4+M*(r-3*M))/((r-2*M)*(2*r+3*M))*K
                   +(3*M-nu*nu*r**3)/(r*(2*r+3*M))*H1)
    assert sp.factor(published_H-H) == 0
    z = 2*nu*nu*r*r/f
    determinant = (4-z)*(6-z)+2*z*r*r*f*B*B
    chart_determinant = sp.factor(sp.diff(H, Z)*sp.diff(Hp, P)-sp.diff(H, P)*sp.diff(Hp, Z))
    assert sp.factor(chart_determinant-determinant/(4*r*r*f)) == 0
    assert sp.factor(a-a.subs(nu, 0)+r*nu*nu/f) == 0

    U, Up, W, Wp = sp.symbols('U Up W Wp')
    common = (6/r**2+1-(omega*omega+nu*nu)/f)/f
    Upp = -d*Up+common*U-2*sp.I*nu*omega*W/(f*f)
    Wpp = -d*Wp+common*W+2*sp.I*nu*omega*U/(f*f)
    flow = {r: 1, U: Up, Up: Upp, W: Wp, Wp: Wpp}
    radial = lambda value: sum(sp.diff(value, v)*dv for v, dv in flow.items())
    Vscalar = f*(1+6/r**2+2*M/r**3)
    for sign in (1, -1):
        q = r*(U+sign*sp.I*W)
        assert sp.expand(f*radial(f*radial(q))+((nu+sign*omega)**2-Vscalar)*q).cancel() == 0
    ratio = sp.Symbol('ratio', nonnegative=True)
    tail_defect = sp.factor(6-r*r*Vz).subs(M, ratio*r)
    expected_defect = (96*ratio+66*ratio**2+54*ratio**3+36*ratio**4)/(2+3*ratio)**2
    assert sp.factor(tail_defect-expected_defect) == 0
    return dict(Moncrief_normalization=True, vacuum_radial_constraint=True,
                vacuum_angular_constraints=2, vacuum_gravitational_evolution=True,
                published_metric_reconstruction=True,
                chart_determinant='Delta_vac/(4*r^2*f_M)', both_massive_channels=True,
                centrifugal_potential_upper=True)


def green_checks():
    x, y, k, beta = sp.symbols('x y k beta', positive=True)
    kernel = (y**4/x**3-x**4/y**3)/7
    assert sp.expand(sp.diff(kernel, y, 2)-12*kernel/y**2) == 0
    assert sp.expand(sp.diff(kernel, x, 2)-12*kernel/x**2) == 0
    assert kernel.subs(y, x) == 0 and sp.diff(kernel, y).subs(y, x) == 1
    assert sp.factor(-sp.diff(kernel, x)-(3*y**4/x**4+4*x**3/y**3)/7) == 0
    # The two branches of the massive Dirichlet Green function have jump +1.
    left = -(sp.exp(-k*(y-x))-sp.exp(-k*(x+y)))/(2*k)
    right = -(sp.exp(-k*(x-y))-sp.exp(-k*(x+y)))/(2*k)
    assert left.subs(x, 0) == 0
    assert sp.simplify((sp.diff(right, x)-sp.diff(left, x)).subs(x, y)) == 1
    particular = (sp.exp(-beta*x)-sp.exp(-k*x))/(beta*beta-k*k)
    assert sp.simplify(sp.diff(particular, x, 2)-k*k*particular-sp.exp(-beta*x)) == 0
    assert particular.subs(x, 0) == 0
    delta = sp.Symbol('delta', positive=True)
    terminal_integrand = (sp.exp(delta*y)-sp.exp(-(2*beta+delta)*y))/(2*(beta+delta))
    assert sp.limit(terminal_integrand, y, sp.oo) == sp.oo
    # An exact tail integral bounds the gravitational C1 operator.
    polynomial = sum(sp.factorial(4)/sp.factorial(4-j)*(1+x)**(4-j)/beta**(j+1) for j in range(5))
    integral_bound = sp.exp(-beta*x)*polynomial
    assert sp.expand(sp.diff(integral_bound, x)+(1+x)**4*sp.exp(-beta*x)) == 0
    return dict(polynomial_gravitational_comparison=True, comparison_derivative=True,
                massive_Dirichlet_jump=True, slower_forced_scalar_tail=True,
                exact_exponential_tail_integral=True,
                terminal_massive_integral_requires_source_decay_above_mass=True)


def exterior_bounds(digits):
    iv.dps = digits
    certificate = certify_exterior(digits)
    assert certificate['verified_decimal_bounds']['decay_rate_lower'] == '0.98'
    assert certificate['verified_decimal_bounds']['logarithmic_slope_upper'] == '1.21'
    R, fmin, amplitude = iv.mpf(40), iv.mpf('.7169'), iv.mpf('1e-8')
    Mmax, omega_max = iv.mpf('5.66'), iv.mpf('.141')
    Omax = omega_max**2*iv.exp(iv.mpf('2e-11'))/fmin
    radial_pressure_weight = (Omax+iv.mpf('1.21')**2)*R*R*amplitude*amplitude/2
    rn_upper = (Mmax/R+4*iv.pi*radial_pressure_weight)/fmin
    assert rn_upper.b < iv.mpf('.2')
    centrifugal_coefficient = 2*fmin*(1-rn_upper)**2
    assert centrifugal_coefficient.a > iv.mpf('.9')
    Xi = 16*iv.pi*R*R*Omax*amplitude*amplitude
    assert Xi.b < iv.mpf('2.2304e-13')
    minimum = 24-4*Xi-iv.mpf('9.1')**2/4
    weighted_minimum = iv.mpf('23.9')-4*Xi-iv.mpf('9.1')**2/(4*iv.mpf('.9'))
    assert minimum.a > iv.mpf('3.29') and weighted_minimum.a > 0

    R0 = iv.mpf('1e6')
    tortoise_ratio = 1+2*Mmax*iv.ln(R0)/R0
    assert tortoise_ratio.b < iv.mpf('1.001')
    assert (6*tortoise_ratio**2).b < 12
    scalar_lower = 1-2*Mmax/R0-(omega_max+iv.mpf('.01'))**2
    scalar_upper = 1+6/R0**2+2*Mmax/R0**3
    k, K, gamma = iv.mpf('.9'), iv.mpf('1.1'), iv.mpf('.5')
    assert scalar_lower.a > (k*k).b and scalar_upper.b < (K*K).a
    assert (K/(2*k)).b < 1
    green_gain = 2*k/(k*k-gamma*gamma)
    assert green_gain.b < 4
    # Polynomial factors can be absorbed while retaining the stated exponent.
    assert (iv.mpf('.98')/tortoise_ratio).a > iv.mpf('.9')
    return dict(exterior_start=40, rn_upper=upper(rn_upper),
                two_r_squared_f_B_squared_lower=lower(centrifugal_coefficient), Xi_upper=upper(Xi),
                determinant_lower=lower(minimum), determinant_display='Delta > 3.29',
                weighted_determinant_display='Delta > 0.1*(1+z^2)',
                weighted_determinant_remainder_lower=lower(weighted_minimum),
                analytic_comparison_radius='1e6', tortoise_radius_ratio_upper=upper(tortoise_ratio),
                massive_potential_lower=lower(scalar_lower), massive_potential_upper=upper(scalar_upper),
                massive_Green_C1_gain_upper=upper(green_gain), massive_Green_C1_gain_display=4,
                scalar_norm_decay='.5', coupling_decay='.75', frequency_absolute_upper='.01',
                coupled_contraction_radius=None, global_matching_frequency_cutoff=None,
                qualitative_scope='A sufficiently large tail radius and a sufficiently small positive frequency interval exist; neither is numerically optimized or certified here.',
                radial_solves=0, quadrature_points=0)


def dependencies():
    hashes = check_inputs()
    core_path = ROOT/'data/frequency-regular-polar-50.json'
    phase_path = ROOT/'data/stationary-phase-50.json'
    threshold_path = ROOT/'data/low-frequency-quadrupole-50.json'
    core = json.loads(core_path.read_text())
    phase = json.loads(phase_path.read_text())
    threshold = json.loads(threshold_path.read_text())
    assert core['status'] == 'frequency_regular_six_dimensional_core_verified'
    assert phase['status'] == 'stationary_phase_form_positive_for_all_l_at_least_2'
    assert threshold['status'] == 'exterior_threshold_law_checked_stellar_continuation_open'
    assert core['core']['input_sha256'] == phase['input_sha256'] == hashes
    assert threshold['conditional_prediction']['inherited_background_sha256'] == hashes
    existing = core['core']['dependency_sha256']
    assert all(hashlib.sha256((REPO/key).read_bytes()).hexdigest() == value for key, value in existing.items())
    paths = [core_path, phase_path, threshold_path, ROOT/'check_frequency_regular_polar.py',
             ROOT/'check_low_frequency_quadrupole.py']
    return dict(input_sha256=hashes, dependency_sha256={**existing,
                **{str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/coupled-outgoing-space.json')
    args = parser.parse_args()
    start = perf_counter()
    checks = dict(vacuum_chart=vacuum_chart(), Green_kernels=green_checks())
    if args.checks_only:
        print(json.dumps(checks, indent=2))
    else:
        result = dict(status='coupled_outgoing_space_continuation_checked_qualitatively', digits=args.digits,
                      scope='Homogeneous quadrupole outgoing space and qualitative matching. Physical conserved forcing and same-source radiation are not established.',
                      checks=checks, bounds=exterior_bounds(args.digits), **dependencies())
        result['elapsed_seconds'] = perf_counter()-start
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
        print(json.dumps(result, indent=2))
