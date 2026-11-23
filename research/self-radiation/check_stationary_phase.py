"""Certify the stationary scalar-phase and polar-current sector.

The positive reduced form excludes a homogeneous zero-frequency mode.
It does not establish a finite-frequency outgoing resolvent or radiation.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from mpmath import iv
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
sys.path.insert(0, str(OLD))
from check_asymptotic_exterior import certify as certify_exterior, lower
from check_barrier_certificate import enclosure, exact
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs, physical_cell
from check_regular_center import Dual, integrands, leading
from trapped_axial import Background

if not __debug__:
    raise SystemExit('Run certificates without python -O.')


def twist_curvature_check():
    """Check the timelike reduction sign directly from a four-metric."""
    x = sp.Symbol('x', real=True)
    b = sp.Function('b')(x)
    alpha = sp.Function('alpha')(x)
    g = sp.Matrix([[-alpha**2, 0, alpha**2*b, 0], [0, 1, 0, 0],
                   [alpha**2*b, 0, 1-alpha**2*b*b, 0], [0, 0, 0, 1]])
    inv = g.inv()
    derivative = lambda value, i: sp.diff(value, x) if i == 1 else 0
    connection = [[[sp.simplify(sum(inv[a, d]*(derivative(g[d, c], b_index)
                    +derivative(g[d, b_index], c)-derivative(g[b_index, c], d))
                    for d in range(4))/2) for c in range(4)]
                    for b_index in range(4)] for a in range(4)]
    curvature = 0
    for i in range(4):
        for j in range(4):
            if inv[i, j] == 0:
                continue
            ricci = sum(derivative(connection[a][i][j], a)
                        -derivative(connection[a][i][a], j)
                        +sum(connection[a][a][d]*connection[d][i][j]
                             -connection[a][j][d]*connection[d][i][a]
                             for d in range(4)) for a in range(4))
            curvature += inv[i, j]*ricci
    assert sp.simplify(curvature-alpha**2*sp.diff(b, x)**2/2+2*sp.diff(alpha, x, 2)/alpha) == 0
    assert g.det() == -alpha**2
    return True


def algebra_checks():
    r, alpha, f, A, omega, L = sp.symbols('r alpha f A omega L', positive=True)
    B, p, q = sp.symbols('B p q', real=True)
    Xi = 16*sp.pi*r*r*omega*omega*A*A/(alpha*alpha)
    P = r*r*alpha*sp.sqrt(f)
    density = L*alpha**3*sp.sqrt(f)*B*B/(32*sp.pi) \
              -P*A*A*(p-omega*B)**2/2-L*P*A*A*q*q/(2*r*r*f)
    eliminated = -16*sp.pi*omega*r*r*A*A*p/(alpha*alpha*(L-Xi))
    assert sp.factor(sp.diff(density, B).subs(B, eliminated)) == 0
    assert sp.factor(p-omega*eliminated-p/(1-Xi/L)) == 0
    effective = -P*A*A*(p*p/(1-Xi/L)+L*q*q/(r*r*f))/2
    assert sp.factor(density.subs(B, eliminated)-effective) == 0

    # Independent zero-frequency constraint from the published polar system.
    Af, phase = sp.Function('A')(r), sp.Function('psi')(r)
    difference = sp.sqrt(2)*sp.I*r*Af*phase
    scalar_bracket = r*omega*difference*sp.diff(Af, r)/sp.sqrt(2) \
        +omega*Af/sp.sqrt(2)*(difference-r*sp.diff(difference, r))
    published = alpha*alpha*B*(L-Xi)/(2*sp.I*r*r) \
        +8*sp.pi*scalar_bracket.subs({sp.diff(Af, r): sp.Symbol('Ap'),
                                     sp.diff(phase, r): p, Af: A})/(r*r)
    expected = -sp.I*(alpha*alpha*B*(L-Xi)/(2*r*r)+8*sp.pi*omega*A*A*p)
    assert sp.simplify(published-expected) == 0

    t = sp.Symbol('t')
    shift, gradient = sp.symbols('shift gradient')
    assert sp.expand((gradient+omega*t)-omega*(shift+t)-(gradient-omega*shift)) == 0
    # The free phase form is the ground-state transform of the imaginary field.
    u, amp, weight = [sp.Function(name)(r) for name in ('u', 'A', 'P')]
    q0 = sp.Symbol('Q0')
    expansion = sp.expand(weight*(sp.diff(u, r)-sp.diff(amp, r)/amp*u)**2
        +sp.diff(weight*sp.diff(amp, r)/amp*u*u, r)
        -weight*(sp.diff(u, r)**2+q0*u*u))
    replacement = q0*amp-sp.diff(weight, r)/weight*sp.diff(amp, r)
    assert sp.simplify(expansion.subs(sp.diff(amp, r, 2), replacement)) == 0
    br, be, phase, g, h, j = [sp.Function(name)(r) for name in ('br', 'be', 'phase', 'g', 'h', 'j')]
    unfixed = g*(sp.diff(be, r)-br)**2/2-h*(sp.diff(phase, r)-omega*br)**2/2 \
              -j*(phase-omega*be)**2/2
    euler = lambda field: sp.diff(unfixed, field)-sp.diff(sp.diff(unfixed, sp.diff(field, r)), r)
    assert sp.expand(-sp.diff(euler(br), r)+euler(be)+omega*euler(phase)) == 0
    radial = h/(1-sp.Function('Xi')(r)/L)
    reduced = -radial*sp.diff(phase, r)**2/2-j*phase*phase/2
    reduced_euler = sp.diff(reduced, phase)-sp.diff(sp.diff(reduced, sp.diff(phase, r)), r)
    assert sp.expand(reduced_euler-sp.diff(radial*sp.diff(phase, r), r)+j*phase) == 0
    return dict(timelike_twist_curvature=twist_curvature_check(),
                radial_shift_variation=True, reduced_phase_form=True,
                gauge_invariant_phase_gradient=True, published_polar_constraint=True,
                imaginary_field_ground_state_transform=True,
                omitted_angular_equation_gauge_identity=True, reduced_phase_equation=True)


def center_bound(bg):
    r0 = exact(float(bg.sol.x[1]))
    nominal = [iv.mpf(1), exact(float(bg.sol(0)[3]))]
    parameters = [v+enclosure(-iv.mpf('2.1e-8'), iv.mpf('2.1e-8')) for v in nominal]
    baseline = leading(nominal, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
    radius = iv.mpf('1e-6')
    box = [v+enclosure(-radius, radius) for v in baseline]
    s = enclosure(0, r0*r0)
    factors = [iv.mpf(1)/2, iv.mpf(1)/3, iv.mpf(1)/3, iv.mpf(1)/2]
    image = integrands(s, box, parameters, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
    image_radius = max(upper(abs(f*v-b)) for f, v, b in zip(factors, image, baseline))
    derivative = integrands(s, [Dual(box[j], np.eye(4, dtype=int)[j]) for j in range(4)],
                           parameters, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
    contraction = max(upper(f*sum(map(abs, v.d), iv.mpf(0))) for f, v in zip(factors, derivative))
    assert exact(image_radius) < radius and exact(contraction) < 1
    A = iv.mpf('.06')*(parameters[0]+s*box[0])
    zeta = parameters[1]+s*box[3]
    assert A.a > 0
    bound = upper(16*iv.pi*s*iv.exp(2*zeta)*A*A)
    return dict(radius=float(r0.mid), Xi_upper=bound,
                image_radius=image_radius, contraction=contraction)


def finite_bound(bg, max_seconds):
    start = perf_counter()
    knots = [Fraction(float(x)) for x in bg.sol.x]
    queue = list(zip(knots[1:-1], knots[2:]))[::-1]
    count, refinements, maximum, worst = 0, 0, 0., None
    while queue:
        if perf_counter()-start > max_seconds:
            raise RuntimeError('Stationary phase calculation stopped at its time budget.')
        left, right = queue.pop()
        piece = bisect_right(knots, (left+right)/2)-1
        r = enclosure(left, right)
        y = physical_cell(bg, piece, left, right)
        A = iv.mpf('.06')*y[0]
        bound = upper(16*iv.pi*r*r*iv.exp(2*y[3])*A*A)
        if exact(bound) >= 3:
            assert count+len(queue)+2 <= 3999, 'Subdivision budget exhausted.'
            middle = (left+right)/2
            queue.extend(((middle, right), (left, middle)))
            refinements += 1
            continue
        count += 1
        if bound > maximum:
            maximum, worst = bound, [str(left), str(right)]
    assert count+1 <= 4000
    return dict(Xi_upper=maximum, maximum_cell_exact=worst,
                cells=count, refinements=refinements, elapsed_seconds=perf_counter()-start)


def exterior_bound(digits):
    certificate = certify_exterior(digits)
    assert certificate['verified_decimal_bounds']['decay_rate_lower'] == '0.98'
    R, amplitude, omega = iv.mpf(40), iv.mpf('1e-8'), iv.mpf('.141')
    fmin = iv.mpf('.7169')
    # r^2 A^2 decreases on the half-line under the inherited exponential bound.
    assert 2/R-2*iv.mpf('.98') < 0
    bound = 16*iv.pi*R*R*amplitude*amplitude*omega*omega*iv.exp(iv.mpf('2e-11'))/fmin
    return dict(Xi_upper=upper(bound), radius=40,
                amplitude_upper='1e-8', decay_rate_lower='.98', metric_factor_lower='.7169',
                log_lapse_factor_absolute_upper='1e-11',
                exterior_contraction=certificate['verified_decimal_bounds']['contraction'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--max-seconds', type=int, default=60)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/stationary-phase.json')
    args = parser.parse_args()
    iv.dps = args.digits
    start = perf_counter()
    checks = algebra_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    hashes = check_inputs()
    bg = Background()
    center, finite, exterior = center_bound(bg), finite_bound(bg, args.max_seconds), exterior_bound(args.digits)
    maximum = max(part['Xi_upper'] for part in (center, finite, exterior))
    assert exact(maximum) < 3
    gap = 1-exact(maximum)/6
    assert gap.a > iv.mpf('.5') and (1/gap).b < 2
    for value, display in ((center['Xi_upper'], '2.1670e-7'),
                           (center['image_radius'], '5.006e-7'),
                           (center['contraction'], '.500001'),
                           (finite['Xi_upper'], '2.799781'),
                           (exterior['Xi_upper'], '2.2304e-13'),
                           (maximum, '2.799780613607389')):
        assert Fraction(value) < Fraction(display)
    assert gap.a > iv.mpf('.5333698') and (1/gap).b < iv.mpf('1.874872')
    checks['outward_rounded_display_bounds'] = True
    # A single ordinary point also shows why the l=1 test cannot be inferred.
    knots = [Fraction(float(x)) for x in bg.sol.x]
    y = physical_cell(bg, bisect_right(knots, Fraction(14))-1, 14, 14)
    at14 = 16*iv.pi*14**2*iv.exp(2*y[3])*(iv.mpf('.06')*y[0])**2
    assert at14.a > 2
    dependencies = [OLD/name for name in ('check_polar_coercivity.py', 'check_regular_center.py',
                    'check_asymptotic_exterior.py', 'check_background_defect.py',
                    'check_barrier_certificate.py', 'check_finite_static_inverse.py',
                    'finite_static_inverse.py', 'trapped_axial.py')]
    result = dict(status='stationary_phase_form_positive_for_all_l_at_least_2', digits=args.digits,
                  scope='Exact inherited star; excludes stationary phase/current kernels. No outgoing resolvent or radiation theorem.',
                  input_sha256=hashes, checks=checks, center=center, finite=finite, exterior=exterior,
                  global_Xi_upper=maximum, quadrupole_denominator_lower=lower(gap),
                  radial_energy_gain_upper=upper(1/gap), Xi_at_14=[lower(at14), upper(at14)],
                  verified_global_bound='Xi < 3', verified_form_comparison='E0 <= Ephase < 2 E0 for nonzero fields, l >= 2',
                  dependency_sha256={str(path.relative_to(ROOT.parent.parent)):
                                     hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies},
                  elapsed_seconds=perf_counter()-start)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
