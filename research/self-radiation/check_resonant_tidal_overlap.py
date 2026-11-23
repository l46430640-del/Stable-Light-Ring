"""Enclose the complete static tidal overlap on the certified azimuth orbit.

The weighted row h*b_m is a polynomial in chi=h-1. Keeping this polynomial
before evaluation preserves the small orbital factors in its derivatives.
All stellar coefficients, including those on the reference orbit, retain
the inherited exact-background uncertainty. No outgoing news is computed.
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
from check_asymptotic_exterior import lower
from check_background_defect import field_equations
from check_barrier_certificate import Jet, exact, enclosure
from check_finite_static_inverse import upper
from check_polar_transverse import hermite
from trapped_axial import Background
from check_azimuth_orbit import OrbitCertificate
from check_iterated_tidal_observation import IntegratedTide, iterated_rows, derivative, truncate

if not __debug__:
    raise SystemExit('Run interval certificates without python -O.')


def poly_add(*values):
    order = values[0][0].order
    return [sum((p[j] for p in values if j < len(p)), Jet(0, order=order))
            for j in range(max(map(len, values)))]


def poly_mul(a, b):
    order = a[0].order
    return [sum((a[i]*b[j-i] for i in range(len(a)) if 0 <= j-i < len(b)), Jet(0, order=order))
            for j in range(len(a)+len(b)-1)]


def poly_truncate(a, order):
    return [truncate(v, order) for v in a]


def poly_dchi(a):
    return [j*a[j] for j in range(1, len(a))] or [Jet(0, order=a[0].order)]


def poly_value(a, value):
    result = a[-1]
    for coefficient in reversed(a[:-1]):
        result = result*value+coefficient
    return result


def weighted_rows(radius, state, iterations=3, extra=2):
    """Polynomial rows c_m=h*b_m; each coefficient is a radial Taylor jet."""
    order = 2*iterations-1+extra
    r = Jet(0, [radius, iv.mpf(1)]+[iv.mpf(0)]*(order-1))
    y = [Jet(v, order=order) for v in state]
    for k in range(order):
        flow = field_equations(r, y, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
        for j in range(4):
            y[j].c[k+1] = flow[j].c[k]/(k+1)
    pi = Jet(iv.pi, order=order)
    A, Ap, f, O2 = y[0]*iv.mpf('.06'), y[1]*iv.mpf('.06'), 1-2*y[2], (2*y[3]).exp()
    ratio = A*A/iv.mpf('.08')**2
    V = A*A*(1-2*ratio)*(1-2*ratio)
    Vs, Vss = (1-2*ratio)*(1-6*ratio), (-8+24*ratio)/iv.mpf('.08')**2
    rho, pressure = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    n = (y[2]/r+4*pi*r*pressure)/f
    d = 2/r+n+(y[2]/r-4*pi*r*rho)/f
    L, K = 2*n-2/r, r*r*f
    Av = 6/(r*r*f)+4*n*n+8*pi*(V-4*O2*A*A)/f
    Au = 6/(r*r*f)+(Vs+2*A*A*Vss-O2)/f+16*pi*Ap*Ap
    C = 4*n*Ap+2*A*(2*O2-Vs)/f
    B = [[0, Fraction(1, 16), 0, 0],
         [16*Av, -d, 64*pi*C*iv.mpf('.06'), 0],
         [0, 0, 0, Fraction(1, 16)],
         [16*C/iv.mpf('.06'), 0, 16*Au, -d]]
    F = [Jet(0, order=order), K]
    Fp = [-truncate(K*L, order-1), derivative(K)-truncate(K*L, order-1)]

    def covariant(row):
        degree = row[0][0].order-1
        value = [poly_truncate(p, degree) for p in row]
        log = truncate(L, degree)
        result = []
        for j, p in enumerate(row):
            radial = [derivative(v) for v in p]
            transport = [-v for v in poly_mul([log, log], poly_dchi(value[j]))]
            matrix = [[v*truncate(B[i][j], degree) for v in value[i]] for i in range(4)]
            result.append(poly_add(radial, transport, *matrix))
        return result

    eighth = Jet(Fraction(1, 8), order=order)
    row = [[2*n-2*L, -2*L], [eighth, eighth],
           [8*pi*Ap*iv.mpf('.06')], [Jet(0, order=order)]]
    rows = []
    for m in range(iterations):
        first = covariant(row)
        second = first if m == 0 else covariant(first)
        degree = second[0][0].order
        old = row if m == 0 else first
        row = [[-v/4 for v in poly_add(poly_mul(poly_truncate(F, degree), second[j]),
                                      poly_mul([v/2 for v in poly_truncate(Fp, degree)],
                                               poly_truncate(old[j], degree)))] for j in range(4)]
        assert all(len(p) <= m+3 for p in row)
        rows.append(row)
    return rows, truncate(L, extra)


def compose(radial, radius):
    assert radial.order >= radius.order
    delta = Jet(0, [iv.mpf(0)]+radius.c[1:])
    result = Jet(radial.c[radius.order], order=radius.order)
    for value in reversed(radial.c[:radius.order]):
        result = result*delta+value
    return result


def cosine(x):
    assert x.order == 2 and x.c[1] == 1 and x.c[2] == 0
    theta, tau = 5*iv.pi*x.c[0], 5*iv.pi
    return Jet(0, [iv.cos(theta), -tau*iv.sin(theta), -tau*tau*iv.cos(theta)/2])


class OverlapCertificate:
    def __init__(self):
        self.bg = Background()
        path = ROOT/'data/azimuth-orbit-50.json'
        self.orbit_record = json.loads(path.read_text())
        assert self.orbit_record['status'] == 'azimuth_orbit_ball_verified'
        assert self.orbit_record['branch']['status'] == 'single_outward_resonant_half_orbit'
        self.orbit = OrbitCertificate(self.bg, '.0005')
        assert self.orbit.hashes == self.orbit_record['input_sha256']
        assert self.orbit.reference_hash == self.orbit_record['reference_sha256']
        self.inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
        self.inputs.update(self.orbit.hashes)
        self.inputs['azimuth-orbit-reference.npz'] = self.orbit.reference_hash
        self.errors = list(map(exact, self.orbit_record['physical_error_bounds']))
        self.tide = IntegratedTide(self.bg)
        self.inputs[self.tide.path.name] = self.tide.digest
        previous_path = ROOT/'data/iterated-tidal-observation-50.json'
        self.previous = json.loads(previous_path.read_text())
        assert self.previous['status'] == 'iterated_tidal_field_observation_enclosed'
        assert self.previous['input_sha256'] == self.orbit.hashes
        assert self.previous['interval']['xi'] == ['3/5', '31/50']
        assert self.previous['interval']['radial_interval'] == ['15.67', '16.31']
        branch = self.orbit_record['branch']
        assert .60 < branch['energy_fraction'][0] < branch['energy_fraction'][1] < .62
        assert 15.67 < branch['radial_extent'][0] < branch['radial_extent'][1] < 16.31
        assert self.orbit_record['observation_budget']['iterated_observation_sha256'] == hashlib.sha256(previous_path.read_bytes()).hexdigest()
        self.inputs[previous_path.name] = hashlib.sha256(previous_path.read_bytes()).hexdigest()
        relative_path = OLD/'data/solitonic-screen/relative-optics.json'
        relative_digest = hashlib.sha256(relative_path.read_bytes()).hexdigest()
        assert relative_digest == self.previous['relative_optics_sha256']
        assert relative_digest == self.orbit_record['branch']['relative_optics_sha256']
        self.inputs[relative_path.name] = relative_digest
        for record in (self.previous, self.orbit_record):
            for name, digest in record['dependency_sha256'].items():
                assert hashlib.sha256((ROOT.parent.parent/name).read_bytes()).hexdigest() == digest
        field_path = ROOT/'data/tidal-field-error-50.json'
        field = json.loads(field_path.read_text())
        assert self.previous['interval']['field_certificate_sha256'] == hashlib.sha256(field_path.read_bytes()).hexdigest()
        assert field['input_sha256'][self.tide.path.name] == self.tide.digest
        self.field_error = exact(self.previous['interval']['field_observation_error_bounds'][2])
        self.weight_error = exact(self.orbit_record['observation_budget']['weight_contribution_upper'])
        self.tide_evaluations, self.row_evaluations = 0, 0

    def orbit_jet(self, cell, value):
        x = Jet(0, [value, iv.mpf(1), iv.mpf(0)])
        a, b = self.orbit.knots[cell:cell+2]
        return [hermite(x, a, b, self.orbit.state[j, cell:cell+2],
                        self.orbit.slope[j, cell:cell+2])[0]*self.orbit.scale[j] for j in range(3)]

    def tide_jets(self, radius, order):
        left, right = Fraction(lower(radius)), Fraction(upper(radius))
        points = sorted({left, right} | {v for v in self.tide.knots if left < v < right})
        results = []
        for a, b in zip(points[:-1], points[1:]):
            index = bisect_right(self.tide.knots, (a+b)/2)-1
            l, u = self.tide.knots[index:index+2]
            r = Jet(0, [enclosure(a, b)]+([iv.mpf(1)]+[iv.mpf(0)]*(order-1) if order else []))
            width, t = exact(u-l), (r-exact(l))/exact(u-l)
            values = []
            for k, j in enumerate((1, 3)):
                y0, y1 = [exact(float(v)) for v in self.tide.state[j, index:index+2]]
                p0, p1 = [exact(float(v)) for v in self.tide.slope[j, index:index+2]]
                c2, c3 = 3*(y1-y0)-width*(2*p0+p1), 2*(y0-y1)+width*(p0+p1)
                value = (((t*(c3/4)+c2/3)*t+width*p0/2)*t+y0)*t*width+exact(self.tide.primitive[index][k])
                flux = hermite(r, l, u, self.tide.state[j, index:index+2], self.tide.slope[j, index:index+2])[0]
                values += [value, 16*flux]
            results.append(values)
            self.tide_evaluations += 1
            if self.tide_evaluations > 20000:
                raise RuntimeError('Overlap calculation exceeded 20000 tidal piece evaluations.')
        return [Jet(0, [enclosure(min(row[j].c[k].a for row in results), max(row[j].c[k].b for row in results))
                        for k in range(order+1)]) for j in range(4)]

    def rows(self, radius, extra):
        self.row_evaluations += 1
        if self.row_evaluations > 16000:
            raise RuntimeError('Overlap calculation exceeded 16000 row evaluations.')
        return weighted_rows(radius, self.orbit.background_box(radius), extra=extra)[0][-1]

    def cell(self, index, left, right):
        width = exact(right-left)
        rm, _, qm = self.orbit_jet(index, exact((left+right)/2))
        row = self.rows(rm.c[0], 0)
        field = self.tide_jets(rm.c[0], 0)
        midpoint = sum((poly_value(p, qm.c[0]).c[0]*v.c[0] for p, v in zip(row, field)), iv.mpf(0))
        midpoint *= iv.cos(5*iv.pi*exact((left+right)/2))

        r, _, q = self.orbit_jet(index, enclosure(left, right))
        radius = r.c[0]+enclosure(-self.errors[0], self.errors[0])
        chi = q.c[0]+enclosure(-self.errors[2], self.errors[2])
        assert (1+chi).a > 0
        row = self.rows(radius, 2)
        field = self.tide_jets(radius, 2)
        evaluated = [poly_value(p, chi) for p in row]
        value = sum((p.c[0]*v.c[0] for p, v in zip(evaluated, field)), iv.mpf(0))
        dr = sum((p.c[1]*v.c[0]+p.c[0]*v.c[1] for p, v in zip(evaluated, field)), iv.mpf(0))/(1+chi)
        dq = sum((poly_value(poly_dchi(p), chi).c[0]*v.c[0] for p, v in zip(row, field)), iv.mpf(0))/(1+chi)-value/(1+chi)**2
        geometry = self.errors[0]*abs(dr)+self.errors[2]*abs(dq)
        weight = 1+q.c[0]
        geometry_integral = width*weight*geometry*abs(iv.cos(5*iv.pi*enclosure(left, right)))

        composed = [poly_value([compose(v, r) for v in p], q) for p in row]
        amplitude = sum((p*compose(v, r) for p, v in zip(composed, field)), Jet(0, order=2))
        x = Jet(0, [enclosure(left, right), iv.mpf(1), iv.mpf(0)])
        integrand = amplitude*cosine(x)
        quadrature = width**3*abs(integrand.c[2])/12
        return dict(midpoint=width*midpoint, quadrature=quadrature, geometry=geometry_integral,
                    radial_derivative=upper(abs(dr)), chi_derivative=upper(abs(dq)),
                    second_taylor_coefficient=upper(abs(integrand.c[2])))

    def denominator(self):
        integral = Fraction(0)
        for i, (a, b) in enumerate(zip(self.orbit.knots[:-1], self.orbit.knots[1:])):
            width = b-a
            y0, y1 = [Fraction(float(v)) for v in self.orbit.state[2, i:i+2]]
            p0, p1 = [Fraction(float(v)) for v in self.orbit.slope[2, i:i+2]]
            integral += width*(y0+y1)/2+width*width*(p0-p1)/12
        return 1+Fraction(1, 1000)*integral


def algebra_checks():
    r, q, s = sp.symbols('r q s')
    L, B = sp.Function('L')(r), sp.Function('B')(r)
    b = sp.Function('b')(r, q)
    def transport(value):
        return sp.diff(value, r)-L*(1+q)*sp.diff(value, q)
    assert sp.expand(transport((1+q)*b)+(1+q)*b*B-(1+q)*(transport(b)+b*B-L*b)) == 0
    c = sp.Function('c')(r, q)
    second = sp.diff(c, r, 2)-2*L*(1+q)*sp.diff(c, r, q)
    second += (L*L-sp.diff(L, r))*(1+q)*sp.diff(c, q)+L*L*(1+q)**2*sp.diff(c, q, 2)
    assert sp.expand(transport(transport(c))-second) == 0
    quadratures = 0
    for power in (0, 1, 2, 3, 4):
        value = s**power
        integral = sp.integrate(value, (s, -1, 1))
        curvature = max(power*(power-1), 0)
        assert abs(integral-2*value.subs(s, 0)) <= sp.Rational(8, 24)*curvature
        quadratures += 1
    for knot in (sp.Rational(-3, 4), sp.Rational(-1, 4), 0, sp.Rational(1, 4), sp.Rational(3, 4)):
        integral = (1-knot)**3/3
        midpoint = knot*knot if knot < 0 else 0
        assert abs(integral-2*midpoint) <= sp.Rational(8, 24)*2
        quadratures += 1
    d, D, Dp = sp.symbols('d D Dp')
    y, p, a, ap, j0 = [sp.Matrix(sp.symbols(f'{name}0:2')) for name in ('y', 'p', 'a', 'ap', 'j')]
    Q = sp.Matrix(2, 2, sp.symbols('Q0:4'))
    yp = p+a*D
    pp = Q*y-d*p+j0-(ap+d*a)*D
    assert all(sp.expand(v) == 0 for v in pp+ap*D+a*Dp+d*yp-Q*y-j0-a*Dp)
    return dict(weighted_transport_identity=True, second_transport_derivative=True,
                midpoint_remainder_models=quadratures, piecewise_C1_models=5,
                weak_source_flux_shift_components=2)


def checks(obj):
    maximum, derivatives_checked = 0., 0
    for radius, chi in ((15.69, '0'), (15.9, '.0001'), (16.1, '.00008'), (16.28, '.000003')):
        state = [exact(float(v)) for v in obj.bg.sol(radius)]
        polynomial, L = weighted_rows(exact(radius), state, extra=2)
        old = iterated_rows(exact(radius), state, 1+iv.mpf(chi), 4)
        h, q = 1+iv.mpf(chi), iv.mpf(chi)
        for p, b in zip(polynomial[-1], old[2]):
            c, cq, cqq = [poly_value(v, q) for v in (p, poly_dchi(p), poly_dchi(poly_dchi(p)))]
            expected = [h*b.c[0], h*(b.c[1]-L.c[0]*b.c[0]),
                        h*(2*b.c[2]-2*L.c[0]*b.c[1]+(L.c[0]**2-L.c[1])*b.c[0])]
            computed = [c.c[0], c.c[1]-L.c[0]*h*cq.c[0],
                        2*c.c[2]-2*L.c[0]*h*cq.c[1]+(L.c[0]**2-L.c[1])*h*cq.c[0]+L.c[0]**2*h*h*cqq.c[0]]
            for a, b in zip(computed, expected):
                difference = upper(abs(a-b))
                assert difference < 1e-20
                maximum = max(maximum, difference)
                derivatives_checked += 1
    field_difference, field_checks = 0., 0
    for radius in (15.69, 15.9, 16.1, 16.28):
        values = obj.tide_jets(exact(radius), 2)
        previous = obj.tide.fields(Fraction(radius), Fraction(radius))
        for value, reference in zip(values, previous):
            field_difference = max(field_difference, upper(abs(value.c[0]-reference)))
            field_checks += 1
        for j in (0, 2):
            assert upper(abs(values[j].c[1]-values[j+1].c[0]/16)) < 1e-11
            assert upper(abs(values[j].c[2]-values[j+1].c[1]/32)) < 1e-11
            field_checks += 2
    assert field_difference < 1e-11
    return dict(weighted_row_and_derivative_checks=derivatives_checked, maximum_interval_identity_difference=maximum,
                independent_trial_field_checks=field_checks, maximum_trial_field_interval_difference=field_difference)


def run(obj, refinements, max_seconds, screen):
    start = perf_counter()
    rows = []
    indices = (0, 64, 128, 192, 255) if screen else range(len(obj.orbit.knots)-1)
    for i in indices:
        a, b = obj.orbit.knots[i:i+2]
        for j in range(refinements):
            if perf_counter()-start > max_seconds:
                raise RuntimeError('Overlap calculation stopped at its time budget.')
            rows.append(obj.cell(i, a+(b-a)*Fraction(j, refinements), a+(b-a)*Fraction(j+1, refinements)))
    denominator = exact(obj.denominator())
    midpoint = sum((row['midpoint'] for row in rows), iv.mpf(0))/denominator
    quad = sum((row['quadrature'] for row in rows), iv.mpf(0))/denominator
    geometry = sum((row['geometry'] for row in rows), iv.mpf(0))/denominator
    reference = midpoint+enclosure(-quad.b, quad.b)
    error = geometry+obj.field_error+obj.weight_error
    complete = reference+enclosure(-error.b, error.b)
    positive = not screen and complete.a > 0
    multipole = -iv.mpf(9)/32*exact(obj.tide.tide.tidal_scale)**2*complete
    return dict(status='positive_static_tidal_overlap_verified' if positive else 'screen_only' if screen else 'static_overlap_sign_not_closed',
                midpoint_integral=[lower(midpoint), upper(midpoint)], quadrature_error_upper=upper(quad),
                reference_overlap=[lower(reference), upper(reference)], geometry_error_upper=upper(geometry),
                inherited_field_error_upper=upper(obj.field_error), inherited_weight_error_upper=upper(obj.weight_error),
                complete_overlap=[lower(complete), upper(complete)], denominator=[lower(denominator), upper(denominator)],
                static_multipole_change_per_energy_eta_at_pi_over_3=[lower(multipole), upper(multipole)],
                multipole_scope='Conditional on the static response and source assumptions of tidal reciprocity; no radiation bound.',
                maximum_radial_derivative=max(row['radial_derivative'] for row in rows),
                maximum_chi_derivative=max(row['chi_derivative'] for row in rows),
                maximum_second_taylor_coefficient=max(row['second_taylor_coefficient'] for row in rows),
                cells=len(rows), row_evaluations=obj.row_evaluations,
                background_piece_evaluations=obj.orbit.background_evaluations,
                tide_piece_evaluations=obj.tide_evaluations, elapsed_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--refine', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/resonant-tidal-overlap.json')
    args = parser.parse_args()
    iv.dps = args.digits
    obj = OverlapCertificate()
    validation = dict(algebra=algebra_checks(), implementation=checks(obj))
    if args.checks_only:
        print(json.dumps(validation, indent=2))
        return
    result = run(obj, args.refine, args.max_seconds, args.screen)
    result.update(digits=args.digits, refinement=args.refine, input_sha256=obj.inputs, checks=validation,
                  scope='Complete static tidal pairing on the classical resonance; physical wave realization and outgoing news remain open.')
    dependencies = [ROOT/'check_azimuth_orbit.py', ROOT/'check_iterated_tidal_observation.py',
                    OLD/'check_barrier_certificate.py', OLD/'check_background_defect.py',
                    OLD/'check_polar_transverse.py', OLD/'check_polar_coercivity.py',
                    OLD/'check_finite_static_inverse.py', OLD/'check_asymptotic_exterior.py']
    result['dependency_sha256'] = {str(path.relative_to(ROOT.parent.parent)):
                                   hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}
    if not args.screen:
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='ascii')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
