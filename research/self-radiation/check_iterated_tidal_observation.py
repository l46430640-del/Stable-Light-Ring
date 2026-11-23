"""Iterate orbital integration by parts on the complete tidal observation.

Exact resonant orbits have a fixed azimuth interval. The recurrence uses
the homogeneous tidal equations only for the exact field, and the saved
field-error certificate then bounds each new trial observation directly.
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
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.integrate import solve_ivp, trapezoid
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
sys.path.insert(0, str(OLD))
from check_asymptotic_exterior import lower
from check_background_defect import field_equations
from check_barrier_certificate import Jet, exact, enclosure
from check_finite_static_inverse import upper
from check_polar_transverse import hermite
from check_slow_orbit_reduction import ResonanceEnclosure, reduced_row, load_tide
from finite_width_axial import OrbitSource
from screen_slow_multipole import phase_quadrature


if not __debug__:
    raise SystemExit('Run interval checks without python -O.')


def truncate(value, order):
    return Jet(0, value.c[:order+1]) if isinstance(value, Jet) else Jet(value, order=order)


def derivative(value):
    assert value.order > 0
    return Jet(0, [(j+1)*value.c[j+1] for j in range(value.order)])


def iterated_rows(radius, state, ratio, iterations=3):
    """Rows acting on (v,16*v',u/.06,16*u'/.06), at fixed energy."""
    order = 2*iterations-1
    r = Jet(0, [radius, iv.mpf(1)]+[iv.mpf(0)]*(order-1))
    y = [Jet(v, order=order) for v in state]
    for k in range(order):
        flow = field_equations(r, y, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
        for j in range(4):
            y[j].c[k+1] = flow[j].c[k]/(k+1)
    pi = Jet(iv.pi, order=order)
    A, Ap, f, O2 = y[0]*iv.mpf('.06'), y[1]*iv.mpf('.06'), 1-2*y[2], (2*y[3]).exp()
    x = A*A/iv.mpf('.08')**2
    V, Vs, Vss = A*A*(1-2*x)*(1-2*x), (1-2*x)*(1-6*x), (-8+24*x)/iv.mpf('.08')**2
    rho, pressure = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    n = (y[2]/r+4*pi*r*pressure)/f
    d = 2/r+n+(y[2]/r-4*pi*r*rho)/f
    L = 2*n-2/r
    Av = 6/(r*r*f)+4*n*n+8*pi*(V-4*O2*A*A)/f
    Au = 6/(r*r*f)+(Vs+2*A*A*Vss-O2)/f+16*pi*Ap*Ap
    C = 4*n*Ap+2*A*(2*O2-Vs)/f
    B = [[0, Fraction(1, 16), 0, 0],
         [16*Av, -d, 64*pi*C*iv.mpf('.06'), 0],
         [0, 0, 0, Fraction(1, 16)],
         [16*C/iv.mpf('.06'), 0, 16*Au, -d]]
    h = Jet(ratio, order=order)
    for k in range(order):
        h.c[k+1] = (-L*h).c[k]/(k+1)
    F = r*r*f*(h-1)
    Fp = derivative(F)

    def covariant(row):
        degree = row[0].order-1
        value = [truncate(v, degree) for v in row]
        return [derivative(row[j])+sum((value[i]*truncate(B[i][j], degree) for i in range(4)), Jet(0, order=degree))
                -truncate(L, degree)*value[j] for j in range(4)]

    b = [2*n/h-2*L, Jet(Fraction(1, 8), order=order), 8*pi*Ap*iv.mpf('.06')/h, Jet(0, order=order)]
    db = covariant(b)
    degree = order-1
    row = [-(truncate(F, degree)*db[j]+truncate(Fp, degree)*truncate(b[j], degree)/2)/4 for j in range(4)]
    rows = [row]
    for _ in range(1, iterations):
        one = covariant(row)
        two = covariant(one)
        degree = row[0].order-2
        row = [-(truncate(F, degree)*two[j]+truncate(Fp, degree)*truncate(one[j], degree)/2)/4 for j in range(4)]
        rows.append(row)
    return rows


def algebra_checks():
    r = sp.Symbol('r', positive=True)
    F, U, g = [sp.Function(s)(r) for s in ('F', 'U', 'g')]
    L = sp.diff(U, r)/U
    weighted = sp.simplify(U*(F*sp.diff(g/U, r, 2)+sp.diff(F, r)*sp.diff(g/U, r)/2))
    expanded = F*(sp.diff(g, r, 2)-2*L*sp.diff(g, r)+(L*L-sp.diff(L, r))*g)
    expanded += sp.diff(F, r)*(sp.diff(g, r)-L*g)/2
    assert sp.simplify(weighted-expanded) == 0
    arow = sp.Matrix([[sp.Function('a0')(r), sp.Function('a1')(r)]])
    matrix = sp.Matrix(2, 2, lambda i, j: sp.Function(f'B{i}{j}')(r))
    first = arow.diff(r)+arow*matrix-L*arow
    second = first.diff(r)+first*matrix-L*first
    direct = arow.diff(r, 2)+2*arow.diff(r)*matrix+arow*(matrix.diff(r)+matrix*matrix)
    direct += -2*L*(arow.diff(r)+arow*matrix)+(L*L-sp.diff(L, r))*arow
    assert all(sp.expand(v) == 0 for v in second-direct)
    h, n, Ap, A0, k, s = sp.symbols('h n Ap A0 k s', nonzero=True)
    Ls = sp.Symbol('L')
    a = sp.Matrix([[2, 0, 0, 0, -1/h]])
    B = sp.zeros(5)
    B[0, 1], B[4, 0], B[4, 2] = 1/s, -2*n, -8*sp.pi*Ap*A0
    da = a.diff(h)*(-Ls*h)+a*B-Ls*a
    expected = sp.Matrix([[2*n/h-2*Ls, 2/s, 8*sp.pi*Ap*A0/h, 0, 0]])
    assert sp.simplify(da-expected) == sp.zeros(1, 5)
    z = sp.Symbol('z', nonzero=True)
    # Exact Fourier coefficients replace an unnecessary symbolic integration.
    model = sp.expand((1+(z+1/z)/2)**8)
    coefficient = model.coeff(z, 5)
    assert coefficient > 0 and coefficient == model.coeff(z, -5)
    for order in (0, 1, 2, 3):
        transformed = sp.expand(sum(model.coeff(z, j)*sp.Rational(j, 5)**(2*order)*z**j for j in range(-8, 9)))
        assert transformed.coeff(z, 5) == coefficient
    r, p, q, f, fp, L = sp.symbols('r p q f fp L')
    G = ((2*r*f+r*r*fp)*q-r*r*f*(1+q)*L)/8
    qprime = -(1+q)*L*p
    assert sp.expand((2*r*f+r*r*fp)*q*p+r*r*f*qprime-8*p*G) == 0
    a, b, ap, bp, width = sp.symbols('a b ap bp width')
    c2, c3 = 3*(b-a)-width*(2*ap+bp), 2*(a-b)+width*(ap+bp)
    assert sp.expand(width*(a+width*ap/2+c2/3+c3/4)-width*(a+b)/2-width**2*(ap-bp)/12) == 0
    return dict(weighted_orbital_operator=True, matrix_covariant_derivative=True, angular_metric_cancellation=True,
                repeated_turning_model=4, azimuth_first_integral=True, exact_flux_primitive=True)


class IntegratedTide:
    """Exact integrals of the shared binary Hermite flux data."""
    def __init__(self, bg):
        self.tide, self.path, self.digest = load_tide(bg, 2e-9, 40)
        with np.load(self.path) as saved:
            self.knots = list(map(lambda v: Fraction(float(v)), saved['knots']))
            self.state, self.slope = [saved[key].copy() for key in ('state', 'slope')]
        assert len(self.knots) <= 4000
        self.primitive = [[Fraction(float(self.state[j, 0])) for j in (0, 2)]]
        for i, (left, right) in enumerate(zip(self.knots[:-1], self.knots[1:])):
            h = right-left
            value = []
            for k, j in enumerate((1, 3)):
                a, b = [Fraction(float(v)) for v in self.state[j, i:i+2]]
                ap, bp = [Fraction(float(v)) for v in self.slope[j, i:i+2]]
                value.append(self.primitive[-1][k]+h*(a+b)/2+h*h*(ap-bp)/12)
            self.primitive.append(value)

    def fields(self, left, right):
        i = bisect_right(self.knots, (left+right)/2)-1
        a, b = self.knots[i:i+2]
        assert a <= left <= right <= b
        h, t = exact(b-a), enclosure((left-a)/(b-a), (right-a)/(b-a))
        values = []
        for k, j in enumerate((1, 3)):
            y0, y1 = [exact(float(v)) for v in self.state[j, i:i+2]]
            p0, p1 = [exact(float(v)) for v in self.slope[j, i:i+2]]
            c2, c3 = 3*(y1-y0)-h*(2*p0+p1), 2*(y0-y1)+h*(p0+p1)
            integrated = exact(self.primitive[i][k])+h*(t*y0+t*t*h*p0/2+t**3*c2/3+t**4*c3/4)
            flux = hermite(enclosure(left, right), a, b, self.state[j, i:i+2], self.slope[j, i:i+2])[0]
            values += [integrated, 16*flux]
        return values


def implementation_checks(cert):
    trial = IntegratedTide(cert.bg)
    orbit = OrbitSource(cert.bg, .61)
    maximum, flux_excess, coefficients = 0., 0., 0
    for radius in (15.7, 15.9, 16.1, 16.3):
        ratio = orbit.energy/cert.bg.fields(np.array([radius]))['U'][0]
        state = [exact(float(v)) for v in cert.bg.sol(radius)]
        first = iterated_rows(exact(radius), state, exact(float(ratio)), 2)
        second = iterated_rows(exact(radius), state, exact(float(ratio)), 3)
        for rows in zip(first, second):
            for a, b in zip(*rows):
                assert abs(a.c[0]-b.c[0]).b < iv.mpf('1e-24')
                coefficients += 1
        old = reduced_row(np.array([radius]), cert.bg.sol(np.array([radius])), np.array([ratio]), 2, 16., np.pi, .06, .08)
        maximum = max(maximum, float(np.max(abs(np.asarray(old).ravel()-[float(v.c[0].mid) for v in first[0]]))))
        v, vp, _, u, up, *_ = trial.tide.physical_jet(radius)
        reference = [float(v[0]), 16*float(vp[0]), float(u[0])/.06, 16*float(up[0])/.06]
        for value, box in zip(reference, trial.fields(Fraction(radius), Fraction(radius))):
            flux_excess = max(flux_excess, max(0., float(box.a)-value, value-float(box.b)))
    assert maximum < 2e-14 and flux_excess < 2e-12
    return dict(truncation_consistency_components=coefficients, independent_first_row_difference=maximum,
                independent_integrated_field_components=16, maximum_integrated_field_float_excess=flux_excess)


def interval_bounds(cert, iterations):
    xi = enclosure(Fraction(3, 5), Fraction(31, 50))
    energy = iv.exp(cert.relative.log_rings[1])*(1-(1-xi)*cert.relative.depth_ratio)
    maxima = [[0.]*4 for _ in range(iterations)]
    amplitudes, components = [0.]*iterations, [0.]*4
    counts = 0
    trial = IntegratedTide(cert.bg)
    points = sorted(set(cert.partition(Fraction('15.67'), Fraction('16.31')))
                    | {v for v in trial.knots if Fraction('15.67') < v < Fraction('16.31')})
    assert len(points) <= 4001
    for left, right in zip(points[:-1], points[1:]):
        r, state, _ = cert.geometry(left, right)
        h = energy/iv.exp(cert.relative.log_ratio(left, right))
        if h.b < 1:
            continue
        h = enclosure(max(h.a, iv.mpf(1)), h.b)
        rows = iterated_rows(r, state, h, iterations)
        field = trial.fields(left, right)
        components = [max(old, upper(abs(v))) for old, v in zip(components, field)]
        for j, row in enumerate(rows):
            maxima[j] = [max(v, upper(abs(coefficient.c[0]))) for v, coefficient in zip(maxima[j], row)]
            amplitudes[j] = max(amplitudes[j], upper(abs(sum((v.c[0]*y for v, y in zip(row, field)), iv.mpf(0)))))
        counts += 1
    path = ROOT/'data/tidal-field-error-50.json'
    field = json.loads(path.read_text())
    assert field['status'] == 'growing_tide_energy_error_enclosed'
    assert all(field['input_sha256'][key] == val for key, val in cert.relative.hashes.items())
    assert field['input_sha256'][trial.path.name] == trial.digest
    errors = list(map(exact, field['local_error']['state_error_upper']))
    bounds = [upper(sum((exact(k)*e for k, e in zip(row, errors)), iv.mpf(0))) for row in maxima]
    return dict(xi=['3/5', '31/50'], radial_interval=['15.67', '16.31'], nonempty_cells=counts,
                partition_cells=len(points)-1, radial_interval_evaluations=cert.calls,
                coefficient_bounds=maxima, field_observation_error_bounds=bounds,
                trial_amplitude_bounds=amplitudes, trial_component_bounds=components,
                field_certificate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def azimuth_diagnostic(bg, orbit):
    calls = 0
    def rhs(theta, state):
        nonlocal calls
        calls += 1
        if calls > 16000:
            raise RuntimeError('Azimuth diagnostic exceeded 16000 right-hand-side evaluations.')
        r, p, q = state[:3]
        assert 15.6 < r < 16.4
        y, yp, ypp = bg.sol(r), bg.sol(r, 1), bg.sol(r, 2)
        f, fp, fpp = 1-2*y[2], -2*yp[2], -2*ypp[2]
        L, Lp = -2*yp[3]-2/r, -2*ypp[3]+2/r**2
        K, Kp, Kpp = r*r*f, 2*r*f+r*r*fp, 2*f+4*r*fp+r*r*fpp
        G = (Kp*q-K*(1+q)*L)/8
        Gr = (Kpp*q-Kp*(1+q)*L-K*(1+q)*Lp)/8
        matrix = np.array([[0., 1., 0.], [Gr, 0., (Kp-K*L)/8],
                           [-(1+q)*Lp*p, -(1+q)*L, -L*p]])
        fundamental = state[3:].reshape(3, 3)
        return np.r_[p, G, -(1+q)*L*p, (matrix@fundamental).ravel()]
    initial = np.r_[orbit.a, 0., 0., np.eye(3).ravel()]
    length = 5*np.pi
    solution = solve_ivp(rhs, (0., length), initial, method='DOP853', rtol=2e-8, atol=2e-10,
                         max_step=.05, dense_output=True)
    assert solution.success and len(solution.t) <= 4000
    terminal = solution.y[:, -1]
    D = terminal[3:].reshape(3, 3)[1, 0]
    assert abs(D) > .001
    records = []
    for count in (65, 129):
        theta = np.linspace(0, length, count)
        state = solution.sol(theta)
        Phi = state[3:].T.reshape(count, 3, 3)
        inverse = np.linalg.inv(Phi)
        K = np.einsum('aij,bjk->abik', Phi, inverse)
        K *= (theta[:, None] >= theta[None, :])[:, :, None, None]
        projection = np.einsum('ai,bj->abij', Phi[:, :, 0], (Phi[-1]@inverse)[:, 1, :])/D
        K -= projection
        gains = np.max(trapezoid(abs(K), theta, axis=1), axis=0)
        rr, pp, qq = state[:3]
        f = 1-2*bg.sol(rr)[2]
        constraint = max(abs(4*pp*pp-rr*rr*f*qq))
        records.append(dict(points=count, green_component_integral_max=gains.tolist(), first_integral_defect=float(constraint)))
    return dict(scope='Floating trajectory and inverse-kernel diagnostics only.', rtol=2e-8, atol=2e-10, rhs_evaluations=calls,
                time_nodes=len(solution.t), left_turning=orbit.a, endpoint_radial_difference=float(terminal[0]-orbit.b),
                endpoint_velocity=float(terminal[1]), endpoint_energy_ratio_minus_one=float(terminal[2]),
                shooting_derivative=float(D), records=records)


def diagnostics(cert, iterations, interpolation_nodes):
    bg = cert.bg
    tide, path, digest = load_tide(bg, 2e-9, 40)
    prior = json.loads((ROOT/'data/slow-orbit-reduction.json').read_text())
    xi = prior['floating']['roots']['5'][-1]
    orbit = OrbitSource(bg, xi)
    r, time, phi, ht, ha, weight, gap, points = phase_quadrature(orbit, tide, 128)
    v, vp, _, u, up, _, _, _, _ = tide.physical_jet(r)
    state = np.array([v, 16*vp, u/.06, 16*up/.06])
    cosine = np.cos(2*phi)
    records, maxfirst = [], 0.
    for count in interpolation_nodes:
        nodes = np.cos(np.pi*np.arange(count+1)/count)
        radii = (15.67+16.31)/2+(16.31-15.67)*nodes/2
        values = []
        for radius in radii:
            ratio = orbit.energy/bg.fields(np.array([radius]))['U'][0]
            rows = iterated_rows(exact(float(radius)), [exact(float(v)) for v in bg.sol(radius)], exact(float(ratio)), iterations)
            values.append([[float(v.c[0].mid) for v in row] for row in rows])
            first = reduced_row(np.array([radius]), bg.sol(np.array([radius])), np.array([ratio]), 2, 16., np.pi, .06, .08)
            maxfirst = max(maxfirst, float(np.max(abs(np.array(values[-1][0])-np.asarray(first).ravel()))))
        assert maxfirst < 2e-14
        coefficients = chebfit(nodes, np.array(values).reshape(count+1, -1), count)
        evaluated = chebval((2*r-15.67-16.31)/(16.31-15.67), coefficients).reshape(iterations, 4, -1)
        amplitudes = np.sum(evaluated*state[None, :, :], axis=1)
        rows = []
        for j, amplitude in enumerate(amplitudes):
            rows.append(dict(iterations=j+1, overlap=float(weight@(amplitude*cosine)),
                             amplitude_L1=float(weight@abs(amplitude)), amplitude_sup=float(max(abs(amplitude))),
                             absolute_integral=float(weight@abs(amplitude*cosine))))
        records.append(dict(interpolation_nodes=count+1, rows=rows,
                            coefficient_tail=float(np.max(abs(coefficients[-5:])))))
    return dict(xi=xi, half_azimuth=ha, azimuth_resonance_defect=abs(2*ha-5*np.pi),
                points=points, field_sha256=digest, first_row_difference=maxfirst, records=records,
                azimuth=azimuth_diagnostic(bg, orbit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--iterations', type=int, choices=(2, 3), default=3)
    parser.add_argument('--interval-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data/iterated-tidal-observation.json')
    args = parser.parse_args()
    start = perf_counter()
    cert = ResonanceEnclosure(args.digits)
    algebra = algebra_checks()
    checks = implementation_checks(cert)
    interval = interval_bounds(cert, args.iterations)
    report = dict(status='iterated_tidal_field_observation_enclosed', scope='Field error on the exact orbit only; no nonzero overlap or physical news certificate.',
                  digits=args.digits, algebra=algebra, implementation_checks=checks, interval=interval, input_sha256=cert.relative.hashes,
                  relative_optics_sha256=cert.relative.record_sha256)
    if not args.interval_only:
        report['diagnostics'] = diagnostics(cert, args.iterations, (48, 72))
        records = report['diagnostics']['records']
        for left, right in zip(records[0]['rows'], records[1]['rows']):
            assert abs(left['overlap']-right['overlap']) < 1e-12
        assert max(abs(row['overlap']-records[-1]['rows'][0]['overlap']) for row in records[-1]['rows']) < 2e-12
        a = report['diagnostics']['azimuth']
        assert abs(a['endpoint_velocity']) < 1e-9 and abs(a['endpoint_radial_difference']) < 1e-9
        assert max(row['first_integral_defect'] for row in a['records']) < 1e-10
    paths = [ROOT/'check_slow_orbit_reduction.py', ROOT/'screen_slow_multipole.py', ROOT/'check_tidal_field_error.py',
             ROOT/'data/slow-orbit-reduction-50.json', ROOT/'data/tidal-field-error-50.json',
             OLD/'check_barrier_certificate.py', OLD/'check_background_defect.py', OLD/'check_polar_transverse.py',
             OLD/'check_polar_coercivity.py', OLD/'check_relative_optics.py']
    report['dependency_sha256'] = {str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['elapsed_seconds'] = perf_counter()-start
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
