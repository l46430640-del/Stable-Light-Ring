"""Reduce the resonant tidal overlap and enclose a lower orbital resonance.

The resonance enclosure uses inherited exact-background certificates.
Tidal overlaps remain floating diagnostics; no radiation sign is certified.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from mpmath import iv
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
sys.path.insert(0, str(OLD))
from check_asymptotic_exterior import lower
from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import upper
from check_polar_coercivity import physical_cell, fields
from check_relative_optics import RelativePotential
from check_regular_center import exponential
from check_quadratic_channels import FrequencyScreen
from screen_slow_multipole import TidalResponse, ZeroSource, phase_quadrature
from finite_width_axial import OrbitSource
from polar_static import angular_parameters, coefficients
from trapped_axial import Background


def bounds(value):
    return [lower(value), upper(value)]


def reduced_row(r, background_state, h, k, radial_scale, pi, central, sigma):
    """Coefficients on (v, radial_scale*v', u/central, radial_scale*u'/central)."""
    aa, bb, mass, zeta = background_state
    A, Ap = central*aa, central*bb
    f = 1-2*mass
    O2 = np.exp(2*zeta) if isinstance(zeta, np.ndarray) else exponential(2*zeta)
    ratio = A*A/(sigma*sigma)
    V, Vs = A*A*(1-2*ratio)*(1-2*ratio), (1-2*ratio)*(1-6*ratio)
    rho = (O2*A*A+f*Ap*Ap+V)/2
    pressure = (O2*A*A+f*Ap*Ap-V)/2
    n = (mass/r+4*pi*r*pressure)/f
    fp = 2*mass/r-8*pi*r*rho
    d = 2/r+n+fp/(2*f)
    F0 = 4*pi*(2*O2*A*A-V)
    np_ = F0/f-d*n
    App = (Vs-O2)*A/f-d*Ap
    C = 4*n*Ap+2*A*(2*O2-Vs)/f
    Av = 4*n*n+k*(k+1)/(r*r*f)+8*pi*(V-4*O2*A*A)/f
    logU, logUp = 2*n-2/r, 2*np_+2/(r*r)
    z = [n-h*logU, h/radial_scale, 4*pi*Ap*central, 0]
    zp = [h*(Av+logU*logU-logUp)+np_, (h*(-d-2*logU)+n)/radial_scale,
          4*pi*(h*C+App)*central, 4*pi*Ap*central/radial_scale]
    radial_fraction = 1-1/h
    B = radial_fraction*(n+fp/(2*f)-logU)-logU/(2*h)
    return [-2*r*r*f/(k*k)*(radial_fraction*a+B*b) for a, b in zip(zp, z)]


def load_tide(bg, tolerance, outer):
    path = ROOT/f'data/slow-tide-l2-tol-{tolerance:.0e}-R{outer:g}.npz'
    record = json.loads((ROOT/'data/slow-static-multipole.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    matched = [r for r in record['trials'] if r['field_path'] == path.name]
    assert len(matched) == 1 and matched[0]['field_sha256'] == digest
    with np.load(path) as data:
        x, y, slope = [data[key].copy() for key in ('knots', 'state', 'slope')]
        tide = TidalResponse.__new__(TidalResponse)
        tide.inner, tide.outer = float(data['inner']), float(data['outer'])
        tide.tidal_scale = float(data['tidal_scale'])
    tide.background, tide.source = bg, ZeroSource(bg)
    tide.degree, tide.source_derivative_shift = 2, False
    tide.eigenvalue, tide.funk, tide.gain = angular_parameters(2)
    tide.solution = SimpleNamespace(x=x, y=y, sol=CubicHermiteSpline(x, y, slope, axis=1))
    tide._integrated_states = tide.solution.sol.antiderivative()
    g, _, _, npp, App, *_ = coefficients(bg, x, 2)
    v, vp, u, up = y[:4]
    u, up = bg.central*u, bg.central*up
    wp = -2*g['nup']*v-8*np.pi*g['Ap']*u
    wpp = -2*npp*v-2*g['nup']*vp-8*np.pi*(App*u+g['Ap']*up)
    tide._wprime = CubicHermiteSpline(x, wp, wpp)
    tide._wintegral = tide._wprime.antiderivative()
    return tide, path, digest


def algebra_checks():
    r = sp.Symbol('r', positive=True)
    e, k = sp.symbols('e k', positive=True)
    v, U, P, a = [sp.Function(s)(r) for s in ('v', 'U', 'P', 'a')]
    Q = P*P*(1-U/e)
    Z = e*sp.diff(v/U, r)+a
    reduced = -2/k**2*(Q*sp.diff(Z/U, r)+sp.diff(Q, r)*Z/(2*U))
    h, lp = e/U, sp.diff(U, r)/U
    candidate = -2*P*P/(U*k*k)*((1-1/h)*sp.diff(Z, r)
        +((1-1/h)*(sp.diff(P, r)/P-lp)-lp/(2*h))*Z)
    assert sp.simplify(reduced-candidate) == 0
    w, theta = sp.Function('w')(r), sp.Symbol('theta', real=True)
    primitive = (2*sp.sqrt(e)*v/(k*U)-w/(k*sp.sqrt(e)))*sp.sin(theta)
    primitive += 2*sp.sqrt(Q)*Z*sp.cos(theta)/(k*k*U)
    derivative = sp.sqrt(Q)*sp.diff(primitive, r)+k*U/sp.sqrt(e)*sp.diff(primitive, theta)
    remainder = (2*v-U*w/e)*sp.cos(theta)-derivative-reduced*sp.cos(theta)
    remainder -= sp.sqrt(Q)*(sp.diff(w, r)+2*a)*sp.sin(theta)/(k*sp.sqrt(e))
    assert sp.simplify(remainder) == 0
    F, b = sp.Function('F')(r), sp.Function('b')(r)
    twice = -Q*sp.diff(sp.diff(F/b, r)/b, r)-sp.diff(Q, r)*sp.diff(F/b, r)/(2*b)
    direct = -(Q*sp.diff(F, r, 2)+(sp.diff(Q, r)/2-3*Q*sp.diff(b, r)/b)*sp.diff(F, r)
        +(3*Q*(sp.diff(b, r)/b)**2-Q*sp.diff(b, r, 2)/b-sp.diff(Q, r)*sp.diff(b, r)/(2*b))*F)/b**2
    assert sp.simplify(twice-direct) == 0
    energy = sp.Symbol('energy', positive=True)
    wr, action = sp.Function('wr')(energy), sp.Function('J')(energy)
    angular = sp.sqrt(energy)-action*wr
    derivative = sp.diff(angular/wr, energy).subs(sp.diff(action, energy), 1/(2*sp.sqrt(energy)*wr))
    assert sp.simplify(derivative+sp.sqrt(energy)*sp.diff(wr, energy)/wr**2) == 0
    # A smooth turning orbit verifies both integrations, including its endpoints.
    t = sp.Symbol('t', real=True)
    for harmonic in (1, 2, 3, 5):
        Ft = (1+sp.cos(t))**5
        original = sp.integrate(sp.expand_trig(Ft*sp.cos(harmonic*t)), (t, 0, sp.pi))
        transformed = sp.integrate(sp.expand_trig(-sp.diff(Ft, t, 2)*sp.cos(harmonic*t)/harmonic**2), (t, 0, sp.pi))
        assert sp.simplify(original-transformed) == 0
    return dict(reduced_row_identity=True, twice_integrated_identity=True,
                exact_boundary_primitive_and_mixed_residual=True,
                frequency_ratio_monotonicity=True, turning_endpoint_models=4)


def diagnostic(tide, xi, j, subdivisions):
    bg, k = tide.background, 2
    orbit = OrbitSource(bg, xi)
    r, time, phi, ht, ha, weight, gap, points = phase_quadrature(orbit, tide, subdivisions)
    v, vp, vpp, u, up, _, w, wp, wpp = tide.physical_jet(r)
    g = bg.fields(r)
    e, U = orbit.energy, g['U']
    phase = k*phi+(j*np.pi-k*ha)*time
    delta = (k*ha-j*np.pi)/ht
    F = 2*v-U/e*w
    # This version differentiates the represented functions, not their ODE.
    bgp, bgpp = bg.sol(r, 1), bg.sol(r, 2)
    logU = -2*bgp[3]-2/r
    logUp = -2*bgpp[3]+2/r**2
    Up, Upp = U*logU, U*(logU*logU+logUp)
    p = g['p']
    Q = p*p*gap/e
    Qp = p*p/e*(2*(-bgp[3]-bgp[2]/g['f'])*gap-Up)
    Fp = 2*vp-(Up*w+U*wp)/e
    Fpp = 2*vpp-(Upp*w+2*Up*wp+U*wpp)/e
    b, bp, bpp = k*U/np.sqrt(e)-delta, k*Up/np.sqrt(e), k*Upp/np.sqrt(e)
    assert min(b) > 0
    twice = -(Q*Fpp+(Qp/2-3*Q*bp/b)*Fp+(3*Q*(bp/b)**2-Q*bpp/b-Qp*bp/(2*b))*F)/(b*b)
    row = np.asarray(reduced_row(r, bg.sol(r), e/U, k, bg.rs, np.pi, bg.central, bg.scale))
    state = np.asarray([v, bg.rs*vp, u/bg.central, bg.rs*up/bg.central])
    reduced = np.sum(row*state, axis=0)
    cosine = np.cos(phase)
    return dict(xi=xi, j=j, subdivisions=subdivisions, points_including_gap=points,
                beat=delta, original=float(weight@(F*cosine)),
                twice_differentiated=float(weight@(twice*cosine)),
                ode_reduced=float(weight@(reduced*cosine)),
                absolute_original=float(weight@abs(F*cosine)),
                absolute_reduced=float(weight@abs(reduced*cosine)),
                reduced_amplitude_L1=float(weight@abs(reduced)),
                coefficient_absolute_integrals=(abs(row*cosine)*weight).sum(axis=1).tolist(),
                maximum_reduced_amplitude=float(max(abs(reduced))))


class ResonanceEnclosure:
    def __init__(self, digits):
        iv.dps = digits
        self.started = perf_counter()
        self.bg = Background()
        self.relative = RelativePotential.load(self.bg)
        self.knots = [Fraction(float(x)) for x in self.bg.sol.x]
        self.calls = self.max_cells = 0
        path = OLD/'data/s08a06/orbital-frequency-monotonicity-50.json'
        self.monotonicity = json.loads(path.read_text())['certificate']
        assert self.monotonicity['input_sha256'] == self.relative.hashes
        assert self.monotonicity['status'] == 'exact_star_orbital_frequency_strictly_decreasing'
        self.monotonicity_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def partition(self, left, right, subdivisions=0):
        points = {left, right} | {x for x in self.knots if left < x < right}
        points |= {x for x in self.relative.nodes if left < x < right}
        if subdivisions:
            points |= {left+(right-left)*Fraction(i, subdivisions) for i in range(1, subdivisions)}
        points = sorted(points)
        self.max_cells = max(self.max_cells, len(points)-1)
        if len(points) > 4001:
            raise RuntimeError('The interval integral exceeds 4000 cells.')
        return points

    def geometry(self, left, right):
        if perf_counter()-self.started > 120:
            raise RuntimeError('Stopped after the 120 second enclosure budget.')
        self.calls += 1
        piece = bisect_right(self.knots, (left+right)/2)-1
        r = enclosure(left, right)
        state = physical_cell(self.bg, piece, left, right)
        g = fields(r, state)
        assert g['f'].a > 0
        return r, state, g

    def cap(self, bracket, cut, left_cap):
        a, b = bracket
        lo, hi = (a, cut) if left_cap else (cut, b)
        partition = self.partition(lo, hi)
        slope_lo, slope_hi, Klo, Khi = float('inf'), 0., float('inf'), 0.
        for l, rr in zip(partition[:-1], partition[1:]):
            r, _, g = self.geometry(l, rr)
            index = bisect_right(self.relative.nodes, (l+rr)/2)-1
            slope = (1 if not left_cap else -1)*self.relative.ranged[index][0]
            K = 1/(r*iv.sqrt(g['f']))
            slope_lo, slope_hi = min(slope_lo, lower(slope)), max(slope_hi, upper(slope))
            Klo, Khi = min(Klo, lower(K)), max(Khi, upper(K))
        assert slope_lo > 0
        dmin, dmax = (cut-b, cut-a) if left_cap else (a-cut, b-cut)
        assert dmin > 0
        low = 2*exact(Klo)*iv.sqrt(exact(dmin)/(exact(slope_hi)*iv.exp(exact(slope_hi)*exact(dmax))))
        high = 2*exact(Khi)*iv.sqrt(exact(dmax)/exact(slope_lo))
        return enclosure(low.a, high.b), dict(turning_bracket=list(map(str, bracket)), cut=str(cut),
            log_slope_bounds=[slope_lo, slope_hi], geometric_bounds=[Klo, Khi], integral=bounds(enclosure(low.a, high.b)))

    def angle(self, fraction, subdivisions):
        xi = Fraction(fraction)
        orbit = OrbitSource(self.bg, float(xi))
        turns = []
        for reference in (orbit.a, orbit.b):
            proposal = (Fraction(float(reference)) - Fraction('.001'), Fraction(float(reference)) + Fraction('.001'))
            bracket = self.relative.turning_bracket(*proposal, fraction=xi)
            turns.append(tuple(map(Fraction, bracket['bracket'])))
        left, right = turns[0][1]+Fraction('.003'), turns[1][0]-Fraction('.003')
        cap_a, cap_a_record = self.cap(turns[0], left, True)
        cap_b, cap_b_record = self.cap(turns[1], right, False)
        energy = iv.exp(self.relative.log_rings[1])*(1-(1-exact(xi))*self.relative.depth_ratio)
        points = self.partition(left, right, subdivisions)
        total = iv.mpf(0)
        for a, b in zip(points[:-1], points[1:]):
            r, _, g = self.geometry(a, b)
            h = energy/iv.exp(self.relative.log_ratio(a, b))
            assert h.a > 1
            total += exact(b-a)/(r*iv.sqrt(g['f'])*iv.sqrt(h-1))
        angle = total+cap_a+cap_b
        return dict(xi=str(xi), subdivisions=subdivisions, ratio=bounds(angle/iv.pi),
                    half_azimuth=bounds(angle), interior=bounds(total),
                    caps=[cap_a_record, cap_b_record], interval_cells=len(points)-1)

    def coefficient_bound(self, xi_low, xi_high):
        xi = enclosure(Fraction(xi_low), Fraction(xi_high))
        # This interval contains the turning points for all selected energies.
        left, right = Fraction('15.67'), Fraction('16.31')
        for r0 in (left, right):
            assert self.relative.fraction(r0, r0)[0].a > xi.b
        energy = iv.exp(self.relative.log_rings[1])*(1-(1-xi)*self.relative.depth_ratio)
        bound = [0.]*4
        count = 0
        for a, b in zip((points:=self.partition(left, right))[:-1], points[1:]):
            r, state, _ = self.geometry(a, b)
            h = energy/iv.exp(self.relative.log_ratio(a, b))
            if h.b < 1:
                continue
            h = enclosure(max(h.a, iv.mpf(1)), h.b)
            row = reduced_row(r, state, h, 2, exact(float(self.bg.rs)), iv.pi, iv.mpf('.06'), iv.mpf('.08'))
            bound = [max(old, upper(abs(value))) for old, value in zip(bound, row)]
            count += 1
        return dict(xi=[str(Fraction(xi_low)), str(Fraction(xi_high))], radial_interval=[str(left), str(right)],
                    components=['v', 'r_0 v_prime', 'u/0.06', 'r_0 u_prime/0.06'],
                    reference_radius_exact=str(Fraction(float(self.bg.rs))),
                    component_bounds=bound, sum_bound=upper(sum(map(exact, bound), iv.mpf(0))),
                    nonempty_cells=count,
                    scope='Exact-background coefficients on exact orbits in the stated energy range; not a tidal-field error bound.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT/'data/slow-orbit-reduction.json')
    parser.add_argument('--interval-only', action='store_true')
    args = parser.parse_args()
    if not __debug__ or not 25 <= args.digits <= 60:
        raise SystemExit('Use checked Python and 25 through 60 interval digits.')
    start = perf_counter()
    report = dict(status='slow_orbit_reduction_in_progress', algebra=algebra_checks(), digits=args.digits)
    cert = ResonanceEnclosure(args.digits)
    reports = [cert.angle(xi, 1024) for xi in ('.58', '.6', '.62', '.65')]
    reports += [cert.angle(xi, 3584) for xi in ('.6', '.62')]
    print(json.dumps(dict(angles=[(r['xi'], r['ratio']) for r in reports]), indent=2), flush=True)
    good_low = [r for r in reports if r['ratio'][1] < 2.5]
    good_high = [r for r in reports if r['ratio'][0] > 2.5]
    assert good_low and good_high
    left = max(good_low, key=lambda r: Fraction(r['xi']))['xi']
    right = min(good_high, key=lambda r: Fraction(r['xi']))['xi']
    assert Fraction(left) < Fraction(right)
    report['resonance'] = dict(status='unique_exact_background_classical_resonance_enclosed', k=2, j=5,
        bracket=[left, right], angles=reports, coefficient_bound=cert.coefficient_bound(left, right),
        monotonicity_sha256=cert.monotonicity_hash, inherited_background_hashes=cert.relative.hashes,
        relative_optics_sha256=cert.relative.record_sha256, maximum_interval_cells=cert.max_cells,
        interval_cell_evaluations=cert.calls)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
    paths = [OLD/'check_relative_optics.py', OLD/'check_polar_coercivity.py', OLD/'check_regular_center.py',
             OLD/'data/solitonic-screen/relative-optics.json', OLD/'data/s08a06/orbital-frequency-monotonicity-50.json']
    if not args.interval_only:
        screen = FrequencyScreen()
        roots = {}
        for j in (5, 6, 7, 8):
            roots[j] = [brentq(lambda xi: 2*screen.orbit(xi, n)['ratio']-j, .1, .998, xtol=2e-11) for n in (64, 128)]
        report['floating'] = dict(roots=roots, trials=[])
        for tolerance, outer in ((1e-6, 40.), (2e-7, 40.), (2e-9, 40.), (2e-9, 38.)):
            tide, path, digest = load_tide(screen.background, tolerance, outer)
            rows = [diagnostic(tide, roots[j][-1], j, n) for j in (5, 6, 7, 8) for n in (64, 128)]
            assert max(abs(r['original']-r['twice_differentiated']) for r in rows) < 2e-11
            assert max(abs(r['original']-r['ode_reduced']) for r in rows) < 2e-10
            report['floating']['trials'].append(dict(tolerance=tolerance, outer=outer, rows=rows,
                input_path=str(path.relative_to(ROOT.parents[1])), input_sha256=digest))
            paths.append(path)
        report['floating'].update(root_orbit_evaluations=screen.calls,
            maximum_root_points=screen.max_with_gap_points, bvp_solves=0)
        paths += [ROOT/'screen_slow_multipole.py', ROOT/'check_quadratic_channels.py',
                  ROOT/'data/slow-static-multipole.json', OLD/'polar_static.py', OLD/'finite_width_axial.py', OLD/'trapped_axial.py']
    report.update(status='classical_resonance_certified_overlap_not_certified', seconds=perf_counter()-start,
        input_sha256={str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='A classical 5:2 resonance and reduced-observation coefficients on the inherited exact star. No discrete wave pair, nonzero static multipole certificate or physical radiation estimate.')
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
    print(json.dumps(dict(status=report['status'], bracket=report['resonance']['bracket'],
        coefficient_bounds=report['resonance']['coefficient_bound']['component_bounds'], seconds=report['seconds']), indent=2))


if __name__ == '__main__':
    main()
