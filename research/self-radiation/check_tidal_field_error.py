"""Bound a growing quadrupolar tide using the inherited static coercivity.

The exterior extension uses the exact star's ADM mass, uniformly over its
certified interval. A nearby nominal mass would not give a finite-energy
error. This script does not certify the resonant overlap or radiation.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import factorial
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
from check_polar_coercivity import check_inputs, physical_cell
from check_polar_endpoint_budget import central_coefficients, trace_bound, norm, tail_constants
from check_polar_interval_residual import ResidualEnclosure, coefficients
from check_polar_transverse import central_bound, hermite
from polar_transverse import TransverseDeformation
from trapped_axial import Background
from check_slow_orbit_reduction import load_tide


if not __debug__:
    raise SystemExit('Run certificates without python -O.')


class TidalEnclosure(ResidualEnclosure):
    def __init__(self):
        self.bg = Background()
        self.hashes = check_inputs()
        self.tide, self.path, digest = load_tide(self.bg, 2e-9, 40)
        self.hashes[self.path.name] = digest
        with np.load(self.path) as data:
            self.knots, self.state, self.slope = [data[k].copy() for k in ('knots', 'state', 'slope')]
        self.knots = [Fraction(float(v)) for v in self.knots]
        self.background_knots = [Fraction(float(v)) for v in self.bg.sol.x]
        self.deformation = TransverseDeformation(self.bg)
        self.deformation_knots = [Fraction(float(v)) for v in self.deformation.spline.x]
        deformation_path = OLD/'data/solitonic-screen/polar-transverse-deformation.npz'
        with np.load(deformation_path) as data:
            self.values, self.slopes = [data[k].copy() for k in ('values', 'slopes')]
        self.hashes[deformation_path.name] = hashlib.sha256(deformation_path.read_bytes()).hexdigest()
        path = OLD/'data/solitonic-screen/polar-transverse-50.json'
        certificate = json.loads(path.read_text())
        assert certificate['status'] == 'all_polar_degrees_coercivity_verified'
        assert certificate['input_sha256'] == check_inputs()
        assert certificate['deformation_sha256'] == self.hashes[deformation_path.name]
        self.hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = json.loads((OLD/'data/s08a06/global-background.json').read_text())
        self.omega = enclosure(*report['endpoint_graphs']['frequency_bound_on_the_graph'])
        self.mass = enclosure(*report['endpoint_graphs']['ADM_mass_interval'])
        path = ROOT/'data/slow-orbit-reduction-50.json'
        observation = json.loads(path.read_text())
        assert observation['status'] == 'classical_resonance_certified_overlap_not_certified'
        assert observation['resonance']['inherited_background_hashes'] == check_inputs()
        c = observation['resonance']['coefficient_bound']
        assert c['xi'] == ['3/5', '31/50'] and c['radial_interval'] == ['1567/100', '1631/100']
        radius = Fraction(c['reference_radius_exact'])
        for i, (bound, target) in enumerate(zip(c['component_bounds'], ('.003354', '.003355', '.000830', '.000004418'))):
            assert Fraction(bound)*(radius/16 if i in (1, 3) else 1) < Fraction(target)
        self.hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = ROOT/'data/slow-static-multipole.json'
        self.hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.partition = sorted(set(self.knots) | set(self.background_knots[1:])
                                | set(self.deformation_knots) | {Fraction(15), Fraction(20)}
                                | set(map(Fraction, ('15.4', '16.31', '16.6'))))
        additions = {a+(b-a)*Fraction(j, 4) for a, b in zip(self.partition[:-1], self.partition[1:])
                     if Fraction('.1') <= a < Fraction('.5') for j in (1, 2, 3)}
        self.partition = sorted(set(self.partition) | additions)
        assert self.partition[0] == Fraction(.002) and self.partition[-1] == 40
        assert len(self.knots) <= 4000 and len(self.partition)-1 <= 6500

    def source_fields(self, *args, **kwargs):
        return Jet(0, order=1), Jet(0, order=1)

    def background_cell(self, piece, radius):
        y = physical_cell(self.bg, piece, radius.a, radius.b)
        if hasattr(self, 'tail_anchors') and radius.a >= 25:
            amp = min(upper(exact(value)*iv.exp(-iv.mpf('.98')*(radius-anchor)))
                      for anchor, value in self.tail_anchors.items() if radius.a >= anchor)
            amp = exact(amp)
            rate = iv.mpf('1.96')
            delta_mass = 21*amp*amp*(radius**2/rate+2*radius/rate**2+2/rate**3)
            delta_eta = 30*amp*amp*(radius/rate+1/rate**2)
            m = self.mass-enclosure(0, delta_mass.b)
            f = 1-2*m/radius
            bounds = [enclosure(0, amp/iv.mpf('.06')), enclosure(-iv.mpf('1.5')*amp/iv.mpf('.06'), 0),
                      m/radius, iv.ln(self.omega)-iv.ln(f)/2+enclosure(0, delta_eta.b)]
            for i, bound in enumerate(bounds):
                lo, hi = max(y[i].a, bound.a), min(y[i].b, bound.b)
                assert lo <= hi, (radius, i)
                y[i] = enclosure(lo, hi)
        return y

    def geometry(self, left, right):
        middle = (left+right)/2
        piece = bisect_right(self.background_knots, middle)-1
        radius = enclosure(left, right)
        y = self.background_cell(piece, radius)
        flow = field_equations(radius, y, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
        r = Jet(0, [radius, iv.mpf(1)])
        state = [Jet(0, [v, F]) for v, F in zip(y, flow)]
        c = coefficients(r, state, self.transverse(r, middle))
        P = radius**2*self.omega*iv.exp(-y[3])*iv.sqrt(c['f'].c[0])
        return radius, y, c, P

    def jet_residual(self, left, right, point=False):
        middle = (left+right)/2
        radius = exact(middle) if point else enclosure(left, right)
        piece = bisect_right(self.background_knots, middle)-1
        index = bisect_right(self.knots, middle)-1
        r = Jet(0, [radius, iv.mpf(1), iv.mpf(0), iv.mpf(0)])
        y = [Jet(v, order=3) for v in self.background_cell(piece, radius)]
        for k in range(3):
            flow = field_equations(r, y, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
            for j in range(4):
                y[j].c[k+1] = flow[j].c[k]/(k+1)
        pi = Jet(iv.pi, order=3)
        root4pi = Jet(iv.sqrt(4*iv.pi), order=3)
        A, Ap, f, O2 = y[0]*iv.mpf('.06'), y[1]*iv.mpf('.06'), 1-2*y[2], (2*y[3]).exp()
        x = A*A/iv.mpf('.08')**2
        V, Vs, Vss = A*A*(1-2*x)*(1-2*x), (1-2*x)*(1-6*x), (-8+24*x)/iv.mpf('.08')**2
        rho, pressure = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
        n = (y[2]/r+4*pi*r*pressure)/f
        d = 2/r+n+(y[2]/r-4*pi*r*rho)/f
        W = 1/(r*r*f)
        cross = root4pi*(4*n*Ap+2*A*(2*O2-Vs)/f)
        Q = [[6*W+4*n*n+8*pi*(V-4*O2*A*A)/f, cross],
             [cross, 6*W+(Vs+2*A*A*Vss-O2)/f+16*pi*Ap*Ap]]
        state, prime = [], []
        for j in range(4):
            v, vp = hermite(r, self.knots[index], self.knots[index+1],
                            self.state[j, index:index+2], self.slope[j, index:index+2])
            state.append(v)
            prime.append(vp)
        scale = Jet(iv.sqrt(4*iv.pi)*iv.mpf('.06'), order=3)
        field, derivative = [state[0], scale*state[2]], [prime[0], scale*prime[2]]
        flux, fluxp = [state[1], scale*state[3]], [prime[1], scale*prime[3]]
        r0 = [derivative[j]-flux[j] for j in range(2)]
        raw = [fluxp[j]+d*flux[j]-sum((Q[j][k]*field[k] for k in range(2)), Jet(0, order=3))
               for j in range(2)]
        return r0+raw

    def cell(self, left, right):
        center, ranged = self.jet_residual(left, right, True), self.jet_residual(left, right)
        offset = enclosure(-(right-left)/2, (right-left)/2)
        narrowed = []
        for a, b in zip(center, ranged):
            value = sum((a.c[k]*offset**k for k in range(3)), iv.mpf(0))+b.c[3]*offset**3
            lo, hi = max(value.a, b.c[0].a), min(value.b, b.c[0].b)
            assert lo <= hi
            narrowed.append(enclosure(lo, hi))
        _, _, c, P = self.geometry(left, right)
        self.last_geometry = c, P
        r0 = narrowed[:2]
        q = [narrowed[j+2]-sum((c['T'][j][k].c[0]*r0[k] for k in range(2)), iv.mpf(0)) for j in range(2)]
        energy = exact(right-left)*P*(sum((abs(v)**2 for v in r0), iv.mpf(0))
                                     +sum((abs(v)**2 for v in q), iv.mpf(0))/(4*c['W'].c[0]))
        return upper(energy), 0., [upper(abs(v)) for v in r0+q]


def algebra_checks():
    r, R, M, A, b = sp.symbols('r R M A b', positive=True)
    f = 1-2*M/r
    g = A*(r*r-2*M*r)
    d = 2/r+2*M/(r*r*f)
    Q = 6/(r*r*f)+4*M*M/(r**4*f*f)
    assert sp.factor(sp.diff(g, r, 2)+d*sp.diff(g, r)-Q*g) == 0
    trial = b*R/r*sp.exp(-(r-R))
    assert sp.simplify(sp.diff(trial, r, 2)+2/r*sp.diff(trial, r)-trial) == 0
    a = sp.Symbol('a', positive=True)
    for n in (0, 2, 4, 6, 8):
        formula = sum(sp.binomial(n, j)*R**(n-j)*sp.factorial(j)/a**(j+1) for j in range(n+1))
        # The moment recursion avoids a large symbolic improper integral.
        if n:
            previous = sum(sp.binomial(n-1, j)*R**(n-1-j)*sp.factorial(j)/a**(j+1) for j in range(n))
            assert sp.expand(formula-R**n/a-n*previous/a) == 0
    y, p, t, q, d = [sp.Function(name)(r) for name in ('y', 'p', 't', 'q', 'd')]
    r0 = sp.diff(y, r)-p
    raw = sp.diff(p, r)+d*p-q*y
    test = sp.Function('test')(r)
    assert sp.expand(r0*(sp.diff(test, r)-t*test)-(raw-t*r0)*test
                     -(r0*sp.diff(test, r)-raw*test)) == 0
    return dict(growing_vacuum_solution=True, scalar_extension=True, exponential_moments=True,
                deformed_weak_residual_identity=True)


def sharpen_tail(obj):
    """Maximum-principle bounds from r=25 to the known decaying exterior."""
    cuts = sorted({Fraction(25), Fraction(40)} | {x for x in obj.background_knots if 25 < x < 40})
    qmin, dmin, qmax, dmax, Omax = float('inf'), float('inf'), 0., 0., 0.
    for left, right in zip(cuts[:-1], cuts[1:]):
        r, y, c, _ = obj.geometry(left, right)
        A, zeta, f = iv.mpf('.06')*y[0], y[3], c['f'].c[0]
        x = A*A/iv.mpf('.08')**2
        q = ((1-2*x)*(1-6*x)-iv.exp(2*zeta))/f
        qmin, dmin = min(qmin, lower(q)), min(dmin, lower(c['d'].c[0]))
        qmax, dmax = max(qmax, upper(q)), max(dmax, upper(c['d'].c[0]))
        Omax = max(Omax, upper(iv.exp(2*zeta)))
    assert exact(qmin) > iv.mpf('.98')**2 and dmin > 0
    assert iv.mpf('1.5')**2-iv.mpf('1.5')*exact(dmax)-exact(qmax) > 0
    assert exact(Omax) < iv.mpf('.04')
    # The inherited exterior family supplies these same comparisons for r>=40.
    f40, a40 = iv.mpf('.7169'), iv.mpf('1e-8')
    O40 = iv.mpf('.141')**2*iv.exp(iv.mpf('2e-11'))/f40
    x40 = a40*a40/iv.mpf('.08')**2
    assert (1-2*x40)*(1-6*x40)-O40 > iv.mpf('.98')**2
    assert 2-4*iv.pi*40**2*a40*a40/f40 > 0
    assert iv.mpf('1.5')**2-iv.mpf('1.5')*iv.mpf('.061')-1/f40 > 0
    assert O40 < iv.mpf('.04')
    fmin = 1-2*obj.mass/25
    assert 4*iv.pi*iv.mpf('1.65') < 21
    assert 4*iv.pi*(iv.mpf('.04')/fmin+iv.mpf('1.5')**2) < 30
    assert (iv.mpf('.04')+iv.mpf('1.5')**2+1)/2 < iv.mpf('1.65')
    anchors = {}
    for anchor in (25, 28, 30):
        i = bisect_right(obj.background_knots, Fraction(anchor))-1
        value = iv.mpf('.06')*physical_cell(obj.bg, i, anchor, anchor)[0]
        if anchor == 25:
            assert value.a > 0
        anchors[anchor] = upper(abs(value))
    amplitude40 = exact(min(upper(exact(v)*iv.exp(-iv.mpf('.98')*(40-r))) for r, v in anchors.items()))
    assert amplitude40 < iv.mpf('3e-14')
    obj.tail_anchors = anchors
    return dict(start=25, cells=len(cuts)-1, anchor_evaluations=3, scalar_potential_lower=qmin,
                scalar_potential_upper=qmax, first_derivative_coefficient_upper=dmax,
                first_derivative_coefficient_lower=dmin, local_frequency_squared_upper=Omax,
                amplitude_anchor_upper=anchors, amplitude_at_40_upper=upper(amplitude40),
                logarithmic_scalar_slope_interval=['.98', '1.5'], mass_integrand_factor='21', lapse_integrand_factor='30')


def vacuum_slope(mass, radius):
    """Legendre Q_2 associated mode; mass is an interval, not a nominal float."""
    x = radius/mass-1
    Q0 = iv.ln((x+1)/(x-1))/2
    Q1 = x*Q0-1
    Q2 = ((3*x*x-1)*Q0-3*x)/2
    Q2p = 2*(x*Q2-Q1)/(x*x-1)
    mode = 6*Q2-2*x*Q2p
    assert mode.a > 0
    return (4*Q2p-2*x*mode/(x*x-1))/(mass*mode)


def endpoint_bounds(obj, digits):
    bg, R, eps = obj.bg, iv.mpf(40), exact(.002)
    central_bound(bg)
    old_tail = tail_constants(bg, digits)
    tail = sharpen_tail(obj)
    central = central_coefficients(bg, enclosure(0, eps))
    assert central['alpha'].b < 1 and central['mass'].a > 0
    scale = iv.sqrt(4*iv.pi)*iv.mpf('.06')
    left, right = [[exact(float(v)) for v in obj.state[:, i]] for i in (0, -1)]
    yl, yr = [left[0], scale*left[2]], [right[0], scale*right[2]]
    pl, pr = [left[1], scale*left[3]], [right[1], scale*right[3]]
    B = 2*central['dbar']-12*central['mass']/central['f']
    C = max(upper(abs(B-central[k])+abs(central['cross'])) for k in ('vv', 'uu'))
    center = exact(C)*norm(yl)*eps**iv.mpf('2.5')/iv.sqrt(36)
    T, Pmin, PWmin = trace_bound(bg, eps)
    Cinner = (2/eps+4*eps*T*T)/(4*PWmin)+4*eps/Pmin
    inner = eps**2*iv.sqrt(Cinner)*norm([pl[j]-2*yl[j]/eps for j in range(2)])

    # The extension is a family indexed by the exact, fixed ADM mass.
    Ag = 1/exact(float(obj.tide.tidal_scale))**2
    M = obj.mass
    gR, gpR = Ag*(R*R-2*M*R), 2*Ag*(R-M)
    decaying_value = yr[0]-gR
    slope = vacuum_slope(M, R)
    outer_jump = norm([gpR+slope*decaying_value-pr[0], -(1+1/R)*yr[1]-pr[1]])
    Couter = (2+4*iv.mpf('.07')**2)/(4*iv.mpf('.99'))+4/iv.mpf(1100)
    outer = R*R*iv.sqrt(Couter)*outer_jump

    amp, beta, fmin = exact(tail['amplitude_at_40_upper']), iv.mpf('.98'), iv.mpf('.7169')
    assert 1-2*M/R > fmin
    assert M+4*iv.pi*iv.mpf('1.25')*R**3*amp*amp < iv.mpf('5.67')
    assert iv.mpf('5.67')/fmin < 8
    assert 6/(R*R*fmin)+iv.mpf('1.5') < 2
    assert (16/(R*R)+4*iv.pi*R*amp*amp/fmin)*(1+1/R) < 1
    # |delta m| <= 14000 amp^2 (r/R)^2 exp(-2 beta s).
    # These constants multiply amp^2 exp(-2 beta s).
    assert 14000/(R*R*fmin*fmin) < 18
    assert 4*iv.pi*iv.mpf('1.25')/fmin < 22
    assert 28000/(R*R*fmin*fmin) < 35
    assert 4*iv.pi/fmin < 18
    assert 12*14000/(R**3*fmin*fmin)+64*(18+22*R)/R**2+8*iv.pi/fmin < 78
    vbound = abs(Ag)*R*R+abs(decaying_value)
    vpbound = 2*abs(Ag)*R+3*abs(decaying_value)/(R*fmin)
    moment = lambda n, rate: sum((iv.mpf(factorial(n))/factorial(n-j)*R**(n-j)/rate**(j+1)
                                  for j in range(n+1)), iv.mpf(0))
    # ||r^2 residual||_2, componentwise triangle inequalities.
    Nv = amp*amp*(35*vpbound/R*iv.sqrt(moment(6, 4*beta))
                  +(18*vpbound/R+78*vbound/(R*R))*iv.sqrt(moment(8, 4*beta)))
    Nv += 11*amp*abs(yr[1])*R*iv.sqrt(moment(2, 2*(beta+1)))
    Nu = 11*amp*vbound/(R*R)*iv.sqrt(moment(8, 2*beta))
    Nu += 2*abs(yr[1])*R*iv.sqrt(moment(2, 2))
    exterior = norm([Nv, Nu])/2
    return dict(center_residual_upper=upper(center), inner_interface_dual_upper=upper(inner),
                outer_interface_dual_upper=upper(outer), exterior_residual_upper=upper(exterior),
                endpoint_energy_dual_upper=upper(center+inner+outer+exterior),
                exact_ADM_mass_interval=[lower(M), upper(M)], vacuum_slope_interval=[lower(slope), upper(slope)],
                exterior_metric_value_bound=upper(vbound), exterior_metric_derivative_bound=upper(vpbound),
                scalar_tail=tail, inherited_tail=old_tail)


def local_bounds(rows, energy, bridge):
    """Component traces and mean flux on I=[15.4,16.6]."""
    length = iv.mpf('1.2')
    Pmin = min(row['Pmin'] for row in rows)
    Pmax = max(row['Pmax'] for row in rows)
    PWmin = min(row['PWmin'] for row in rows)
    T = [[max(row['T'][i][j] for row in rows) for j in range(2)] for i in range(2)]
    Q = [[max(row['Q'][i][j] for row in rows) for j in range(2)] for i in range(2)]
    T, Q = [[[exact(v) for v in row] for row in matrix] for matrix in (T, Q)]
    Pmin, Pmax, PWmin = map(exact, (Pmin, Pmax, PWmin))
    r0 = [exact(max(row['residual'][i] for row in rows)) for i in range(2)]
    q = [exact(max(row['residual'][i+2] for row in rows)) for i in range(2)]
    derivative_norm = [1/iv.sqrt(Pmin)+norm(row)/iv.sqrt(4*PWmin) for row in T]
    trace = [iv.sqrt(1/(4*length*PWmin)+2*B/iv.sqrt(4*PWmin)) for B in derivative_norm]
    field = [S*energy for S in trace]
    flux = []
    for i in range(2):
        forcing = sum((Q[i][j]*field[j]+T[i][j]*r0[j] for j in range(2)), iv.mpf(0))+q[i]
        flux.append(Pmax/Pmin*(derivative_norm[i]*energy/iv.sqrt(length)+r0[i]+length*forcing/2))
    scale = iv.sqrt(4*iv.pi)*iv.mpf('.06')
    state_error = [field[0]+bridge[0], 16*flux[0], (field[1]+bridge[1])/scale, 16*flux[1]/scale]
    weights = list(map(iv.mpf, ('.003354', '.003355', '.000830', '.000004418')))
    observed = sum((a*b for a, b in zip(weights, state_error)), iv.mpf(0))
    return dict(interval=['15.4', '16.6'], P_lower=lower(Pmin), P_upper=upper(Pmax), PW_lower=lower(PWmin),
                T_entry_upper=[[upper(v) for v in row] for row in T], Q_entry_upper=[[upper(v) for v in row] for row in Q],
                trace_constants_upper=[upper(v) for v in trace], flux_errors_upper=[upper(v) for v in flux],
                integrated_trial_bridge_upper=[upper(v) for v in bridge],
                state_error_upper=[upper(v) for v in state_error], field_observation_error_upper=upper(observed))


def implementation_checks(obj):
    from polar_transverse import transverse_coefficients
    from check_polar_endpoint_budget import vacuum_slope_interval
    radii = (.003, .16, .34, 5., 11., 15.7, 16., 16.3, 26., 28.5, 35., 39.9)
    maximum = 0.
    for radius in radii:
        r = Fraction(radius)
        enclosed = obj.jet_residual(r, r, True)
        state = obj.tide.solution.sol(radius)
        prime = obj.tide.solution.sol(radius, 1)
        scale = np.sqrt(4*np.pi)*.06
        y, p = state[[0, 2]]*np.array([1., scale]), state[[1, 3]]*np.array([1., scale])
        yp, pp = prime[[0, 2]]*np.array([1., scale]), prime[[1, 3]]*np.array([1., scale])
        c = transverse_coefficients(obj.bg, np.array([radius]))
        Q = c['Q1'][0]+4*c['W'][0]*np.eye(2)
        values = np.r_[yp-p, pp+c['d'][0]*p-Q@y]
        for value, box in zip(values, enclosed):
            excess = max(0., float(box.c[0].a)-value, value-float(box.c[0].b))
            maximum = max(maximum, excess)
            assert excess < 2e-12
    rational_checks = 0
    for radius in (11., 16.):
        index = bisect_right(obj.knots, Fraction(radius))-1
        left, right = obj.knots[index:index+2]
        h, midpoint = right-left, (left+right)/2
        polynomials = []
        for j in (0, 1):
            a, b = map(lambda v: Fraction(float(v)), obj.state[j, index:index+2])
            ap, bp = map(lambda v: Fraction(float(v)), obj.slope[j, index:index+2])
            polynomials.append([a, h*ap, 3*(b-a)-h*(2*ap+bp), 2*(a-b)+h*(ap+bp)])
        coefficients0 = [(j+1)*polynomials[0][j+1]/h-polynomials[1][j] for j in range(3)]+[-polynomials[1][3]]
        jet = obj.jet_residual(midpoint, midpoint, True)[0]
        for k in range(4):
            value = sum(Fraction(factorial(j), factorial(k)*factorial(j-k))*coefficients0[j]
                        *Fraction(1, 2)**(j-k)/h**k for j in range(k, 4))
            box = exact(value)
            assert jet.c[k].a <= box.b and box.a <= jet.c[k].b
            rational_checks += 1
    slope, _ = vacuum_slope_interval(obj.bg.mass, 40, 2, iv.dps)
    enclosed_slope = vacuum_slope(obj.mass, iv.mpf(40))
    assert enclosed_slope.a <= slope.a and slope.b <= enclosed_slope.b
    Ag = 1/exact(float(obj.tide.tidal_scale))**2
    for derivative, value in ((0, Ag*(40**2-2*obj.mass*40)), (1, 2*Ag*(40-obj.mass))):
        assert abs(float(value.mid)-float(obj.tide.growing(40., derivative))) < 2e-14
    return dict(independent_residual_values=4*len(radii), maximum_float_excess=maximum,
                exact_rational_polynomial_jet_checks=rational_checks,
                vacuum_slope_recurrence=True, independent_tidal_normalization=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--endpoints-only', action='store_true')
    parser.add_argument('--sample', type=Path, help='Re-evaluate the worst cells of a prior certificate only.')
    parser.add_argument('--output', type=Path, default=ROOT/'data/tidal-field-error.json')
    args = parser.parse_args()
    iv.dps = args.digits
    start = perf_counter()
    obj = TidalEnclosure()
    algebra = algebra_checks()
    endpoints = endpoint_bounds(obj, args.digits)
    checks = implementation_checks(obj)
    if args.sample:
        old = json.loads(args.sample.read_text())
        rows = []
        for row in old['worst_cells']:
            left, right = map(Fraction, row['interval'])
            energy, _, components = obj.cell(left, right)
            rows.append(dict(interval=row['interval'], old=row['energy_squared_upper'], new=energy,
                             residual_components_upper=components))
        print(json.dumps(dict(rows=rows, elapsed_seconds=perf_counter()-start), indent=2))
        return
    report = dict(status='growing_tide_endpoints_enclosed', digits=args.digits,
                  scope='Growing quadrupolar tide on the inherited exact star; no overlap or news sign.',
                  input_sha256=obj.hashes, algebra=algebra, implementation_checks=checks, endpoints=endpoints)
    if not args.endpoints_only:
        energy, largest, local = iv.mpf(0), [], []
        bridge = [iv.mpf(0), iv.mpf(0)]
        for count, (left, right) in enumerate(zip(obj.partition[:-1], obj.partition[1:]), 1):
            E2, _, components = obj.cell(left, right)
            energy += exact(E2)
            largest.append(dict(interval=[str(left), str(right)], energy_squared_upper=E2,
                                residual_components_upper=components[:4]))
            if right <= Fraction('16.31'):
                for i in range(2):
                    bridge[i] += exact(right-left)*exact(components[i])
            if left >= Fraction('15.4') and right <= Fraction('16.6'):
                c, P = obj.last_geometry
                local.append(dict(Pmin=lower(P), Pmax=upper(P), PWmin=lower(P*c['W'].c[0]),
                                  T=[[upper(abs(v.c[0])) for v in row] for row in c['T']],
                                  Q=[[upper(abs(v.c[0])) for v in row] for row in c['Q']], residual=components))
            if perf_counter()-start > 300:
                raise RuntimeError('Tidal enclosure stopped at its 300-second budget.')
        E = iv.sqrt(energy)+exact(endpoints['endpoint_energy_dual_upper'])
        evaluations = 3*count+endpoints['scalar_tail']['cells']+endpoints['scalar_tail']['anchor_evaluations']
        assert evaluations <= 20000
        report.update(status='growing_tide_energy_error_enclosed', finite_cells=count,
                      radial_enclosure_evaluations=evaluations,
                      finite_residual_dual_upper=upper(iv.sqrt(energy)), energy_error_upper=upper(E),
                      worst_cells=sorted(largest, key=lambda x: x['energy_squared_upper'], reverse=True)[:8],
                      local_residual_components_upper=np.max([x['residual'] for x in local], axis=0).tolist(),
                      local_error=local_bounds(local, E, bridge))
    report['elapsed_seconds'] = perf_counter()-start
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256', 'worst_cells')}, indent=2))


if __name__ == '__main__':
    main()
